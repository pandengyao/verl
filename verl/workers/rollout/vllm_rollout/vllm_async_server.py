# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import os
import pickle
from typing import Any, Callable, Optional

import numpy as np
import ray
import zmq
from omegaconf import DictConfig, ListConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from vllm import SamplingParams
from vllm.config import CompilationConfig, CompilationLevel
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor
from vllm.worker.worker_base import WorkerWrapperBase

from verl.utils import hf_processor
from verl.utils.fs import copy_to_local
from verl.workers.rollout.async_server import AsyncServerBase, TokenOutput

logger = logging.getLogger(__file__)


def _get_model_runner_workers(vllm_config, init_ray: bool = True):
    assert vllm_config.instance_id is not None, "instance_id must be set for external ray actors."

    fields = vllm_config.instance_id.split(":")
    assert len(fields) == 4, (
        f"instance_id: {vllm_config.instance_id} must be in the format of "
        f"<namespace>:<wg_prefix>:<vllm_dp_size>:<vllm_dp_rank>."
    )
    namespace, wg_prefix, vllm_dp_size, vllm_dp_rank = fields[0], fields[1], int(fields[2]), int(fields[3])

    # Make sure subprocess in same namespace as parent actor.
    # actor name format: {name_prefix}WorkerDict_{pg_idx}:{local_rank}
    if init_ray:
        ray.init(namespace=namespace)
    actor_names = [
        actor_name for actor_name in ray.util.list_named_actors() if actor_name.startswith(f"{wg_prefix}WorkerDict")
    ]

    vllm_tp_size = vllm_config.parallel_config.tensor_parallel_size
    assert len(actor_names) == vllm_dp_size * vllm_tp_size, (
        f"instance_id: {vllm_config.instance_id} has {len(actor_names)} actors, but vllm_dp_size: "
        f"{vllm_dp_size} * vllm_tp_size: {vllm_tp_size} = {vllm_dp_size * vllm_tp_size} is expected."
    )

    def get_pg_index_and_local_rank(actor_name) -> tuple[int, int]:
        fields = actor_name.split(":")
        assert len(fields) == 2, f"invalid actor name: {actor_name}"
        pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])
        return pg_index, local_rank

    # sort actor names by pg_index and local_rank
    actor_names = sorted(actor_names, key=get_pg_index_and_local_rank)
    actor_names = actor_names[vllm_dp_rank * vllm_tp_size : (vllm_dp_rank + 1) * vllm_tp_size]
    workers: list[WorkerWrapperBase] = [ray.get_actor(actor_name) for actor_name in actor_names]
    print(f"instance_id: {vllm_config.instance_id} initializes with external actors: {actor_names}")

    return workers


class ExternalRayDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        self.workers = _get_model_runner_workers(vllm_config=self.vllm_config, init_ray=True)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")
        print(f"instance_id: {self.vllm_config.instance_id} initializes finished.")

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        # TODO(wuxibin): support ray compiled graph
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = pickle.dumps(method)
        del method

        # ~3ms overhead per schedule step due to SchedulerOutput/ModelRunnerOutput serialization/deserialization.
        outputs = ray.get(
            [worker.execute_method.remote(sent_method, *args, **(kwargs or {})) for worker in self.workers]
        )
        return outputs

    def check_health(self):
        return


class ExternalZeroMQDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        addresses = os.environ["VERL_VLLM_ZMQ_ADDRESSES"].split(",")
        self.context = zmq.Context()
        self.sockets = []
        for address in addresses:
            socket = self.context.socket(zmq.REQ)
            socket.connect(address)
            self.sockets.append(socket)

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = pickle.dumps(method)
        del method

        message = pickle.dumps((sent_method, args, kwargs or {}))
        for socket in self.sockets:
            socket.send(message, zmq.DONTWAIT)

        outputs = []
        for socket in self.sockets:
            outputs.append(pickle.loads(socket.recv()))
        return outputs

    def check_health(self):
        return


@ray.remote(num_cpus=1)
class AsyncvLLMServer(AsyncServerBase):
    """
    AsyncvLLMServer is a wrapper for AsyncLLM, it uses ExternalRayDistributedExecutor to launch engines
    in hybrid rollout workers, i.e AsyncActorRolloutRefWorker.

    AsyncvLLMServer works as follows:
    1. Start FastAPI server first.
    2. Initialize AsyncLLM with ExternalRayDistributedExecutor.
    3. AsyncLLM spawn EngineCore in subprocess.
    4. EngineCore initialize ExternalRayDistributedExecutor.
    5. ExternalRayDistributedExecutor lookup its corresponding actors by name.
    6. ExternalRayDistributedExecutor init executor: init_worker, init_device, load_model.

    For vLLM AsyncLLM design, see: https://github.com/vllm-project/vllm/pull/9826
    """

    def __init__(self, config: DictConfig, vllm_dp_size: int, vllm_dp_rank: int, wg_prefix: str):
        """
        Args:
            config: DictConfig.
            vllm_dp_size: int, vllm data parallel size.
            vllm_dp_rank: int, vllm data parallel rank.
            wg_prefix: str, worker group prefix, used to lookup actors.
        """
        super().__init__()

        self.config = config.actor_rollout_ref
        self.vllm_dp_size = vllm_dp_size
        self.vllm_dp_rank = vllm_dp_rank
        self.wg_prefix = wg_prefix
        self.engine: AsyncLLM = None

    async def init_engine(self):
        """Init vLLM AsyncLLM engine."""
        config = self.config
        model_path = config.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(model_path)
        trust_remote_code = config.model.get("trust_remote_code", False)
        config = config.rollout

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)
        max_model_len = config.max_model_len if config.max_model_len else config.prompt_length + config.response_length
        self.max_model_len = int(max_model_len)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        kwargs = dict(
            n=1,
            logprobs=0,
            repetition_penalty=1.0,
            max_new_tokens=config.response_length,
        )
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        print(f"override_generation_config: {kwargs}")

        backend = os.environ.get("VERL_VLLM_DISTRIBUTED_BACKEND", "zeromq")
        if backend == "zeromq":
            distributed_executor_backend = ExternalZeroMQDistributedExecutor
        elif backend == "ray":
            distributed_executor_backend = ExternalRayDistributedExecutor
        else:
            distributed_executor_backend = None

        compilation_config = {}

        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        # enforce_eager must be False to use cudagraph
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_config["compilation_config"] = CompilationConfig(
                    level=CompilationLevel.PIECEWISE, cudagraph_capture_sizes=cudagraph_capture_sizes
                )
            else:
                logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

        engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}

        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        engine_args = AsyncEngineArgs(
            model=local_path,
            enable_sleep_mode=config.free_cache_engine,
            override_generation_config=kwargs,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=distributed_executor_backend,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=self.max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format="dummy" if config.load_format.startswith("dummy") else config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **compilation_config,
            **engine_kwargs,
        )

        # init async llm engine
        vllm_config = self._create_engine_config(engine_args)
        self.engine = AsyncLLM.from_vllm_config(vllm_config)

        # build serving chat
        model_config = self.engine.model_config
        BASE_MODEL_PATHS = [BaseModelPath(name=model_name, model_path=model_path)]
        models = OpenAIServingModels(self.engine, model_config, BASE_MODEL_PATHS)
        self.openai_serving_chat = OpenAIServingChat(
            self.engine,
            model_config,
            models,
            "assistant",
            request_logger=RequestLogger(max_log_len=4096),
            chat_template=None,
            chat_template_content_format="auto",
            enable_auto_tools=config.multi_turn.tool_config_path is not None,
            tool_parser=config.multi_turn.format,  # hermes, llama3_json, ...
        )

        # used for Qwen2.5-VL
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    def _create_engine_config(self, engine_args: AsyncEngineArgs):
        vllm_config = engine_args.create_engine_config()
        namespace = ray.get_runtime_context().namespace
        vllm_config.instance_id = f"{namespace}:{self.wg_prefix}:{self.vllm_dp_size}:{self.vllm_dp_rank}"

        # VERL_VLLM_ZMQ_ADDRESSES
        if engine_args.distributed_executor_backend == ExternalZeroMQDistributedExecutor:
            self.workers = _get_model_runner_workers(vllm_config=vllm_config, init_ray=False)
            zmq_addresses = ray.get([worker.get_zeromq_address.remote() for worker in self.workers])
            print(f"VERL_VLLM_ZMQ_ADDRESSES: {zmq_addresses}")
            os.environ["VERL_VLLM_ZMQ_ADDRESSES"] = ",".join(zmq_addresses)

        return vllm_config

    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        request_json = await raw_request.json()
        request = ChatCompletionRequest(**request_json)
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        max_tokens = self.max_model_len - len(prompt_ids)
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", self.config.rollout.get("repetition_penalty", 1.0))
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt_ids = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.processor)
        prompt = TokensPrompt(
            prompt_token_ids=prompt_ids, multi_modal_data={"image": image_data} if image_data else None
        )
        generator = self.engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)
        # Get final response
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            #[AsyncvLLMServer.generate] output: RequestOutput(request_id=acb7572fe62d48eab8eac24cc6b08fc8, prompt=None, prompt_token_ids=[151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 2016, 672, 27896, 27627, 429, 1059, 12534, 6837, 1410, 2167, 990, 264, 12963, 24527, 2473, 13, 2932, 27627, 279, 1850, 1616, 311, 6757, 374, 553, 279, 27287, 13, 8886, 27287, 7049, 220, 17, 20, 30191, 311, 70698, 13, 1913, 279, 1156, 1899, 11, 220, 20, 1251, 389, 72861, 3697, 553, 311, 633, 2176, 30700, 69854, 13, 220, 18, 1251, 3697, 553, 311, 633, 678, 862, 489, 26165, 30700, 69854, 13, 17375, 11, 825, 1697, 3520, 3697, 553, 389, 264, 650, 26165, 13, 2585, 1657, 11192, 1521, 1340, 1281, 429, 1899, 30, 6771, 594, 1744, 3019, 553, 3019, 323, 2550, 279, 1590, 4226, 1283, 330, 820, 3263, 151645, 198, 151644, 77091, 198], encoder_prompt=None, encoder_prompt_token_ids=None, prompt_logprobs=None, outputs=[CompletionOutput(index=0, text='To solve this problem, we need to calculate the total cost of inflating the tires and then add the', token_ids=[1249, 11625, 419, 3491, 11, 582, 1184, 311, 11047, 279, 2790, 2783, 315, 4601, 1095, 279, 30700, 323, 1221, 912, 279], cumulative_logprob=-5.10562704241601, logprobs=[{1249: Logprob(logprob=-0.006476958282291889, rank=1, decoded_token='To')}, {11625: Logprob(logprob=-0.8980275392532349, rank=1, decoded_token=' solve')}, {419: Logprob(logprob=-0.0037878446746617556, rank=1, decoded_token=' this')}, {3491: Logprob(logprob=-0.06232811510562897, rank=1, decoded_token=' problem')}, {11: Logprob(logprob=-0.00023314618738368154, rank=1, decoded_token=',')}, {582: Logprob(logprob=-0.011114707216620445, rank=1, decoded_token=' we')}, {1184: Logprob(logprob=-0.007258828263729811, rank=1, decoded_token=' need')}, {311: Logprob(logprob=-1.1920928244535389e-07, rank=1, decoded_token=' to')}, {11047: Logprob(logprob=-0.3296605348587036, rank=1, decoded_token=' calculate')}, {279: Logprob(logprob=-0.014150755479931831, rank=1, decoded_token=' the')}, {2790: Logprob(logprob=-0.04398912191390991, rank=1, decoded_token=' total')}, {2783: Logprob(logprob=-0.4354758560657501, rank=1, decoded_token=' cost')}, {315: Logprob(logprob=-0.05776512250304222, rank=1, decoded_token=' of')}, {4601: Logprob(logprob=-0.2467779964208603, rank=1, decoded_token=' infl')}, {1095: Logprob(logprob=-0.000568228424526751, rank=1, decoded_token='ating')}, {279: Logprob(logprob=-0.3104530870914459, rank=1, decoded_token=' the')}, {30700: Logprob(logprob=-0.03183897212147713, rank=1, decoded_token=' tires')}, {323: Logprob(logprob=-0.8176729083061218, rank=1, decoded_token=' and')}, {1221: Logprob(logprob=-0.18666093051433563, rank=1, decoded_token=' then')}, {912: Logprob(logprob=-1.0203628540039062, rank=1, decoded_token=' add')}, {279: Logprob(logprob=-0.621023416519165, rank=1, decoded_token=' the')}], finish_reason=None, stop_reason=None)], finished=False, metrics=None, lora_request=None, num_cached_tokens=16, multi_modal_placeholders={})
            # print(f"[AsyncvLLMServer.generate] output: {output}") 
            final_res = output
        assert final_res is not None

        token_ids = final_res.outputs[0].token_ids
        log_probs = None
        if sampling_params.logprobs is not None:
            log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(final_res.outputs[0].logprobs)]
        
        result = TokenOutput(token_ids=token_ids, log_probs=log_probs)
        #[AsyncvLLMServer.generate] result: token_ids=[1249, 11625, 419, 11, 582, 1184, 311, 975, 28412, 504, 279, 1995, 2661, 1447, 16, 13, 86786, 374, 220, 22, 7541, 16217, 624, 17, 13, 69999, 374, 1378, 7541, 23327, 1091, 86786, 11, 773, 69999, 374, 220, 22, 481, 220, 17, 284, 220, 20, 7541, 16217, 624, 18, 13, 21998, 374, 825, 4478, 49909, 1091, 69999, 11, 773, 21998, 374, 220, 20, 488, 220, 16, 284, 220, 21, 7541, 16217, 382, 54815, 11, 21998, 594, 62235, 374, 220, 21, 7541, 1293, 382, 820, 220, 21, 198, 73594, 12669, 198, 21, 198, 73594, 151645] log_probs=[-0.02655377797782421, -0.0035339067690074444, -0.006755015812814236, -0.4292033910751343, -0.14289677143096924, -0.04100458696484566, 0.0, -1.583308458328247, -0.35821279883384705, -0.016901502385735512, -0.5044326782226562, -0.39609551429748535, -0.028039496392011642, -0.44026267528533936, -0.00020656836568377912, -0.0011974553344771266, -0.03701769560575485, -0.01000862568616867, -0.0021540552843362093, -0.0031963707879185677, -3.838465272565372e-05, -7.1403817855753e-05, -0.02910926565527916, -0.004683121107518673, -5.722029527532868e-06, -0.0694093182682991, -0.002138829091563821, -0.22814695537090302, -3.4450891689630225e-05, -0.012102387845516205, -1.6689286894688848e-06, -0.0005770448478870094, -0.22395852208137512, -0.000542493537068367, -0.014941448345780373, -0.05660134553909302, -0.5407248139381409, -0.14519046247005463, -0.05598834529519081, -3.6954811548639555e-06, -1.4305104514278355e-06, -6.151010165922344e-05, -3.516612196108326e-05, -4.768370445162873e-07, -1.0967194612021558e-05, -0.00041059168870560825, -0.000756216119043529, -4.386805812828243e-05, -1.4305104514278355e-06, -0.7143080830574036, -0.028131874278187752, -0.032612692564725876, -0.0015983913326635957, -0.002217336092144251, -1.0728830375228426e-06, -2.8967437174287625e-05, -0.0009396428358741105, -3.2305197237292305e-05, -0.000581572181545198, -0.006168138235807419, -4.8040190449682996e-05, -4.911301948595792e-05, -0.0005504761938937008, -3.576278118089249e-07, -1.0728830375228426e-06, -1.311301275563892e-06, -3.611976353568025e-05, -2.3841830625315197e-06, -8.583032467868179e-06, -0.00015841660206206143, -0.6939682960510254, -0.30478528141975403, -0.0001740304142003879, -0.02484528161585331, -0.014247246086597443, -3.576214658096433e-05, -0.05881717801094055, -0.0034085765946656466, -0.0005758534534834325, -5.173549288883805e-05, -0.005892643239349127, -0.0013373488327488303, -0.0030121691524982452, -5.006777428206988e-06, -2.3245540432981215e-05, -1.7881233361549675e-05, -0.0015216212486848235, -0.0032286918722093105, -0.000783732277341187, -0.0004508670826908201, -3.659658250398934e-05, -0.004780411254614592, -0.004354993812739849]
        print(f"[AsyncvLLMServer.generate] result: {result}")
        
        return result

    async def wake_up(self):
        if self.config.rollout.free_cache_engine:
            await asyncio.gather(*[worker.wake_up.remote() for worker in self.workers])

    async def sleep(self):
        # TODO: https://github.com/vllm-project/vllm/issues/17103
        await self.engine.reset_prefix_cache()
        if self.config.rollout.free_cache_engine:
            await asyncio.gather(*[worker.sleep.remote() for worker in self.workers])


def _qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor):
    """Deduplicate consecutive image tokens in prompt_ids for Qwen2.5-VL, since vLLM will replicate the
    <|image_pad|> token by image_data.

    For example,
    ```
    <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
    =>
    <|vision_start|><|image_pad|><|vision_end|>
    ```
    """
    if processor is not None and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        prompt_ids = np.array(prompt_ids)

        # Create a mask where True indicates elements to keep
        mask = np.ones(len(prompt_ids), dtype=bool)

        # Find where the array equals the value
        is_value = prompt_ids == processor.image_token_id

        # Find consecutive duplicates by checking if previous element is also the value
        mask[1:] &= ~(is_value[1:] & is_value[:-1])

        return prompt_ids[mask].tolist()
    else:
        return prompt_ids
