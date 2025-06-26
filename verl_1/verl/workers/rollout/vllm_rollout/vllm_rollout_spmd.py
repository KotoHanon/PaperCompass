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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import copy
from contextlib import contextmanager
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import vllm_version
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from verl.workers.sharding_manager import FSDPVLLMShardingManager

from transformers import AutoTokenizer
from agents.prompts import SYSTEM_PROMPTS, HINT_PROMPTS, AGENT_PROMPTS
from agents.prompts.task_prompt import formulate_input
from agents.envs.actions import ErrorAction
from typing import Dict, Any, List

from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), (
            "disable CUDA graph (enforce_eager = False) if free cache engine"
        )
        self.tokenizer = tokenizer
        self.image_limit = self.config.agent.get("image_limit", 10)  # default to 10 images
        self.length_limit = self.config.agent.get("length_limit", 32)  # default to 32k tokens
        self.dataset = self.config.agent.get("dataset", None)
        self.agent_method = self.config.agent.get("agent_method", "neusym_rag")
        self.max_turns = self.config.agent.get("num_turns", 20)
        self.window_size = self.config.agent.get("window_size", 5)  # default to 5 turns

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            if vllm_version in (
                "0.5.4",
                "0.6.3",
            ):
                train_tp = kwargs.get("train_tp")
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(
                    tensor_model_parallel_size=tensor_parallel_size, num_tp_per_train_tp=num_tp_per_train_tp
                )
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, (
            "model context length should be greater than total sequence length"
        )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        limit_mm_per_prompt = None
        if config.get("limit_images", None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != "0.3.1":
            kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, env: AgentEnv, output_kwargs: Dict[str, Any], **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.init_cache_engine()

        agent_prompt = AGENT_PROMPTS[self.agent_method].format(
            system_prompt=SYSTEM_PROMPTS[self.agent_method],
            action_space_prompt=env.action_space_prompt,
            hint_prompt=HINT_PROMPTS[self.agent_method],
            max_turn=self.max_turns
        )

        prompt_ids = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        prompt_mask = prompts.batch["attention_mask"]
        prompt_position_ids = prompts.batch["position_ids"]
        non_tensor_batch = prompts.non_tensor_batch

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = prompt_ids.size(0)

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }
        
        active_mask = [True] * batch_size
        obss = [[] for _ in range(batch_size)]
        prompt_completion_ids_list = [[] for _ in range(batch_size)]
        completion_ids_list = [[] for _ in range(batch_size)]
        initial_prompt_ids = None
        initial_prompt_mask = None

        # generate message
        prompt_texts = self.tokenizer.batch_decode(
            prompt_ids, skip_special_tokens=True)
        messages = []
        for idx, example in enumerate(prompt_texts):
            task_prompt, image_messages = formulate_input(self.dataset, example, use_pdf_id=True)
            if image_messages:
                task_prompt = [{'type': 'text', 'text': task_prompt}] + image_messages
            messages.append([
                {'role': 'system', 'content': agent_prompt},
                {'role': 'user', 'content': task_prompt}
            ])

        for turn in range(self.max_turns):

            prompt_texts = []
            for message in messages:
                if len(message) > (self.window_size + 1) * 2: # each turn has two messages from assistant and user, respectively
                    current_message = message[:2] + message[-self.window_size * 2:]
                else: current_message = message
            try:
                formatted_message = self.convert_message_from_gpt_format(messages=current_message)
                prompt_texts.append(formatted_message)
            except Exception as e:
                default_prompt = "system: An error occurred while processing the previous turn."
                prompt_texts.append(default_prompt)
                logger.warning(f"Error in converting message: {e}")
            
            prompt_inputs = self.tokenizer(
                text=prompt_texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            )
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, prompt_ids[i]) for i in range(batch_size)], dtype=object
            )

            if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
                raise RuntimeError("vllm sharding manager is not work properly.")

            if "multi_modal_data" in non_tensor_batch:
                vllm_inputs = []
                for raw_prompt_ids, multi_modal_data in zip(
                    non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
                ):
                    vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
            else:
                vllm_inputs = [
                    {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
                ]

            # ensure the type of `prompt_token_ids` passed to vllm is list[int]
            # https://github.com/volcengine/verl/pull/772
            for input_data in vllm_inputs:
                if isinstance(input_data["prompt_token_ids"], np.ndarray):
                    input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
                elif not isinstance(input_data["prompt_token_ids"], list):
                    raise TypeError(
                        f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                    )

            if self.max_prompt_length is not None:
                prompt_ids = prompt_ids[:, -self.max_prompt_length:]
                prompt_mask = prompt_mask[:, -self.max_prompt_length:]
                if turn == 0:
                    initial_prompt_ids = prompt_ids
                    initial_prompt_mask = prompt_mask

            # users can customize different sampling_params at different run
            with self.update_sampling_params(**kwargs):
                outputs = self.inference_engine.generate(
                    prompts=vllm_inputs,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )

                # TODO(sgm): disable logprob when recompute_log_prob is enable
                # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

                response = []
                for output in outputs:
                    for sample_id in range(len(output.outputs)):
                        response.append(output.outputs[sample_id].token_ids)

                response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                    prompt_ids.device
                )

                if self.sampling_params.n > 1 and do_sample:
                    prompt_ids = _repeat_interleave(prompt_ids, self.sampling_params.n)
                    prompt_mask = _repeat_interleave(prompt_mask, self.sampling_params.n)
                    prompt_position_ids = _repeat_interleave(prompt_position_ids, self.sampling_params.n)
                    batch_size = batch_size * self.sampling_params.n
                    if "multi_modal_inputs" in non_tensor_batch.keys():
                        non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
                            non_tensor_batch["multi_modal_inputs"], self.sampling_params.n
                        )

                #seq = torch.cat([prompt_ids, response], dim=-1)
            
            response_texts = self.tokenizer.batch_decode(
                response, skip_special_tokens=True
            )

            active_mask_copy = active_mask.copy()

            for idx in range(batch_size):
                if active_mask[idx]:
                    response_text = response_texts[idx]
                    obs, _, flag, _ = env.step(response_text, **output_kwargs)
                    obss[idx].append(obs)
                    if flag:
                        active_mask_copy[idx] = False
                else:
                    obss[idx].append(None)
                    response[idx] = torch.full_like(response[idx], self.pad_token_id) # padding with pad_token_id
            
            active_num = sum(active_mask_copy)
            active_idx = [idx for idx, mask in enumerate(active_mask_copy) if mask]
            actions: List[Action] = env.parsed_actions[-active_num:] # we only have active_num actions in the last turn, which is less than batch_size

            if turn == 0:
                for idx in range(batch_size):
                    prompt_completion_ids_list[idx].append(response[idx])
                    completion_ids_list[idx].append(response[idx])
            
            else:
                for idx in range(batch_size):
                    if idx not in active_idx: continue
                    if isinstance(actions[active_idx.index(idx)], ErrorAction) is False: # append if the action is not an instance of ErrorAction
                        prompt_completion_ids_list[idx].append(response[idx])
                        completion_ids_list[idx].append(response[idx])
            
            action_msgs = []
            obs_msgs = []
            
            for idx in range(batch_size):
                if obss[idx][-1] is not None: # the idx sample has a valid observation
                    obs_msgs.append(obss[idx][-1].convert_to_message())
                else:
                    obs_msgs.append(None) # padding
                
                if idx not in active_idx:
                    action_msgs.append(None) # padding
                else:
                    action_msgs.append(actions[active_idx.index(idx)].convert_to_message(env.action_format, env.interact_protocol))
            
            for idx, (action_msg, obs_msg) in enumerate(zip(action_msgs, obs_msgs)):
                # we clean the None messages in crop_image_count_in_messages
                messages[idx].append(action_msg)
                messages[idx].append(obs_msg)
        
        # keep the last valid observation for each sample as the prev_answer
        real_obss = [] # [batch_size]
        for idx_1 in range(batch_size):
            for idx_2 in reversed(range(len(obss[idx_1]))):
                if obss[idx_1][idx_2] is not None:
                    real_obss.append(obss[idx_1][idx_2])
                    break
        
        for idx in range(batch_size):
            if len(prompt_completion_ids_list[idx]) >= 10:
                prompt_completion_ids_list[idx] = prompt_completion_ids_list[idx][-10:]
                completion_ids_list[idx] = completion_ids_list[idx][-10:]
            
            prompt_completion_ids_list[idx].insert(0, initial_prompt_ids[idx]) # insert the initial prompt at the beginning

            prompt_completion_ids_list[idx] = torch.cat(prompt_completion_ids_list[idx], dim=0)
            completion_ids_list[idx] = torch.cat(completion_ids_list[idx], dim=0)

        prompt_completion_ids = pad_sequence(prompt_completion_ids_list, batch_first=True, padding_value=self.pad_token_id, padding_side='left')
        completion_ids = pad_sequence(completion_ids_list, batch_first=True, padding_value=self.pad_token_id, padding_side='right')
        # only keep the last EOS token in each sample, replace others with PAD token
        def replace_all_but_last_eos(tensor, eos_token_id, pad_token_id):
            if tensor.dim() == 2 and tensor.size(0) == batch_size:  # [batch_size, seq_len]
                for batch_idx in range(batch_size):
                    eos_positions = (tensor[batch_idx] == eos_token_id).nonzero(as_tuple=True)[0]
                    if len(eos_positions) > 1: 
                        for pos in eos_positions[:-1]:
                            tensor[batch_idx, pos] = pad_token_id
            return tensor
        
        if prompt_completion_ids.numel() > 0:
            seq = replace_all_but_last_eos(
                prompt_completion_ids, 
                eos_token_id, 
                self.pad_token_id
            )
        
        if completion_ids.numel() > 0:
            response = replace_all_but_last_eos(
                completion_ids, 
                eos_token_id, 
                pad_token_id
            )

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=prompt_position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if prompt_position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=prompt_mask.dtype
        )
        attention_mask = torch.cat((prompt_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # free vllm cache engine
        if (
            vllm_version
            in (
                "0.5.4",
                "0.6.3",
            )
            and self.config.free_cache_engine
        ):
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def crop_image_count_in_messages(
        self,
        messages: List[Dict[str, Any]],
        image_limit: int = 10,
        keep_msg: int = 2,
        in_place: bool = False
        ) -> List[Dict[str, Any]]:
        """ Crop the image count in the messages.
        @param
            messages: the messages to be cropped.
            image_limit: the maximum number of images to be kept.
            keep_msg: the number of preceding messages to keep the images.
            in_place: whether to modify the messages in place.
        @return
            the cropped messages.
        """
        image_count = 0
        if not in_place: messages = copy.deepcopy(messages)

        messages = [msg for msg in messages if msg is not None] 

        # images in the first two messages are maintained in the original order (usually system/task prompt)
        for i in range(min(keep_msg, len(messages))):
            if isinstance(messages[i]['content'], list):
                for msg in messages[i]['content']:
                    if msg['type'] == 'image_url':
                        image_count += 1
                        if image_count > image_limit:
                            msg['type'] = 'text'
                            if 'image_url' in msg: del msg['image_url']
                            msg['text'] = f'The image stream is omitted due to the incapability of handling >{image_limit} images.'

        # images in the rest messages are preserved in the reverse order
        for msg in reversed(messages[keep_msg:]):
            if isinstance(msg['content'], list):
                for msg_dict in msg['content'][::-1]:
                    if msg_dict['type'] == 'image_url':
                        image_count += 1
                        if image_count > image_limit:
                            msg_dict['type'] = 'text'
                            if 'image_url' in msg_dict: del msg_dict['image_url']
                            msg_dict['text'] = f'The image stream is omitted due to the incapability of handling >{image_limit} images.'
        return messages

    def convert_message_from_gpt_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """ For VLLM-deployed open-source models, there are some limitations on:
        1. the input prompt length (e.g., 32k tokens)
        2. the number of images in the prompt (e.g., only one image)
        """
        keep_msg = 2 # one system message and one task message
        messages = self.crop_image_count_in_messages(messages=messages, image_limit=self.image_limit, keep_msg=keep_msg, in_place=True)

        if len(messages) > keep_msg:
            message_max_tokens = min(self.length_limit * 1000, self.tokenizer.model_max_length) # by default, qwen2.5-72b-instruct is 32k

            truncated_messages = messages[:keep_msg]
            current_tokens = sum(len(self.tokenizer.encode(str(message))) for message in truncated_messages)
            for i in range(len(messages) - 1, keep_msg - 1, -2):
                pair = messages[i-1:i+1]
                pair_tokens = sum(len(self.tokenizer.encode(str(message))) for message in pair)
                if current_tokens + pair_tokens > message_max_tokens:
                    break
                truncated_messages.insert(keep_msg, pair[1])
                truncated_messages.insert(keep_msg, pair[0])
                current_tokens += pair_tokens
            messages = truncated_messages

        return messages


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, *args, **kwargs):
        # Engine is deferred to be initialized in init_worker
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager: FSDPVLLMShardingManager = None
        self.is_sleep = False

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is intialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
