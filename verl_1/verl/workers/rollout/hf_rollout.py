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
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single GPU model.
Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model to perform generation.
"""

import contextlib

import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig
from transformers import AutoTokenizer
from agents.prompts import SYSTEM_PROMPTS, HINT_PROMPTS, AGENT_PROMPTS
from typing import Dict, Any, List
from agents.envs import AgentEnv

from verl import DataProto
from verl.utils.torch_functional import get_response_mask

from .base import BaseRollout

from agents.prompts.task_prompt import formulate_input

__all__ = ["HFRollout"]


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.critic.model.get("tokenizer_path", "gpt2"))
        self.image_limit = self.config.agent.get("image_limit", 10)  # default to 10 images
        self.length_limit = self.config.agent.get("length_limit", 32)  # default to 32k tokens
        self.dataset = self.config.agent.get("dataset", None)
        self.agent_method = self.config.agent.get("agent_method", "neusym_rag")
        self.max_turns = self.config.agent.get("num_turns", 20)

    # TODO: transfer the two params: # `env` and `output_kwargs` to the generate_sequences method.
    def generate_sequences(self, prompts: DataProto, env: AgentEnv, output_kwargs: Dict[str, Any] = {}) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto, env: AgentEnv, output_kwargs: Dict[str, Any] = {}) -> DataProto:
        agent_prompt = AGENT_PROMPTS[self.agent_method].format(
            system_prompt=SYSTEM_PROMPTS[self.agent_method],
            action_space_prompt=env.action_space_prompt,
            hint_prompt=HINT_PROMPTS[self.agent_method],
            max_turn=max_turn
        )

        prompt_ids = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        prompt_position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        # make sampling args can be overriden by inputs
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = prompts.meta_info.get("top_k", self.config.get("top_k", 0))

        if top_k is None:
            top_k = 0
        top_k = max(0, top_k)  # to be compatible with vllm

        temperature = prompts.meta_info.get("temperature", self.config.temperature)

        generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

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

        for turn in range(self.config.get("num_turns", 1)):

            prompt_texts = []
            for message in messages:
                if len(message) > (window_size + 1) * 2: # each turn has two messages from assistant and user, respectively
                    current_message = message[:2] + message[-window_size * 2:]
                else: current_message = message
            try:
                formatted_message = convert_message_from_gpt_format(messages=current_message)
                prompt_texts.append(formatted_message)
            except Exception as e:
                default_prompt = "system: An error occurred while processing the previous turn."
                prompt_texts.append(default_prompt)
                logger.warning(f"Error in converting message: {e}")
            
            prompt_inputs = self.tokenizer(
                text=prompt_texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            )
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

            if self.max_prompt_length is not None:
                prompt_ids = prompt_ids[:, -self.max_prompt_length:]
                prompt_mask = prompt_mask[:, -self.max_prompt_length:]
                if turn == 0:
                    initial_prompt_ids = prompt_ids
                    initial_prompt_mask = prompt_mask

            if isinstance(self.module, FSDP):
                # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
                param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
            with param_ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.module.generate(
                    input_ids=idx,
                    attention_mask=prompt_mask,
                    do_sample=do_sample,
                    max_new_tokens=response_length,
                    # max_length=max_length,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    generation_config=generation_config,
                    # renormalize_logits=True,
                    output_scores=False,  # this is potentially very large
                    return_dict_in_generate=True,
                    use_cache=True,
                )
            # TODO: filter out the seq with no answers like ds-chat
            seq = output.sequences

            # huggingface generate will stop generating when all the batch reaches [EOS].
            # We have to pad to response_length
            sequence_length = prompt_length + self.config.response_length
            delta_length = sequence_length - seq.shape[1]

            if delta_length > 0:
                delta_tokens = torch.ones(size=(batch_size, delta_length), device=seq.device, dtype=seq.dtype)
                delta_tokens = pad_token_id * delta_tokens
                seq = torch.cat((seq, delta_tokens), dim=1)

            assert seq.shape[1] == sequence_length

            prompt = seq[:, :prompt_length]  # (bs, prompt_length)
            response = seq[:, prompt_length:]  # (bs, response_length)

            response_texts = self.tokenizer.batch_decode(
                response, skip_special_tokens=True
            )
            response_length = response.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

            response_position_ids = position_ids[:, -1:] + delta_position_id
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

            response_attention_mask = get_response_mask(
                response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
            )
            attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

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
                    response[idx] = torch.full_like(response[idx], pad_token_id) # padding with pad_token_id
                    seq[idx, prompt_length:] = response[idx]
            
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
        
        prompt_completion_ids = pad_sequence(prompt_completion_ids_list, batch_first=True, padding_value=pad_token_id, padding_side='left')
        completion_ids = pad_sequence(completion_ids_list, batch_first=True, padding_value=pad_token_id, padding_side='right')
            
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
            prompt_completion_ids = replace_all_but_last_eos(
                prompt_completion_ids, 
                eos_token_id, 
                pad_token_id
            )
        
        if completion_ids.numel() > 0:
            completion_ids = replace_all_but_last_eos(
                completion_ids, 
                eos_token_id, 
                pad_token_id
            )
        
            # we pad for: 1. everyone after the first eos token, and 2. pad_token_id for the rest
        is_eos = completion_ids == eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1)).expand(is_eos.size(0), -1)
        eos_position_mask = sequence_indices <= eos_idx.unsqueeze(1)
        non_padding_mask = completion_ids != pad_token_id
        combined_mask = eos_position_mask & non_padding_mask
        completion_mask = combined_mask.int()

        attention_prompt_mask = prompt_mask
        attention_completion_mask = completion_mask.clone()

        prompt_completion_len = prompt_completion_ids.size(1)
        prompt_len = initial_prompt_ids.size(1)
        completion_len = completion_ids.size(1)

        assert prompt_completion_len == prompt_len + completion_len, \
            f'Prompt completion length {prompt_completion_len} does not match the sum of prompt length {prompt_len} and completion length {completion_len}.'

        # concatenate the attention mask
        attention_mask = torch.cat([attention_prompt_mask, attention_completion_mask], dim=1)


        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": completion_ids,
                "input_ids": prompt_completion_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # empty cache before compute old_log_prob
        torch.cuda.empty_cache()

        self.module.train()
        return DataProto(batch=batch)
    
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
        messages = crop_image_count_in_messages(messages=messages, image_limit=self.image_limit, keep_msg=keep_msg, in_place=True)

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
