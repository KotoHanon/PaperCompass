import collections
import os
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Sized
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union, List, Dict, Tuple
import copy
import datasets
import torch
import torch.utils.data
import transformers
from torch.nn.utils.rnn import pad_sequence
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_peft_available

from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..extras.profiling import profiling_context, profiling_decorator
from ..extras.vllm_client import VLLMClient
from ..import_utils import is_liger_kernel_available, is_rich_available, is_vllm_available
from ..models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from .callbacks import SyncRefModelCallback
from .grpo_config import GRPOConfig
from .utils import (
    disable_dropout_in_model,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)

from agents.envs.actions import ErrorAction

from agents.envs import infer_env_class, AgentEnv
from agents.models import infer_model_class, LLMClient
from agents.frameworks import infer_agent_class, AgentBase
from utils.eval_utils import evaluate, print_result, load_test_data, write_jsonl
from utils.hyperparam_utils import parse_args, get_result_folder, get_result_logger
from agents.prompts import convert_database_schema_to_prompt, convert_vectorstore_schema_to_prompt
from agents.envs.actions import Action, Observation
from utils.functions.common_functions import truncate_tokens
from agents.prompts import SYSTEM_PROMPTS, HINT_PROMPTS, AGENT_PROMPTS
from agents.prompts.task_prompt import formulate_input
import logging, json, tiktoken

def adaption_layer(trainer, inputs, prompt_ids, prompt_mask, window_size: int = 3, output_path: Optional[str] = None, output_kwargs: Dict[str, Any] = {}, use_consistency_selection: bool = False, consistency_N: int = 5, logger: logging.Logger = None, prepare_input_function = None) -> str:
    trainer.messages = []
    for example in inputs:
        task_prompt, image_messages = formulate_input(trainer.dataset, example, use_pdf_id=True)

        task_prompt = "\n".join([
            task_prompt,
            f"[Database Schema]: {trainer.database_prompt}",
            f"[Vectorstore Schema]: {trainer.vectorstore_prompt}"
        ])
        if image_messages:
            task_prompt = [{'type': 'text', 'text': task_prompt}] + image_messages
        trainer.messages.append(
            [
            {'role': 'system', 'content': trainer.agent_prompt},
            {'role': 'user', 'content': task_prompt}
        ]
        )


    completions_texts, prompt_completion_ids, completion_ids = interact(trainer, prepare_input_function=prepare_input_function, messages=trainer.messages, use_consistency_selection=False, consistency_N=1, logger=logger) # 使用GRPO的生成方法替代model.get_response

        # 预处理：由于多步性，我是直接把多个response拼接在一起，因此有多个eos。
        # 因此需要找到每个response的eos，并将其作为结束标志。只留下最后的eos以便后续的处理

    # mask掉两个内容：1. 第一个eos之后的token； 2. 是pad的token
    is_eos = completion_ids == trainer.processing_class.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=trainer.accelerator.device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=trainer.accelerator.device).expand(is_eos.size(0), -1)
    eos_position_mask = sequence_indices <= eos_idx.unsqueeze(1)
    non_padding_mask = completion_ids != trainer.processing_class.pad_token_id
    combined_mask = eos_position_mask & non_padding_mask
    completion_mask = combined_mask.int()

    if trainer.mask_truncated_completions:
        truncated_completions = ~is_eos.any(dim=1)
        completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

    # 创建用于模型前向传播的 attention_mask
    # 修改：拼接 prompt_mask 和 completion_mask 的副本，并处理两者长度与prompt_completion_ids 一致
    attention_prompt_mask = prompt_mask
    attention_completion_mask = completion_mask.clone()

    # 调整 prompt_mask 和 completion_mask 以适应 prompt_completion_ids 的长度
    prompt_completion_len = prompt_completion_ids.size(1)
    prompt_len = prompt_ids.size(1)
    completion_len = completion_ids.size(1)

    if prompt_len + completion_len != prompt_completion_len:
        # 如果长度不一致，可能需要填充或截断
        if prompt_len + completion_len < prompt_completion_len:
            # 需要填充
            pad_len = prompt_completion_len - (prompt_len + completion_len)
            attention_completion_mask = torch.nn.functional.pad(attention_completion_mask, (pad_len, 0), value=0)
        else:
            # 需要截断
            attention_completion_mask = attention_completion_mask[:, :(prompt_completion_len-prompt_len)]

    # 拼接 attention_mask
    attention_mask = torch.cat([attention_prompt_mask, attention_completion_mask], dim=1)

    if trainer.accelerator.is_main_process: 
        logger.info(f'[Prompt Completion IDs]: {prompt_completion_ids.shape}')

    return completions_texts, prompt_completion_ids, completion_ids, attention_mask, completion_mask, is_eos


def interact(trainer, prepare_input_function, messages: List[List[Dict[str, Any]]], window_size: int = 3, output_path: Optional[str] = None, output_kwargs: Dict[str, Any] = {}, use_consistency_selection: bool = False, consistency_N: int = 5, logger: logging.Logger = None) -> str:
    # prev_cost = self.model.get_cost()
    trainer.env.reset()
    true_prompt_completion_ids = []
    true_completion_ids = []
    default_judge_list = []
    batch_size = len(messages)
    prompt_completion_ids_list = [[] for _ in range(batch_size)]
    completion_ids_list = [[] for _ in range(batch_size)]

    if trainer.accelerator.is_main_process:
        logger.info(f'[Batch Size]: {batch_size}')

    active_mask = [True] * batch_size # 用于记录每个样本是否已经完成任务
    obss = [[] for _ in range(batch_size)] # 理论上是[batch_size, num_turn]

    for turn in range(trainer.max_turn):
        if sum(active_mask) == 0: # 如果所有样本都完成任务，则退出循环
            if trainer.accelerator.is_main_process:
                logger.info(f'[Info]: all samples have completed the task at turn {turn + 1}.')
            break
        if trainer.accelerator.is_main_process:
            logger.info(f'[Interaction Turn]: {turn + 1}')
            logger.info(f'[Active Samples]: {sum(active_mask)}')

        prompt_texts = [] # 需要批量处理的prompt
        for message in messages:
            if len(message) > (window_size + 1) * 2: # each turn has two messages from assistant and user, respectively
                current_message = message[:2] + message[-window_size * 2:]
            else: current_message = message

            try:
                formatted_message = convert_message_from_gpt_format(trainer=trainer, messages=current_message)
                prompt_texts.append(maybe_apply_chat_template({"prompt": formatted_message}, trainer.processing_class)["prompt"])
            except Exception as e:
                default_prompt = "system: An error occurred while processing the previous turn."
                prompt_texts.append(default_prompt)
                logger.warning(f"Error in converting message: {e}")
        
        prompt_inputs = trainer.processing_class(
            text=prompt_texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        # 此时prompt_inputs是[batch_size, emb_size]
        prompt_inputs = prepare_input_function(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"].to(trainer.accelerator.device), prompt_inputs["attention_mask"].to(trainer.accelerator.device)
        
        if trainer.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -trainer.max_prompt_length:]
            prompt_mask = prompt_mask[:, -trainer.max_prompt_length:]
        
        iter = 1 if use_consistency_selection else consistency_N
        responsess = []
        prompt_completion_idss = []
        completion_idss = []

        with unwrap_model_for_generation(
                trainer.model_wrapped, trainer.accelerator, gather_deepspeed3_params=trainer.args.ds3_gather_for_generation
            ) as unwrapped_model:
            with (
                    FSDP.summon_full_params(trainer.model_wrapped, recurse=False)
                    if trainer.is_fsdp_enabled
                    else nullcontext()
                ):
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=trainer.generation_config
                    )

        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        
        completion_texts = trainer.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        active_mask_copy = active_mask.copy() # 需要在for循环中修改active_mask，因此需要复制一份
        
        # 因为step是串行交互的，因此这里通过for循环转串行
        for idx in range(batch_size):
            if active_mask[idx]:
                response = completion_texts[idx]
                obs, _, flag, _ = trainer.env.step(response, **output_kwargs)
                obss[idx].append(obs) # [batch_size, num_turn]
                if flag:
                    active_mask[idx] = False # 对应的样本已经完成任务
            else:
                obss[idx].append(None)
                completion_ids[idx] = torch.full_like(completion_ids[idx], trainer.processing_class.pad_token_id)
                prompt_completion_ids[idx, prompt_length:] = completion_ids[idx]

        # clean the last iter_num actions, and only keep the most consistent action
        
        active_num = sum(active_mask_copy) # 上一轮active的样本数
        active_idx = [idx for idx, mask in enumerate(active_mask_copy) if mask] # 目前active的样本的idx
        actions: List[Action] = trainer.env.parsed_actions[-active_num:] # 一共有active_num个action，按response的顺序依次添加

        if turn == 0:
            for idx in range(batch_size):
                prompt_completion_ids_list[idx].append(prompt_completion_ids[idx])
                completion_ids_list[idx].append(completion_ids[idx])

        # 第二轮之后，如果一个batch里面都没有成功解析，就不加入最终的序列
        else:
            for idx in range(batch_size):
                if idx not in active_idx: continue
                if isinstance(actions[active_idx.index(idx)], ErrorAction) is False: # 如果成功解析出动作，那就加入
                    prompt_completion_ids_list[idx].append(completion_ids[idx])
                    completion_ids_list[idx].append(completion_ids[idx])
            
        if trainer.accelerator.is_main_process:
            logger.info(f'[Response]: {completion_texts[0]}')

        action_msgs = []
        obs_msgs = []
        
        # 把action和obs转换成message的格式
        for idx in range(batch_size):
            if obss[idx][-1] is not None: # 若obss[idx][-1]为None，则说明样本在第turn轮已经完成任务
                obs_msgs.append(obss[idx][-1].convert_to_message())
            else:
                obs_msgs.append(None) # 相当于padding
            
            if idx not in active_idx:
                action_msgs.append(None) # 相当于padding
            else:
                action_msgs.append(actions[active_idx.index(idx)].convert_to_message(trainer.env.action_format, trainer.env.interact_protocol))

        if trainer.accelerator.is_main_process:
            logger.info(f"[Action]:{actions[0]}")

        # update history messages

        # 分发给batch中的每个Q
        for idx, (action_msg, obs_msg) in enumerate(zip(action_msgs, obs_msgs)):
            messages[idx].append(action_msg)
            messages[idx].append(obs_msg)
    
    if output_path is not None:
        with open(output_path, 'w', encoding='utf-8') as f:
            for m in messages:
                f.write(json.dumps(m, ensure_ascii=False) + '\n')

    # 对obss进行处理，把最后的非None值保留作为这个response的prev_answer.
    real_obss = [] # [batch_size]
    for idx_1 in range(batch_size):
        for idx_2 in reversed(range(len(obss[idx_1]))):
            if obss[idx_1][idx_2] is not None:
                real_obss.append(obss[idx_1][idx_2])
                break
    
    for idx in range(batch_size):
        # 只拼接最后3个成功解析的动作
        if len(prompt_completion_ids_list[idx]) >= 3:
            prompt_completion_ids_list[idx] = prompt_completion_ids_list[idx][-3:]
            completion_ids_list[idx] = completion_ids_list[idx][-3:]
        
        prompt_completion_ids_list[idx] = torch.cat(prompt_completion_ids_list[idx], dim=0)
        completion_ids_list[idx] = torch.cat(completion_ids_list[idx], dim=0)
    
    # 得到[batch_size, max_effective_turn * seq_len]的张量
    first_dim_flattened_prompt_completion_ids = pad_sequence(prompt_completion_ids_list, batch_first=True, padding_value=trainer.processing_class.pad_token_id, padding_side='left')
    first_dim_flattened_completion_ids = pad_sequence(completion_ids_list, batch_first=True, padding_value=trainer.processing_class.pad_token_id, padding_side='left')
        
    # 处理flattened张量中的EOS标记，只保留最后一个
    def replace_all_but_last_eos(tensor, eos_token_id, pad_token_id):
        """只保留张量中最后一个EOS标记，其他EOS标记替换为PAD标记"""
        if tensor.dim() == 2 and tensor.size(0) == batch_size:  # 确保是[batch_size, seq_len]形状
            # 对批次中的每个样本进行处理
            for batch_idx in range(batch_size):
                # 找出当前样本中所有EOS标记的位置
                eos_positions = (tensor[batch_idx] == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 1:  # 如果有多个EOS标记
                    # 仅保留最后一个EOS标记，其他的替换为PAD标记
                    for pos in eos_positions[:-1]:
                        tensor[batch_idx, pos] = pad_token_id
        return tensor
    
    # 应用EOS替换函数
    if first_dim_flattened_prompt_completion_ids.numel() > 0:
        first_dim_flattened_prompt_completion_ids = replace_all_but_last_eos(
            first_dim_flattened_prompt_completion_ids, 
            trainer.processing_class.eos_token_id, 
            trainer.processing_class.pad_token_id
        )
    
    if first_dim_flattened_completion_ids.numel() > 0:
        first_dim_flattened_completion_ids = replace_all_but_last_eos(
            first_dim_flattened_completion_ids, 
            trainer.processing_class.eos_token_id, 
            trainer.processing_class.pad_token_id
        )

    if trainer.accelerator.is_main_process:
        print(first_dim_flattened_completion_ids.shape)
        print(first_dim_flattened_prompt_completion_ids.shape)
    
    if trainer.accelerator.is_main_process:
        logger.info(f'[Pred]: {[truncate_tokens(str(obs.obs_content)) for obs in real_obss]}')
        
    return [truncate_tokens(str(obs.obs_content)) for obs in real_obss], first_dim_flattened_prompt_completion_ids, first_dim_flattened_completion_ids # 需要注意padding

def crop_image_count_in_messages(
    trainer,
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

def convert_message_from_gpt_format(trainer, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """ For VLLM-deployed open-source models, there are some limitations on:
    1. the input prompt length (e.g., 32k tokens)
    2. the number of images in the prompt (e.g., only one image)
    """
    keep_msg = 2 # one system message and one task message
    messages = crop_image_count_in_messages(trainer=trainer, messages=messages, image_limit=trainer.image_limit, keep_msg=keep_msg, in_place=True)

    if len(messages) > keep_msg:
        message_max_tokens = min(trainer.length_limit * 1000, trainer.processing_class.model_max_length) # by default, qwen2.5-72b-instruct is 32k

        truncated_messages = messages[:keep_msg]
        current_tokens = sum(len(trainer.processing_class.encode(str(message))) for message in truncated_messages)
        for i in range(len(messages) - 1, keep_msg - 1, -2):
            pair = messages[i-1:i+1]
            pair_tokens = sum(len(trainer.processing_class.encode(str(message))) for message in pair)
            if current_tokens + pair_tokens > message_max_tokens:
                break
            truncated_messages.insert(keep_msg, pair[1])
            truncated_messages.insert(keep_msg, pair[0])
            current_tokens += pair_tokens
        messages = truncated_messages

    return messages