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
#from ..extras.vllm_client import VLLMClient
from ..import_utils import is_liger_kernel_available, is_rich_available, is_vllm_available
from ..models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from .callbacks import SyncRefModelCallback
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

def replace_all_but_last_eos(tensor, batch_size: int, eos_token_id: torch.Tensor, pad_token_id: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2 and tensor.size(0) == batch_size:  # [batch_size, seq_len]
        for batch_idx in range(batch_size):
            eos_positions = (tensor[batch_idx] == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 1: 
                for pos in eos_positions[:-1]:
                    tensor[batch_idx, pos] = pad_token_id
    return tensor

def replace_all_but_first_bos(tensor, batch_size: int, bos_token_id: torch.Tensor, pad_token_id: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2 and tensor.size(0) == batch_size:  # [batch_size, seq_len]
        for batch_idx in range(batch_size):
            bos_positions = (tensor[batch_idx] == bos_token_id).nonzero(as_tuple=True)[0]
            if len(bos_positions) > 1: 
                for pos in bos_positions[1:]:
                    tensor[batch_idx, pos] = pad_token_id
    return tensor

def get_completion_mask(trainer, completion_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # we pad for: 1. everyone after the first eos token, and 2. pad_token_id for the rest
    is_eos = completion_ids == trainer.processing_class.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=trainer.accelerator.device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=trainer.accelerator.device).expand(is_eos.size(0), -1)
    eos_position_mask = sequence_indices <= eos_idx.unsqueeze(1)
    non_padding_mask = completion_ids != trainer.processing_class.pad_token_id
    combined_mask = eos_position_mask & non_padding_mask
    completion_mask = combined_mask.int()

    return completion_mask, is_eos

def adaption_layer(trainer, inputs, window_size: int = 5, output_path: Optional[str] = None, output_kwargs: Dict[str, Any] = {}, logger: logging.Logger = None, prepare_input_function = None) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    trainer.messages = []

    for idx, example in enumerate(inputs):
        task_prompt, image_messages, draft_prompt = formulate_input(trainer.dataset, example, use_pdf_id=True, use_draft=True)
        if trainer.accelerator.is_main_process:
            if idx == 0:
                logger.info(f'[Task Prompt]: {task_prompt}')

        '''task_prompt = "\n".join([
            task_prompt,
            f"[Database Schema]: {trainer.database_prompt}",
            f"[Vectorstore Schema]: {trainer.vectorstore_prompt}"
        ])'''
        if image_messages:
            task_prompt = [{'type': 'text', 'text': task_prompt}] + image_messages
        trainer.messages.append([
            {'role': 'system', 'content': trainer.agent_prompt},
            {'role': 'user', 'content': task_prompt}
        ])
    
    messages_with_drafts, drafts_texts, draft_ids = draft_generation(trainer=trainer, prepare_input_function=prepare_input_function, draft_prompt=draft_prompt, messages=trainer.messages, logger=logger)
    completions_texts, prompt_completion_ids, completion_ids, prompt_ids, prompt_mask = interact(trainer, prepare_input_function=prepare_input_function, messages=messages_with_drafts, window_size=window_size, logger=logger)
    #completions_texts, prompt_completion_ids, completion_ids, prompt_ids, prompt_mask = interact(trainer, prepare_input_function=prepare_input_function, messages=trainer.messages, window_size=window_size, logger=logger)

    completion_mask, is_eos = get_completion_mask(trainer, completion_ids)
    draft_mask = get_completion_mask(trainer, draft_ids)[0]

    if trainer.mask_truncated_completions:
        truncated_completions = ~is_eos.any(dim=1)
        completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

    prompt_completion_len = prompt_completion_ids.size(1)
    prompt_len = prompt_ids.size(1)
    completion_len = completion_ids.size(1)


    assert prompt_completion_len == prompt_len + completion_len, \
        f'Prompt completion length {prompt_completion_len} does not match the sum of prompt length {prompt_len} and completion length {completion_len}.'

    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

    return completions_texts, prompt_completion_ids, completion_ids, draft_ids, attention_mask, completion_mask, draft_mask, is_eos

def draft_generation(trainer, prepare_input_function, draft_prompt: str, messages: List[List[Dict[str, Any]]], logger: logging.Logger = None) -> Tuple[List[List[Dict[str, Any]]], List[str], torch.Tensor]:
    # generate the draft.
    prompt_texts = []
    for message in messages:
        try:
            formatted_message = convert_message_from_gpt_format(trainer=trainer, messages=message)
            prompt_texts.append(maybe_apply_chat_template({"prompt": formatted_message}, trainer.processing_class)["prompt"])
        except Exception as e:
            default_prompt = "system: An error occurred while processing the previous turn."
            prompt_texts.append(default_prompt)
            logger.warning(f"Error in converting message: {e}")
        
    prompt_inputs = trainer.processing_class(
        text=prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False
    )
    # prompt_inputs.shape -> [batch_size, emb_size]
    prompt_inputs = prepare_input_function(prompt_inputs)
    prompt_ids, prompt_mask = prompt_inputs["input_ids"].to(trainer.accelerator.device), prompt_inputs["attention_mask"].to(trainer.accelerator.device)
    
    if trainer.max_prompt_length is not None:
        prompt_ids = prompt_ids[:, -trainer.max_prompt_length:]
        prompt_mask = prompt_mask[:, -trainer.max_prompt_length:]

    with unwrap_model_for_generation(trainer.model_wrapped, trainer.accelerator, gather_deepspeed3_params=trainer.args.ds3_gather_for_generation) as unwrapped_model:
        with (FSDP.summon_full_params(trainer.model_wrapped, recurse=False) if trainer.is_fsdp_enabled else nullcontext()):
                prompt_completion_ids = unwrapped_model.generate(prompt_ids, attention_mask=prompt_mask, generation_config=trainer.draft_config)

    prompt_length = prompt_ids.size(1)
    draft_ids = prompt_completion_ids[:, prompt_length:]
    draft_texts = trainer.processing_class.batch_decode(draft_ids, skip_special_tokens=True)

    # assign each draft text to the corresponding message
    new_messages = copy.deepcopy(messages) # messages with draft texts appended
    for idx, (message, draft_text) in enumerate(zip(messages, draft_texts)):
        cur_content = message[-1]["content"]
        new_content = cur_content.replace(draft_prompt, draft_text)
        new_messages[idx][-1]["content"] = new_content
        if trainer.accelerator.is_main_process:
            logger.info(f'[Prompt]: {new_content}')

    return new_messages, draft_texts, draft_ids

def interact(trainer, prepare_input_function, messages: List[List[Dict[str, Any]]], window_size: int = 5, output_path: Optional[str] = None, output_kwargs: Dict[str, Any] = {}, logger: logging.Logger = None) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    trainer.env.reset()
    batch_size = len(messages)
    prompt_completion_ids_list = [[] for _ in range(batch_size)]
    completion_ids_list = [[] for _ in range(batch_size)]
    full_completion_texts = []

    initial_prompt_ids, initial_prompt_mask = None, None

    if trainer.accelerator.is_main_process:
        logger.info(f'[Batch Size]: {batch_size}')

    active_mask = [True] * batch_size
    obss = [[] for _ in range(batch_size)] # [batch_size, num_turn]
    
    for turn in range(trainer.max_turn):
        if sum(active_mask) == 0: # exit if all samples have completed the task
            if trainer.accelerator.is_main_process:
                logger.info(f'[Info]: all samples have completed the task at turn {turn + 1}.')
            break
        if trainer.accelerator.is_main_process:
            logger.info(f'[Interaction Turn]: {turn + 1}')
            logger.info(f'[Active Samples]: {sum(active_mask)}')

        prompt_texts = []
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
            text=prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False, padding_side="left"
        )
        # prompt_inputs.shape -> [batch_size, emb_size]
        prompt_inputs = prepare_input_function(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"].to(trainer.accelerator.device), prompt_inputs["attention_mask"].to(trainer.accelerator.device)
        
        if trainer.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -trainer.max_prompt_length:]
            prompt_mask = prompt_mask[:, -trainer.max_prompt_length:]
            if turn == 0:
                initial_prompt_ids = prompt_ids
                initial_prompt_mask = prompt_mask

        with unwrap_model_for_generation(trainer.model_wrapped, trainer.accelerator, gather_deepspeed3_params=trainer.args.ds3_gather_for_generation) as unwrapped_model:
            with (FSDP.summon_full_params(trainer.model_wrapped, recurse=False) if trainer.is_fsdp_enabled else nullcontext()):
                    prompt_completion_ids = unwrapped_model.generate(prompt_ids, attention_mask=prompt_mask, generation_config=trainer.generation_config)

        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]   
        completion_texts = trainer.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        active_mask_copy = active_mask.copy()
        
        for idx in range(batch_size):
            if active_mask[idx]:
                response = completion_texts[idx]
                obs, _, flag, _ = trainer.env.step(response, **output_kwargs)
                obss[idx].append(obs) # [batch_size, num_turn]
                if flag:
                    active_mask[idx] = False # corresponding sample has completed the task
            else:
                obss[idx].append(None)
                completion_ids[idx] = torch.full_like(completion_ids[idx], trainer.processing_class.pad_token_id) # padding with pad_token_id
                prompt_completion_ids[idx, prompt_length:] = completion_ids[idx]

        active_num = sum(active_mask_copy)
        active_idx = [idx for idx, mask in enumerate(active_mask_copy) if mask]
        actions: List[Action] = trainer.env.parsed_actions[-active_num:] # notice that we only have active_num actions in the last turn, which is less than batch_size

        if turn == 0:
            for idx in range(batch_size):
                prompt_completion_ids_list[idx].append(completion_ids[idx])
                completion_ids_list[idx].append(completion_ids[idx])

        else:
            for idx in range(batch_size):
                if idx not in active_idx: continue
                if isinstance(actions[active_idx.index(idx)], ErrorAction) is False: # append if the action is not an instance of ErrorAction
                    prompt_completion_ids_list[idx].append(completion_ids[idx])
                    completion_ids_list[idx].append(completion_ids[idx])
        
        completion_texts = trainer.processing_class.batch_decode(completion_ids, skip_special_tokens=True) # decode again, since we add pad tokens for inactive samples
        completion_texts = [text + "[/Action]" for text in completion_texts]
        full_completion_texts.append(completion_texts)
            
        if trainer.accelerator.is_main_process:
            logger.info(f'[Response]: {completion_texts[0]}')

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
                action_msgs.append(actions[active_idx.index(idx)].convert_to_message(trainer.env.action_format, trainer.env.interact_protocol))

        if trainer.accelerator.is_main_process:
            logger.info(f"[Action]:{actions[0] if actions[0] is not None else 'None'}")
            logger.info(f"[Observation]:{obs_msgs[0]['content'] if obs_msgs[0] is not None else 'None'}")

        # update history messages
        for idx, (action_msg, obs_msg) in enumerate(zip(action_msgs, obs_msgs)):
            # we clean the None messages in crop_image_count_in_messages
            messages[idx].append(action_msg)
            messages[idx].append(obs_msg)
    
    if output_path is not None:
        with open(output_path, 'w', encoding='utf-8') as f:
            for m in messages:
                f.write(json.dumps(m, ensure_ascii=False) + '\n')

    # keep the last valid observation for each sample as the pred_answer
    real_obss = [] # [batch_size]
    for idx_1 in range(batch_size):
        for idx_2 in reversed(range(len(obss[idx_1]))):
            if obss[idx_1][idx_2] is not None:
                real_obss.append(obss[idx_1][idx_2])
                break
    
    for idx in range(batch_size):
        if len(prompt_completion_ids_list[idx]) >= 5:
            prompt_completion_ids_list[idx] = prompt_completion_ids_list[idx][-5:]
            completion_ids_list[idx] = completion_ids_list[idx][-5:]
        
        prompt_completion_ids_list[idx].insert(0, initial_prompt_ids[idx]) # insert the initial prompt at the beginning

        prompt_completion_ids_list[idx] = torch.cat(prompt_completion_ids_list[idx], dim=0)
        completion_ids_list[idx] = torch.cat(completion_ids_list[idx], dim=0)
    
    #first_dim_flattened_prompt_completion_ids = pad_sequence(prompt_completion_ids_list, batch_first=True, padding_value=trainer.processing_class.pad_token_id, padding_side='left')
    #first_dim_flattened_completion_ids = pad_sequence(completion_ids_list, batch_first=True, padding_value=trainer.processing_class.pad_token_id, padding_side='left')
        
    first_dim_flattened_prompt_completion_ids = pad_sequence(prompt_completion_ids_list, batch_first=True, padding_value=trainer.processing_class.pad_token_id, padding_side='left')
    first_dim_flattened_completion_ids = pad_sequence(completion_ids_list, batch_first=True, padding_value=trainer.processing_class.pad_token_id, padding_side='left')
    # only keep the last EOS token in each sample, replace others with PAD token
    
    if first_dim_flattened_prompt_completion_ids.numel() > 0:
        first_dim_flattened_prompt_completion_ids = replace_all_but_last_eos(
            first_dim_flattened_prompt_completion_ids, 
            batch_size,
            trainer.processing_class.eos_token_id, 
            trainer.processing_class.pad_token_id
        )
        first_dim_flattened_completion_ids = replace_all_but_first_bos(
            first_dim_flattened_completion_ids, 
            batch_size,
            trainer.processing_class.bos_token_id, 
            trainer.processing_class.pad_token_id
        )
    
    if first_dim_flattened_completion_ids.numel() > 0:
        first_dim_flattened_completion_ids = replace_all_but_last_eos(
            first_dim_flattened_completion_ids, 
            batch_size,
            trainer.processing_class.eos_token_id, 
            trainer.processing_class.pad_token_id
        )
        first_dim_flattened_completion_ids = replace_all_but_first_bos(
            first_dim_flattened_completion_ids, 
            batch_size,
            trainer.processing_class.bos_token_id, 
            trainer.processing_class.pad_token_id
        )

    if trainer.accelerator.is_main_process:
        print(first_dim_flattened_completion_ids.shape)
        print(first_dim_flattened_prompt_completion_ids.shape)
    
    logger.info(f'[Pred from {trainer.accelerator.process_index}]: {[truncate_tokens(str(obs.obs_content)) for obs in real_obss]}')
    completion_texts = [''.join(column) for column in zip(*full_completion_texts)] # concatenate the completion texts for each sample
    completion_texts = [text + "\n" + "[Observation]: " + truncate_tokens(str(obs.obs_content)) + "[/Observation]" for text, obs in zip(completion_texts, real_obss)]
        
    return completion_texts, first_dim_flattened_prompt_completion_ids, first_dim_flattened_completion_ids, initial_prompt_ids, initial_prompt_mask

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
