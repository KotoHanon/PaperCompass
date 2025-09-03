import sys
import os
import hydra
import argparse, logging
import torch
import json
import tiktoken

from datasets import load_dataset, Dataset
from utils.eval_utils import evaluate, print_result, load_test_data, write_jsonl
from utils.hyperparam_utils import parse_args, get_result_folder, get_result_logger
from typing import List, Dict, Any
from omegaconf import DictConfig, OmegaConf
from argparse import Namespace
from utils.data_loading import load_data
from trl.trainer.dapo_trainer import DAPOTrainer
from trl.trainer.dapo_config import DAPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import wraps
from utils.reward_funcs import correct_reward_router_with_llm, correct_reward_router_without_llm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

@hydra.main(config_path="configs", config_name="dapo_train_config", version_base=None)
def neusym_rag_rl(cfg: DictConfig) -> None:
    train_dataset = load_data(cfg.test_data)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_cache=False,
    )

    args = DAPOConfig(
        num_generations=cfg.num_generations,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_completion_length=cfg.max_completion_length,
        max_draft_length = cfg.max_draft_length,
        max_prompt_length=cfg.max_prompt_length,
        max_steps=cfg.max_steps,
        bf16=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        output_dir=cfg.output_dir,
        report_to=cfg.report_to,
        logging_dir=cfg.logging_dir,
        logging_steps=cfg.logging_steps,
        do_eval=False,
        beta=0.0,
        max_grad_norm=cfg.max_grad_norm,
        num_iterations=cfg.num_iterations,
        epsilon_high=cfg.epsilon_high,
        mask_truncated_completions=True, # Overlong Filtering
        
    )

    trainer = DAPOTrainer(
        model=model,
        args=args,
        solution_reward_funcs=[correct_reward_router_with_llm],
        train_dataset=train_dataset,
        agent_method=cfg.method,
        max_turn=cfg.max_turn,
        dataset=cfg.dataset_name,
        database=cfg.db,
        vectorstore=cfg.vectorstore,
        database_dir=cfg.database_dir,
        vectorstore_dir=cfg.vectorstore_dir,
        launch_method=cfg.launch_method,
        interact_protocol=cfg.interact_protocol,
        db_format=cfg.db_format,
        vs_format=cfg.vs_format,
        action_format=cfg.action_format,
        window_size=cfg.window_size,
    )

    trainer.train()

if __name__ == "__main__":
    
    neusym_rag_rl()
