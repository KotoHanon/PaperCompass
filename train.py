import sys
import os
import hydra
import argparse, logging
import torch
import json

from datasets import load_dataset, Dataset
from utils.eval_utils import evaluate, print_result, load_test_data, write_jsonl
from utils.hyperparam_utils import parse_args, get_result_folder, get_result_logger
from typing import List, Dict, Any
from omegaconf import DictConfig, OmegaConf
from argparse import Namespace
from utils.data_loading import load_data
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer # AutoConfig 也可能有用
from evaluation.evaluator import evaluate_airqa

tiktoken_cache_dir = "."
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
cache_key = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
os.environ['VLLM_API_KEY'] = "lococo"
os.environ['VLLM_BASE_URL'] = "http://localhost:8000/v1"
os.environ["TOWHEE_HOME"] = "./.towhee"
os.environ["NLTK_DATA"] = "./nltk_data"
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def reward_func_adapter(prompts, completions, **reward_kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        gold_str = reward_kwargs["gold_str"][i]
        gold_answer = json.loads(gold_str)
        score = evaluate_airqa(pred_answer=completion, gold=gold_answer)
        rewards.append(float(score))
        '''rewards.append(0.0)  # Placeholder for actual reward calculation'''
    return rewards

@hydra.main(config_path="configs", config_name="train_config", version_base=None)
def neusym_rag_rl(cfg: DictConfig) -> None:
    train_dataset, eval_dataset = load_data()

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    args = GRPOConfig(
        num_generations=cfg.num_generations,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        wandb_log_unique_prompts=True,
        max_completion_length=cfg.max_completion_length,
        max_prompt_length=cfg.max_prompt_length,
        max_steps=cfg.max_steps,
        bf16=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )

    trainer = GRPOTrainer(
        model=model,
        args=args,
        reward_funcs=reward_func_adapter,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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

