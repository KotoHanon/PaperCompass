from datasets import load_dataset, Dataset
from utils.eval_utils import evaluate, print_result, load_test_data, write_jsonl
from utils.hyperparam_utils import parse_args, get_result_folder, get_result_logger
from typing import List, Dict, Any
import sys
import os
from argparse import Namespace
import argparse, logging
import torch
import json
from utils.data_loading import load_data

# 添加本地trl模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
trl_path = os.path.join(current_dir, "trl")
sys.path.append(current_dir)
sys.path.append(trl_path)

from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from transformers import AutoModelForCausalLM
from evaluation.evaluator import evaluate_airqa

# 创建参数对象并填充默认值，不需要从命令行解析
args = Namespace()
args.dataset = "airqa"
args.agent_method = "neusym_rag"
args.action_format = "json"
args.interact_protocol = "react"
args.database = "ai_research"
args.vectorstore = "ai_research"
args.database_path = os.path.join(current_dir, "data/database/ai_research/ai_research.base501.duckdb")
args.vectorstore_path = os.path.join(current_dir, "data/vectorstore/ai_research/ai_research.base501.db")
args.launch_method = "standalone"
args.docker_uri = None 
args.max_turn = 2
args.example = "airqa_example"
args.db_format = "create_sql"
args.vs_format = "detailed_json"
args.database_path = os.path.join(current_dir, "data/database/ai_research/ai_research.base501.duckdb")
args.vectorstore_path = os.path.join(current_dir, "data/vectorstore/ai_research/ai_research.base501.db")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset, eval_dataset = load_data()

def reward_func_adapter(prompts, completions, **reward_kwargs): # 适配层，将evaluate_airqa函数适配为reward_func（reward router）
    rewards = []
    for i, completion in enumerate(completions):
        gold_str = reward_kwargs["gold_str"][i]
        gold_answer = json.loads(gold_str)
        score = evaluate_airqa(pred_answer=completion, gold=gold_answer)
        rewards.append(float(score))
    
    return rewards

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
    ).to(device)

config = GRPOConfig(
    num_generations=8,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    wandb_log_unique_prompts=True,
    max_completion_length=2048
)

trainer = GRPOTrainer(
    model=model,
    args=config,
    reward_funcs=reward_func_adapter,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    agent_method=args.agent_method,
    max_turn=args.max_turn,
    dataset=args.dataset,
    database=args.database,
    vectorstore=args.vectorstore,
    database_path=args.database_path,
    vectorstore_path=args.vectorstore_path,
    launch_method=args.launch_method,
    docker_uri=args.docker_uri,
    interact_protocol=args.interact_protocol,
    db_format=args.db_format,
    vs_format=args.vs_format
)

trainer.train()

# 以下命令行参数定义代码可以保留但不使用
# 如果需要命令行参数，可以注释掉上面的args赋值部分，取消注释下面的代码
"""
parser = argparse.ArgumentParser()
# dataset, database, vectorstore utils
parser.add_argument('--dataset', type=str, default="airqa", help='Which dataset to use.')
parser.add_argument('--database', type=str, default="ai_research", help='Which database to use, i.e., the name of the DB.')
parser.add_argument('--database_path', type=str, help=f'Database path. The default path is `${DATABASE_DIR}/${{dataset}}/${{database}}.db`.')
parser.add_argument('--database_type', type=str, default='duckdb', help='Which database type to use. We only support DuckDB currently.')
parser.add_argument('--vectorstore', type=str, help='Which vectorstore to use, usually the same name with the database.')
parser.add_argument('--launch_method', type=str, default='standalone', choices=['standalone', 'docker'], help='Launch method for vectorstore, chosen from ["docker", "standalone"]. `standalone` -> from `.db` file; `docker` -> from docker containers.')
parser.add_argument('--docker_uri', type=str, default='http://127.0.0.1:19530', help='The host:port for vectorstore started from docker.')
parser.add_argument('--vectorstore_path', type=str, help=f'Path to the vectorstore if launched from method `standalone`. The default path is `${VECTORSTORE_DIR}/${{dataset}}/${{vectorstore}}.db`.')
parser.add_argument('--test_data', type=str, default='test_data.jsonl', help=f'Test data file or path. If file name, search the default filepath `${DATASET_DIR}/${{dataset}}/${{test_data}}`.')

# agent, llm, env utils
parser.add_argument('--db_format', type=str, choices=['create_sql', 'detailed_json'], default='create_sql', help='Database schema serialization format. See agents/prompts/schema_prompt.py for details.')
parser.add_argument('--vs_format', type=str, choices=['detailed_json'], default='detailed_json', help='Vectorstore schema serialization format. See agents/prompts/schema_prompt.py for details.')
parser.add_argument('--action_format', type=str, default='markdown', choices=['markdown', 'json', 'xml', 'yaml'], help='Action format for the environment acceptable inputs.')
parser.add_argument('--output_format', type=str, default='json', choices=['markdown', 'json', 'html', 'string'], help='Output/Observation format of tables for the environment execution results.')
parser.add_argument('--agent_method', type=str, default='neusym_rag', choices=[
    'trivial_question_only', 'trivial_title_with_abstract', 'trivial_full_text_with_cutoff', 'classic_rag', 'two_stage_neu_rag', 'two_stage_sym_rag', 'two_stage_graph_rag', 'two_stage_hybrid_rag', 'iterative_classic_rag', 'iterative_neu_rag', 'iterative_sym_rag', 'iterative_graph_rag', 'neusym_rag'
], help='Various agent / baseline method.')
parser.add_argument('--interact_protocol', type=str, default='react', choices=['react', 'code_block'], help='Interaction protocol for the agent method which is used to extract the parsable action text from LLM response, chosen from ["react", "code_block"].')
parser.add_argument('--llm', type=str, default='gpt-4o-mini', help='LLM name to use. See agents/models for all supported LLMs.')
parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling from the LLM.')
parser.add_argument('--top_p', type=float, default=0.95, help='Top-p for sampling from the LLM.')
parser.add_argument('--max_tokens', type=int, default=1500, help='Maximum number of tokens to generate, a.k.a., the maximum completion tokens.')
parser.add_argument('--max_turn', type=int, default=20, help='Maximum turns for the agent to interact with the environment.')
parser.add_argument('--window_size', type=int, default=5, help='History window size, or the number of previous (action, observation) pairs preserved in the prompt when calling LLMs.')
parser.add_argument('--image_limit', type=int, default=10, help='Maximum number of images to be shown in the agent prompt. Also restricted by the LLMs/VLMs, e.g., --limit_mm_per_prompt.')
parser.add_argument('--length_limit', type=int, default=32, help='The total length limit of the prompt (multiplied by 1000). By default, 32k.')

# method specific hyperparams
parser.add_argument('--collection_name', type=str, default='text_sentence_transformers_all_minilm_l6_v2', help='For Classic-RAG and Iterative Classic-RAG methods, the collection name to retrieve context.')
parser.add_argument('--table_name', type=str, default='chunks', help='For Classic-RAG and Iterative Classic-RAG methods, the table name to retrieve context.')
parser.add_argument('--column_name', type=str, default='text_content', help='For Classic-RAG and Iterative Classic-RAG methods, the column name to retrieve context.')
parser.add_argument('--limit', type=int, default=4, help='For Classic-RAG, the limit or top K of the retrieved chunks.')
parser.add_argument('--cutoff', type=int, default=5, help='For title with abstract and full-text with cutoff baseline, restrict the length of tokens (multiply 1000) for the full-text.')
parser.add_argument('--graphrag_method', type=str, default='local', choices=['local', 'global'], help='For Graph-RAG and Iterative Graph-RAG, the method to use, chosen from ["local", "global"].')
parser.add_argument('--graphrag_embed', type=str, default='text-embedding-3-small', help='For Graph-RAG and Iterative Graph-RAG, the embedding model to use.')

# output, result utils
parser.add_argument('--result_dir', type=str, default='results', help='Directory to save the results')
parser.add_argument('--no_eval', action='store_true', help='Whether not to evaluate the results, because subjective evaluation usually involves LLM-based judgement.')
args = parser.parse_args()
"""
