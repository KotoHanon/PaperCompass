from datasets import load_dataset, Dataset
from utils.eval_utils import evaluate, print_result, load_test_data, write_jsonl
from utils.hyperparam_utils import parse_args, get_result_folder, get_result_logger
from typing import List, Dict, Any
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM

def problem_reward(completions, answers, **kwargs):
    """Reward function for math problems with verifiable answers
    completions: list of completions to evaluate
    answers: list of answers to the problems from the dataset
    """

    rewards = []
    for completion, correct_answer in zip(completions, answers):
        # Extract the answer from the completion
        try:
            # This is a simplified example - you'd need proper parsing
            answer = completion
            # Binary reward: 1 for correct, 0 for incorrect
            reward = 1.0 if answer == correct_answer else 0.0
            rewards.append(reward)
        except:
            # If we can't parse an answer, give a low reward
            rewards.append(0.0)

    return rewards

#args: Namespace = parse_args()
data: List[Dict[str, Any]] = load_test_data("test_data_553.jsonl", "airqa")
formatted_data = []
for item in data:
    if "prompt" not in item:
        prompt = item.get("question") + ' ' + item.get("answer_format")
        formatted_data.append({"prompt": prompt})
    else:
        formatted_data.append(item)

dataset = Dataset.from_list(formatted_data)

train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

config = GRPOConfig(
    dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_turn=20,
)

trainer = GRPOTrainer(
    model=model,
    config=config,
    reward_funcs=problem_reward,
    max_turn=20,
)





