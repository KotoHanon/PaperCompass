import json
import os
from typing import List, Dict, Any
from datasets import load_dataset, Dataset
from utils.eval_utils import evaluate, print_result, load_test_data, write_jsonl
import argparse

def load_data(data_path: str = "test_data_553.jsonl", dataset_name: str = "airqa") -> Dataset:
    data: List[Dict[str, Any]] = load_test_data(data_path, dataset_name)
    formatted_data = []
    for item in data:
        if "prompt" not in item:
            formatted_data.append({"prompt": item.get("question") + ' ' + item.get("answer_format"), "question": item.get("question"), "answer_format": item.get("answer_format"), 
                                "gold": item})
        else:
            formatted_data.append(item)

    output_jsonl_file = "formatted_data.jsonl"

    with open(output_jsonl_file, "w", encoding="utf-8") as f:
        for item_dict in formatted_data:
            json_string = json.dumps(item_dict, ensure_ascii=False)
            f.write(json_string + "\n")

    raw_data = []
    with open('formatted_data.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): 
                item = json.loads(line)
                raw_data.append({
                    "prompt": item.get("prompt", ""),
                    "question": item.get("question", ""),
                    "answer_format": item.get("answer_format", ""),
                    "gold_str": json.dumps(item.get("gold", {}))
                })

    dataset = Dataset.from_list(raw_data)

    train_test_split = dataset.train_test_split(test_size=0.01, seed=114514)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    return train_dataset, eval_dataset

def make_map_fn(split):
    def process_fn(example, idx):
        question = example.pop("prompt")
        solution = example.pop("gold_str")
        data = {
            "prompt": [{"role": "user", "content": question}],
            "ability": "agent",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": split, "index": idx},
        }
        return data
    return process_fn


if __name__ == "__main__":
    train_dataset, test_dataset = load_data(data_path = "test_data_1246.jsonl")
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir")
    args = parser.parse_args()
    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))