import json
from typing import List, Dict, Any
from datasets import load_dataset, Dataset
from utils.eval_utils import evaluate, print_result, load_test_data, write_jsonl

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

    return dataset