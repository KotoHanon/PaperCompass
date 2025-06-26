import re
from collections import Counter
from typing import List
import json
from evaluation.evaluator import evaluate_airqa

def parse_pred_answer(completion: str) -> str:
    """
    Parses the predicted answer from the completion string.
    the answer is enclosed in [Observation]: ... [/Observation].
    """
    observation_pattern = r"\[Observation]:(.*?)\[/Observation]"
    match = re.search(observation_pattern, completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return "No valid answer found"

def parse_action(completion: str) -> List[str]:
    """
    Parses the action from the completion string.
    the answer is enclosed in [Action]: ... [/Action].
    """
    action_pattern = r"\[Action]:(.*?)\[/Action]"
    action_texts = re.findall(action_pattern, completion, re.DOTALL)
    
    return [action.strip() for action in action_texts]


def correct_reward_router(predict_str: str, solution: str) -> float:
    gold_answer = json.loads(solution)
    pred_answer = parse_pred_answer(predict_str)
    score = evaluate_airqa(pred_answer=pred_answer, gold=gold_answer)
    return score

def repetition_penalty(predict_str: str) -> float:
    action_list = parse_action(predict_str)
    action_counts = Counter(action_list)
    # Calculate the repetition penalty
    max_repeats = max(action_counts.values())
    return penalty_coffecient * (max_repeats - 1) if max_repeats > 1 else 0.0

def compute_score(predict_str: str, solution: str) -> float:
    return correct_reward_router(predict_str, solution) - repetition_penalty(predict_str)
