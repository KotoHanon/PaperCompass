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


def correct_reward_router(prompts, completions, **reward_kwargs) -> List[float]:
    rewards = []
    for i, completion in enumerate(completions):
        gold_str = reward_kwargs["gold_str"][i]
        gold_answer = json.loads(gold_str)
        pred_answer = parse_pred_answer(completion)
        score = evaluate_airqa(pred_answer=pred_answer, gold=gold_answer)
        rewards.append(float(score))
        '''rewards.append(0.0)  # Placeholder for actual reward calculation'''
    return rewards

def repetition_penalty(prompts, completions, **reward_kwargs) -> List[float]:
    penalties = []
    penalty_coffecient = -0.1
    for completion in completions:
        action_list = parse_action(completion)
        action_counts = Counter(action_list)
        # Calculate the repetition penalty
        max_repeats = max(action_counts.values())
        penalties.append(penalty_coffecient * (max_repeats - 1) if max_repeats > 1 else 0.0)
    return penalties