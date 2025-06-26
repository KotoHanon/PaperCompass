import re
from collections import Counter
from typing import List
import json
from evaluation.evaluator import evaluate_airqa
import evaluation
from fuzzywuzzy import fuzz, process

def eval_string_fuzzy_match(
        pred: str,
        gold: str,
        fuzz_method: str = 'token_set_ratio',
        threshold: int = 90,
        ignore_blank: bool = False,
        lowercase: bool = False,
        **kwargs
    ) -> float:
    """ Evaluate the predicted answer against the gold answer using fuzzy string match.
    @param:
        pred: str, the predicted answer
        gold: str, the gold answer
        fuzz_method: str, the method for fuzzy string matching, by default, 'ratio'
        threshold: int, the threshold for fuzzy string matching, by default, 95
        ignore_blank: bool, whether to ignore the blank spaces, by default, False
        lowercase: bool, whether to convert the strings to lowercase before comparison, by default, False
    @return
        float, 1.0 or 0.0
    """
    pred, gold = str(pred).strip(), str(gold).strip()
    # fuzz_method chosen from ['ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio']
    fuzz_function = getattr(fuzz, fuzz_method)
    if ignore_blank:
        if fuzz_method in ['token_sort_ratio', 'token_set_ratio']: # for tokens, preserve the blank spaces
            pred, gold = re.sub(r'\s+', ' ', pred), re.sub(r'\s+', ' ', gold)
        else:
            pred, gold = re.sub(r'\s+', '', pred), re.sub(r'\s+', '', gold)
    return float(fuzz_function(pred.lower(), gold.lower()) >= threshold) if lowercase else float(fuzz_function(pred, gold) >= threshold)

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
        if str(pred_answer).startswith('[ERROR]:'): 
            rewards.append(0.0)
            continue
    
        function_name = gold_answer['evaluator']['eval_func']
        eval_func = getattr(evaluation, function_name, None)
        assert eval_func is not None, f"Evaluation function `{function_name}` not found in the evaluation module. Remember to import it in the evaluation/__init__.py file."
        eval_kwargs = gold_answer['evaluator']['eval_kwargs']
        if function_name == "eval_complex_math_formula_with_llm":
            gold_answer = eval_kwargs["formulas"]
            score = eval_string_fuzzy_match(
                pred=pred_answer,
                gold=gold_answer,
            )
        elif function_name == "eval_reference_answer_with_llm":
            gold_answer = eval_kwargs["reference_answer"]
            score = eval_string_fuzzy_match(
                pred=pred_answer,
                gold=gold_answer,
            )
        else:
            try:
                score = evaluate_airqa(pred_answer=pred_answer, gold=gold_answer)
            except Exception as e:
                score = 0.0

        rewards.append(float(score))
        # rewards.append(0.0)  # Placeholder for actual reward calculation'''
    return rewards

'''def correct_reward_router(prompts, completions, **reward_kwargs) -> List[float]:
    rewards = []
    for i, completion in enumerate(completions):
        gold_str = reward_kwargs["gold_str"][i]
        gold_answer = json.loads(gold_str)
        pred_answer = parse_pred_answer(completion)
        score = evaluate_airqa(pred_answer=pred_answer, gold=gold_answer)
        rewards.append(float(score))
        # rewards.append(0.0)  # Placeholder for actual reward calculation
    return rewards'''

def repetition_penalty(prompts, completions, **reward_kwargs) -> List[float]:
    penalties = []
    penalty_coffecient = -0.1
    for completion in completions:
        action_list = parse_action(completion)
        action_counts = Counter(action_list)
        # Calculate the repetition penalty
        try: # if the agent can not parse the action, it will return 10
            max_repeats = max(action_counts.values())
        except Exception as e:
            max_repeats = 10
        penalties.append(penalty_coffecient * (max_repeats - 1) if max_repeats > 1 else 0.0)
    return penalties