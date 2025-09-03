#coding=utf8
import os
from typing import Any, List, Dict, Tuple, Optional
from utils.functions.image_functions import get_image_message
from utils.airqa_utils import get_airqa_paper_metadata
from utils.config import DATASET_DIR


def formulate_input(
        dataset: str, 
        data: Dict[str, Any],
        use_pdf_id: bool = True,
        use_reference_pdf: bool = False,
        use_image: bool = False, 
        use_draft: bool = False
) -> Tuple[str, List[Dict[str, Any]]]:
    question, answer_format = f"[Question]: {data['question']}", f"[Answer Format]: {data['answer_format']}"
    pdf_context, image_messages = "", []

    # get pdf context
    anchor_pdf_id = data.get('anchor_pdf', [])
    dataset_dir = os.path.join(DATASET_DIR, dataset)
    uuid2papers = get_airqa_paper_metadata(dataset_dir=dataset_dir)
    if len(anchor_pdf_id) > 0: 
        if use_pdf_id:
            anchor_pdf = [repr(pdf_id) for pdf_id in anchor_pdf_id]
        else:
            anchor_pdf = [repr(uuid2papers[uid]['title']) for uid in anchor_pdf_id]
        pdf_context += f"[Anchor PDF]: {anchor_pdf[0]}\n" if len(anchor_pdf) == 1 else \
            f"[Anchor PDFs]: [{', '.join(anchor_pdf)}]\n"

    # Currently, we set `use_reference_pdf=False`, because these reference PDFs should be recognized in the Reference or Related Work section in real-world scenarios.
    if use_reference_pdf and data.get('reference_pdf', []):
        reference_pdf_id = data.get('reference_pdf', [])
        if use_pdf_id:
            reference_pdf = [repr(pdf_id) for pdf_id in reference_pdf_id]
        else:
            reference_pdf = [repr(uuid2papers[uid]['title']) for uid in reference_pdf_id]
        pdf_context += f"[Reference PDF]: {reference_pdf[0]}\n" if len(reference_pdf) == 1 else \
            f"[Reference PDFs]: [{', '.join(reference_pdf)}]\n"

    # get conference context
    conference = data.get('conference', [])
    if len(conference) > 0:
        pdf_context += f"[Conference]: {conference[0]}" if len(conference) == 1 else \
            f"[Conferences]: [{', '.join(conference)}]"

    # get image message
    # Currently, we do not use images because this significantly decreases the difficulty of the task.
    if use_image and data.get('anchor_image', []):
        template = "[Images]: Here are some images that you can use to answer the question:" if len(data['anchor_image']) > 1 else "[Image]: Here is one image you can use to answer the question:"
        image_messages = get_image_message(
            template=template,
            image_path=data['anchor_image']
        )['content']

    draft_prompt = f"""Generate a high-level, step-by-step plan for me to execute. Crucially, you must write the entire plan in the second person, phrasing each step as a direct command to me. Each command should describe the high-level goal (the "what") for using a specific tool, not the implementation details. Ensure the final command in your plan is `GenerateAnswer`. [Question]: Is there any paper that utilizes masked language modeling to defend against word-level adversarial attacks? [Answer Format]: Your answer should be the title of the paper WITHOUT ANY EXPLANATION. [Conference]: acl2023 [Agent's Draft]: 1. You should first use the `ClassicRetrieve` tool to search for papers matching the core concepts: 'masked language modeling' and 'word-level adversarial attacks'. 2. Next, you should analyze the search results to identify the most promising `pdf_id` and use the `RetrieveFromDatabase` tool to get its official title from the database. 3. If you cannot find the title in the database, you will need to carefully re-examine the text snippets from your search results to infer the most likely paper title. 4. `GenerateAnswer` --- Here is the question you need to plan, combine the functions of the available tools and the analysis of the tasks to provide a planning for this question. [Question]: {data['question']} [Agent's Draft]:"""

    if use_draft:
        task_prompt = '\n'.join([question, answer_format, pdf_context, draft_prompt]).rstrip()
    else:
        task_prompt = '\n'.join([question, answer_format, pdf_context]).rstrip()
    return task_prompt, image_messages, draft_prompt
