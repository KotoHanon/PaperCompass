# Decomposing Scientific Paper Queries with Draft-and-Follow Policy Optimization to Narrow Knowing-Doing Gap

<p align="center">
  <img src="assets/PaperCompass.png" alt="our-framework">
</p>

<details>
  <summary> 💫 Table of Contents (Click to expand)</summary>

- [💡 Main Contributions](#-main-contributions)
- [💻 Environment Setup](#-environment-setup)
- [⬇️ Data Downloading](#-data-downloading)
- [🏃 Quick Start](#-quick-start)
  - [📄 Draft & Tool-Use Fine-Tuning]()
  - [💪🏼 Draft-and-Follow Policy Optimization]()
  - [🔍️ Evaluation]()
- [✍🏻 Citation](#-citation)

</details>

## 💡 Main Contributions
- We introduce PaperCompass,a novel multi-turn RL framework for  training agents for scientific paper querying to address the challenges of paper-based question answering,as well as the inheren ‘knowing-doing’ gap in LLMs.
- We develop a novel RL algorithm specifically for PaperCompass, named Draft-and-Follow Policy Optimization (DFPO). This algorithm facilitates the hierarchical optimization of both the initial draft and the subsequent solution,uniquely achieving this bi-level refinement by maximizing a single objective function.
- We show that PaperCompass significantly out performs existing baselines and achieves performance comparable to a larger model.

## 💻 Environment Setup
First, prepare the corresponding conda environment and install dependencies.
```bash
# Clone the PaperCompass repository (which includes our TRL)
git clone https://github.com/KotoHanon/PaperCompass

# Create a new conda environment
conda create -n papercompass python=3.10
conda activate papercompass

# Install dependencies
pip install -r requirements.txt
```

Then, download the 3B and 7B models that will be used.

```bash
mkdir .cache/qwen2.5-3b-instruct/ .cache/qwen2.5-7b-instruct/

huggingface-cli login
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir qwen2.5-3b-instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir qwen2.5-7b-instruct
```

Last, prepare the following models for vector encoding.

```bash
cd .cache/

git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
git clone https://huggingface.co/BAAI/bge-large-en-v1.5
git clone https://huggingface.co/openai/clip-vit-base-patch32
```


## ⬇️ Data Downloading
1. Download the dataset-related files into the folder `data/dataset` 👉🏻 [HuggingFace 🔗](https://huggingface.co/datasets/OpenDFM/AirQA-Real)
    - [`AirQA-Real`](https://github.com/OpenDFM/NeuSym-RAG): including the `metadata/`, `papers/`, and `processed_data/`
    - [`SciDQA`](https://github.com/yale-nlp/SciDQA): including the `metadata/`, `papers/`, and `processed_data/`

2. Download constructed databases (`.duckdb`) and vectorstores (`.db` and `bm25.json`) into the folders `data/database/` and `data/vectorstore/`, respectively (👉🏻 [HuggingFace 🔗](https://huggingface.co/datasets/OpenDFM/AirQA-Real)). 
    - The 2 dataset name to database / vectorstore name mappings are:

      | Dataset    | Dataset Name  | Database Name       | Vectorstore Name    |
      |:----------:|:-------------:|:-------------------:|:-------------------:|
      | AirQA-Real | `airqa`       | `ai_research`       | `ai_research`       |
      | SciDQA     | `scidqa`      | `openreview_papers` | `openreview_papers` |


## 🏃 Quick Start
In this stage, you need accomplish the following three steps: `Draft \& Tool-Use Fine-Tuning`, `Draft-and-Follow Policy Optimization`, and `Evaluation`.

## 📄 Draft & Tool-Use Fine-Tuning

```
llamafactory-cli train configs/dtft_config.yaml
```

## 💪🏼 Draft-and-Follow Policy Optimization

### 1. API Key Configuration
Our `reward router` utilizes LLM as a judge, so you need to configure the OPENAI_API_KEY and OPENAI_BASE_URL.
```bash
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

### 2. Configure Training
Set `configs/dfpo_train_config.yaml` with the following content:
```bash
max_completion_length: 256
max_draft_length: 512
max_prompt_length: 10000
max_steps: 400
num_generations: <your_num_generations>
per_device_train_batch_size: <your_per_device_train_batch_size>
gradient_accumulation_steps: <your_gradient_accumulation_steps>
max_turn: 20
temperature: 0.7
top_p: 0.95
max_tokens: 1500
window_size: 5
db_format: create_sql
vs_format: detailed_json
action_format: markdown
output_format: json
method: neusym_rag
interact_protocol: react
dataset_name: airqa
db: ai_research
database_dir: data/database/ai_research/
db_type: duckdb
vectorstore: ${db}
launch_method: standalone
vectorstore_dir: data/vectorstore/ai_research/
test_data: <your_data>
name: gpt-4o-mini
model_name_or_path: <your_model_path>
save_strategy: steps
report_to: tensorboard
save_steps: 400
output_dir: <your_output_dir>
logging_dir: <your_logging_dir>
logging_steps: 1
max_grad_norm : 10.0
```

Also, you need to set your accelerate configuration.
```bash
accelerate config
```

### 3. DFPO Training
You can then run the following script to start training:
```
bash scripts/dfpo_train.sh
```

## 🔍️ Evaluation
### 1. Install vLLM for Ascend NPU
```bash
git clone -b npu_support https://github.com/wangshuai09/vllm.git

cd vllm
VLLM_TARGET_DEVICE=npu pip install -e .
```

### 2. Start Evaluation
```bash
bash scripts/papercompass_eval.sh
```


## ✍🏻 Citation

If you find this project useful, please cite our work:
```txt
{
  "TO BE RELEASED"
}
```
