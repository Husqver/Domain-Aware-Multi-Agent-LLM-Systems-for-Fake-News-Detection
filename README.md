# Domain-Aware Multi-Agent LLM Systems for Fake News Detection

Master's thesis project implementing a domain-aware multi-agent fact-checking pipeline based on Large Language Models (LLMs). Rather than relying on a single general-purpose model, the system routes claims to specialized domain experts via a learned domain router, improving fact-checking accuracy through specialization.

---

## Architecture

The system follows a two-tier architecture:

```
Input Claim
    │
    ▼
┌─────────────────┐
│  Domain Router  │  ← LoRA fine-tuned LLM
│                 │    classifies claim into one domain
└────────┬────────┘
         │ domain label
         ▼
┌─────────────────────────────────────┐
│         Domain Expert Pool          │
│                                     │
│  [Politics] [Economy] [Health] ...  │  ← one LoRA adapter per domain
│                                     │
└────────────────┬────────────────────┘
                 │ verdict (True / False)
                 ▼
            Final Prediction
```

**Domain Router** — A LoRA-adapted LLM that maps an input claim to one of the predefined super-domains. Three domain granularities are supported: 5, 8, and 12 domains.

**Domain Experts** — One independent LoRA adapter per domain, each fine-tuned specifically for fact-checking within that domain. Outputs a binary verdict (`True` / `False`) with an optional explanation, enforced via JSON schema (LMFormatEnforcer).

---

## Project Structure

```
.
├── 1_Basics_Masterthesis/          # Baseline experiments & model comparison
│   ├── 1_Basic_Model_Comparison.ipynb
│   ├── LIAR/                       # Preprocessed test sets
│   └── Results/                    # Baseline prediction CSVs
│
├── 2_Finetuning_Masterthesis/      # LoRA fine-tuning (single-model)
│   ├── factcheck_train_lora.py     # Training script
│   ├── factcheck_eval_predict.py   # Evaluation via log-probability scoring
│   ├── 2_Finetuning.ipynb
│   ├── LIAR/                       # Domain-split train/val/test sets
│   └── results/                    # Per-model metrics and predictions
│
├── 3_Multi_Agent_Experiments/      # Multi-agent experiments with weight routing
│   ├── multiagent_factcheck_eval.py
│   ├── multiagent_factcheck_eval_sequential.py
│   ├── weight_routing.py           # Confidence-weighted domain routing
│   ├── evaluation.py
│   ├── 3_Multi_Agent_Experiments.ipynb
│   ├── LIAR/
│   └── Results/
│
├── 4_Final_Pipeline/               # Production-ready domain-aware pipeline
│   ├── train_domain_expert_domain_classifier.py   # Router training
│   ├── train_domain_expert_domain_config.py       # Domain mapping config
│   ├── train_expert_expert_trainer.py             # Expert training
│   ├── train_expert_expert_config.py              # Expert training config
│   ├── multiagent_pipeline.py                     # End-to-end inference
│   ├── multiagent_pipeline_checkability.py        # Variant with checkability scoring
│   ├── evaluate_domainrouter.py                   # Router evaluation
│   ├── evaluate_experts_all.py                    # Expert evaluation
│   ├── evaluate_router_all.py
│   ├── 4_Final_Pipeline.ipynb
│   └── Results/
│
├── Datasets/                       # Centralized dataset storage
│   ├── LIAR/                       # LIAR dataset (train/val/test splits)
│   └── FakeNewsNet/                # GossipCop & PolitiFact subsets
│
└── requirements.txt
```

---

## Models

The following LLMs were evaluated as backbone models for both the router and the domain experts:

| Model                 | HuggingFace ID                       |
| --------------------- | ------------------------------------ |
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` |
| Gemma 7B Instruct     | `google/gemma-7b-it`               |
| DeepSeek LLM 7B       | `deepseek-ai/deepseek-llm-7b-base` |

All models are fine-tuned with **LoRA** (Low-Rank Adaptation) via the PEFT library. No full model weights are modified or stored.

**LoRA configuration:**

- Rank `r = 8`, alpha `= 16–32`, dropout `= 0.05`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj` (auto-detected per architecture)

---

## Dataset

The primary dataset is [LIAR](https://arxiv.org/abs/1705.00648), a benchmark for fake news detection consisting of political statements with fine-grained veracity labels and subject tags.

For this project, the original multi-class labels are reduced to binary (`True` / `False`) and the subject tags are mapped to coarser super-domains using the `DomainConfig` class in `4_Final_Pipeline/train_domain_expert_domain_config.py`.

**Supported domain granularities:**

| k  | Example domains                                                                                                      |
| -- | -------------------------------------------------------------------------------------------------------------------- |
| 5  | socioeconomic\_policy, foreign\_security, governance\_law, environment\_science, society\_culture                    |
| 8  | economy, health\_social, foreign\_security, law\_rights, politics\_government, environment\_energy, society\_culture |
| 12 | macro\_econ, healthcare\_policy, law\_crime\_rights, foreign\_affairs\_security, environment\_energy\_infra, …      |

---

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (experiments run on CUDA 12.9 / `torch==2.9.0+cu129`)

### Installation

**1. Install PyTorch** (choose the version matching your CUDA):

```bash
# CUDA 12.1 example:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# See https://pytorch.org/get-started/locally/ for other versions
```

**2. Install remaining dependencies:**

```bash
pip install -r requirements.txt
```

---

---

## Citation

If you use this code or build on this work, please cite the thesis (details to be added upon publication).

The LIAR dataset:

```
@inproceedings{wang2017liar,
  title={``Liar, Liar Pants on Fire'': A New Benchmark Dataset for Fake News Detection},
  author={Wang, William Yang},
  booktitle={Proceedings of ACL},
  year={2017}
}
```
