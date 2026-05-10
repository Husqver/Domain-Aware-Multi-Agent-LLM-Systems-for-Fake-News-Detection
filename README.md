# Domain-Aware Multi-Agent LLM Systems for Fake News Detection

Master's thesis project implementing a domain-aware multi-agent
fact-checking pipeline based on Large Language Models (LLMs). Rather
than relying on a single general-purpose model, the system routes claims
to specialized domain experts via a learned domain router, improving
fact-checking accuracy through specialization.

---

## Thesis

This repository accompanies the Master's thesis:

**Domain-Aware Multi-Agent LLM Systems for Fake News Detection**

Lukas Rupp, OTH Amberg-Weiden, Faculty of Electrical Engineering,
Media and Computer Science, 2026

---

## Architecture

The system follows a three-stage pipeline:

```
Input Claim
    │
    ▼
┌─────────────────────┐
│  Checkability Gate  │  ← filters non-verifiable or underspecified claims
└────────┬────────────┘
         │ fact_checkable claims only
         ▼
┌─────────────────┐
│  Domain Router  │  ← LoRA fine-tuned LLM
│                 │    classifies claim into one super-domain
└────────┬────────┘
         │ domain label
         ▼
┌─────────────────────────────────────┐
│         Domain Expert Pool          │
│                                     │
│  [Economy] [Politics] [Health] ...  │  ← one LoRA adapter per domain
│                                     │
└────────────────┬────────────────────┘
                 │ verdict (True / False)
                 ▼
            Final Prediction
```

**Checkability Gate** — Filters non-verifiable inputs before routing.
Classifies each claim into one of five categories: `fact_checkable`,
`non_claim`, `opinion_or_ambiguous`, `needs_additional_context`, or
`sensitive_selfharm`. Only `fact_checkable` claims are forwarded to
the router and expert agents.

**Domain Router** — A LoRA-adapted LLM that maps an input claim to one
of the predefined super-domains. Three domain granularities are
supported: 6, 8, and 13 domains.

**Domain Experts** — One independent LoRA adapter per domain, each
fine-tuned specifically for fact-checking within that domain. Outputs
a binary verdict (`True` / `False`) with an optional explanation,
enforced via JSON schema (LMFormatEnforcer).

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
│   ├── multiagent_pipeline_checkability.py        # Variant with checkability gate
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

The following LLMs were evaluated as backbone models for both the
router and the domain experts:

| Model                 | HuggingFace ID                       |
| --------------------- | ------------------------------------ |
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` |
| Gemma 7B Instruct     | `google/gemma-7b-it`               |
| DeepSeek LLM 7B       | `deepseek-ai/deepseek-llm-7b-base` |

All models are fine-tuned with **LoRA** (Low-Rank Adaptation) via the
PEFT library. No full model weights are modified or stored.

**LoRA configuration:**

* Rank `r = 8`, alpha `= 16-32`, dropout `= 0.05`
* Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj`

---

## Dataset

The primary dataset is [LIAR](https://arxiv.org/abs/1705.00648), a
benchmark for fake news detection consisting of short political
statements with fine-grained veracity labels and subject tags.

For this project, the original six-class labels are reduced to binary
(`True` / `False`) using a deterministic mapping.

**Supported domain granularities:**

| k  | Example domains                                                                                                      |
| -- | -------------------------------------------------------------------------------------------------------------------- |
| 6  | socioeconomic_policy, foreign_security, governance_law, environment_science, society_culture, misc                   |
| 8  | economy, health_social, foreign_security, law_rights, politics_government, environment_energy, society_culture, misc |
| 13 | macro_econ, healthcare_policy, law_crime_rights, foreign_affairs_security, environment_energy_infra, media_meta, ... |

---

## Setup

### Requirements

* Python 3.10+
* CUDA-capable GPU (experiments run on CUDA 12.9 / `torch==2.9.0+cu129`)

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


## Usage

The experiments are organized as Jupyter Notebooks and build on each other:

| Notebook                                                      | Description                                               |
| ------------------------------------------------------------- | --------------------------------------------------------- |
| `1_Basics_Masterthesis/1_Basic_Model_Comparison.ipynb`      | Baseline comparison of different LLMs without fine-tuning |
| `2_Finetuning_Masterthesis/2_Finetuning.ipynb`              | LoRA fine-tuning of models on the LIAR dataset            |
| `3_Multi_Agent_Experiments/3_Multi_Agent_Experiments.ipynb` | Multi-agent experiments with confidence-weighted routing  |
| `4_Final_Pipeline/4_Final_Pipeline.ipynb`                   | Final domain-aware pipeline with checkability gate        |

### Start Notebooks

```bash
jupyter lab
```

The Python scripts in `4_Final_Pipeline/` can also be run directly for training or evaluation outside the notebook environment.

## Citation

If you use this code or build on this work, please cite:

```bibtex
@mastersthesis{rupp2026domainaware,
  title     = {Domain-Aware Multi-Agent LLM Systems for Fake News Detection},
  author    = {Rupp, Lukas},
  school    = {OTH Amberg-Weiden},
  year      = {2026}
}
```

The LIAR dataset:

```bibtex
@inproceedings{wang2017liar,
  title     = {"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection},
  author    = {Wang, William Yang},
  booktitle = {Proceedings of ACL},
  year      = {2017}
}
```
