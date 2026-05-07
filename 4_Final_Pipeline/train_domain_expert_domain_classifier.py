import os

# =========================
# 0) Environment 
# =========================
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  

import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from train_domain_expert_domain_config import DomainConfig

import torch
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from jinja2.exceptions import TemplateError

# =========================
# 2) Utility: dtype + prompt building
# =========================
def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        # Prefer bf16 if supported, else fp16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def create_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    return tok

def render_prompt(tokenizer, system_prompt: str, user_prompt: str, add_generation_prompt: bool) -> str:
    has_chat = hasattr(tokenizer, "apply_chat_template") and bool(getattr(tokenizer, "chat_template", None))

    if has_chat:
        # Try normal system+user
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except (TemplateError, Exception):
            # Some templates (e.g., certain Gemma/others) reject "system"
            merged = f"{system_prompt}\n\n{user_prompt}"
            msgs2 = [{"role": "user", "content": merged}]
            return tokenizer.apply_chat_template(
                msgs2, tokenize=False, add_generation_prompt=add_generation_prompt
            )

    # Plain fallback (e.g., deepseek-llm-7b-base)
    if add_generation_prompt:
        return f"{system_prompt}\n\n{user_prompt}\n\nLabel:"
    return f"{system_prompt}\n\n{user_prompt}\n\n"


# =========================
# 3) Data loading + tokenization
# =========================
def load_liar_dataset(
    path: str,
    domain_config: DomainConfig,
    text_col: str = "statement",
    subjects_col: str = "subjects",
) -> pd.DataFrame:
    """
    Read LIAR-like TSV and add super_domain from subjects mapping.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        quotechar='"',
        engine="python",
        dtype=str,
    )

    df["super_domain"] = df[subjects_col].apply(domain_config.map_subjects_to_super)
    df = df.dropna(subset=[text_col, "super_domain"])
    df = df[df[text_col].str.strip() != ""]
    return df


def tokenize_dataset(
    df: pd.DataFrame,
    system_prompt: str,
    tokenizer,
    max_length: int = 256,
    text_col: str = "statement",
    label_col: str = "super_domain",
) -> Dataset:
    raw_ds = Dataset.from_pandas(df[[text_col, label_col]])

    def tokenize_example(example):
        user_content = (
            f"Claim:\n{example[text_col]}\n\n"
            "Return ONLY ONE label from the valid list.\n"
            "Label:"
        )

        prompt = render_prompt(
            tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_content,
            add_generation_prompt=False,
        )

        text = prompt + " " + str(example[label_col]) + (tokenizer.eos_token if tokenizer.eos_token else "")

        return tokenizer(text, max_length=max_length, truncation=True)

    return raw_ds.map(tokenize_example, batched=False, remove_columns=raw_ds.column_names)


# =========================
# 4) LoRA model creation (robust target module detection)
# =========================
def infer_lora_target_modules(model) -> List[str]:
    """
    Try to infer common projection module names present in the architecture.
    Works well for Llama/Gemma/DeepSeek-style transformer blocks.
    """
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present_suffixes = set()
    for n, _ in model.named_modules():
        present_suffixes.add(n.split(".")[-1])
    found = [m for m in candidates if m in present_suffixes]
    return found if found else candidates


def create_lora_model(base_model_id: str) -> AutoModelForCausalLM:
    dtype = pick_dtype()
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    target_modules = infer_lora_target_modules(base)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(base, lora_config)
    return model


# =========================
# 5) Training
# =========================
def train_domain_classifier(
    train_path: str,
    domain_config: DomainConfig,
    base_model_id: str,
    output_dir: str,
    num_train_epochs: int = 3,
    learning_rate: float = 5e-4,
    per_device_train_batch_size: int = 4,
    max_length: int = 256,
    text_col: str = "statement",
    subjects_col: str = "subjects",
):
    """
    End-to-end training of a LoRA adapter for domain routing.
    """
    df_train = load_liar_dataset(train_path, domain_config, text_col=text_col, subjects_col=subjects_col)
    print("Super label distribution (train):")
    print(df_train["super_domain"].value_counts())

    system_prompt = domain_config.build_system_prompt()

    tokenizer = create_tokenizer(base_model_id)
    train_dataset = tokenize_dataset(
        df_train,
        system_prompt=system_prompt,
        tokenizer=tokenizer,
        max_length=max_length,
        text_col=text_col,
        label_col="super_domain",
    )

    print("Example tokenized sample keys:", train_dataset[0].keys())
    print("Input length:", len(train_dataset[0]["input_ids"]))

    model = create_lora_model(base_model_id)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=(torch.cuda.is_available() and pick_dtype() == torch.float16),
        bf16=(torch.cuda.is_available() and pick_dtype() == torch.bfloat16),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir) 
    tokenizer.save_pretrained(output_dir)

    print(f"✅ LoRA adapter & tokenizer saved to: {output_dir}")


# =========================
# 6) Loading + Inference
# =========================
def load_lora_model_and_tokenizer(
    base_model_id: str,
    lora_dir: str,
) -> Tuple[PeftModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(lora_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = pick_dtype()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, lora_dir)
    model.eval()
    return model, tokenizer


def classify_statement_domain(statement, model, tokenizer, domain_config, max_new_tokens=8):
    label_list = ", ".join(domain_config.super_labels)
    system_prompt = domain_config.build_system_prompt()

    user_content = (
        f"Valid labels:\n{label_list}\n\n"
        f"Claim:\n{statement}\n\n"
        "Return ONLY ONE label from the list above.\n"
        "Label:"
    )

    prompt = render_prompt(tokenizer, system_prompt, user_content, add_generation_prompt=False)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    if torch.cuda.is_available():
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()

    if not answer:
        return "misc" if "misc" in domain_config.super_labels else domain_config.super_labels[0]

    for L in domain_config.super_labels:
        if re.search(r"\b" + re.escape(L.lower()) + r"\b", answer):
            return L
    for L in domain_config.super_labels:
        if L.lower() in answer:
            return L
    return "misc" if "misc" in domain_config.super_labels else domain_config.super_labels[0]



# =========================
# 7) Evaluation
# =========================
def evaluate_on_testset(
    test_path: str,
    domain_config: DomainConfig,
    base_model_id: str,
    lora_dir: str,
    text_col: str = "statement",
    subjects_col: str = "subjects",
):
    """
    Evaluate routing agreement vs subject-derived labels.
    """
    df_test = pd.read_csv(
        test_path,
        sep="\t",
        quotechar='"',
        engine="python",
        dtype=str,
    )

    df_test = df_test.dropna(subset=[text_col, subjects_col])
    df_test = df_test[df_test[text_col].str.strip() != ""]

    print("Test set size:", len(df_test))

    df_test["super_domain_true"] = df_test[subjects_col].apply(domain_config.map_subjects_to_super)
    print("True label distribution (test):")
    print(df_test["super_domain_true"].value_counts())

    model, tokenizer = load_lora_model_and_tokenizer(base_model_id, lora_dir)

    preds: List[str] = []
    bar = tqdm(
        df_test.itertuples(),
        total=len(df_test),
        desc="🧠 Routing statements",
        dynamic_ncols=True,
        smoothing=0.1,
        mininterval=0.1,
    )

    for row in bar:
        claim = getattr(row, text_col)
        pred_label = classify_statement_domain(
            statement=claim,
            model=model,
            tokenizer=tokenizer,
            domain_config=domain_config,
        )
        preds.append(pred_label)

    df_test["super_domain_pred"] = preds

    print("Predicted label distribution (test):")
    print(df_test["super_domain_pred"].value_counts())

    y_true = df_test["super_domain_true"]
    y_pred = df_test["super_domain_pred"]

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy (super_domain) on test set: {acc:.3f}\n")

    print("Classification report (per superlabel):")
    print(classification_report(y_true, y_pred, labels=domain_config.super_labels, zero_division=0))

    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred, labels=domain_config.super_labels)
    print("Labels order:", domain_config.super_labels)
    print(cm)

    return df_test


# =========================
# 8) Example usage (edit these)
# =========================
if __name__ == "__main__":


    base_model_id = "meta-llama/Llama-3.1-8B-Instruct"

    domain_to_super = {
        "taxes": "economy",
        "budget": "economy",
        "health-care": "health_social",
        "foreign-policy": "foreign_security",
        "immigration": "law_rights",
        "elections": "politics_government",
        "energy": "environment_energy",
        "education": "society_culture",
        "misc": "misc",
    }

    domain_config = DomainConfig(
        domain_to_super=domain_to_super,
        super_labels=[
            "economy",
            "health_social",
            "foreign_security",
            "law_rights",
            "politics_government",
            "environment_energy",
            "society_culture",
            "misc",
        ],
    )

    train_path = "own_datasets/train_binary_labels_balanced.csv"
    test_path = "own_datasets/test_binary_labels.csv"
    output_dir = "./Multi_Agent_Models/router_lora_8classes"

    # Train
    train_domain_classifier(
        train_path=train_path,
        domain_config=domain_config,
        base_model_id=base_model_id,
        output_dir=output_dir,
        num_train_epochs=3,
        learning_rate=5e-4,
        per_device_train_batch_size=4,
        max_length=256,
    )

    # Eval
    evaluate_on_testset(
        test_path=test_path,
        domain_config=domain_config,
        base_model_id=base_model_id,
        lora_dir=output_dir,
    )
