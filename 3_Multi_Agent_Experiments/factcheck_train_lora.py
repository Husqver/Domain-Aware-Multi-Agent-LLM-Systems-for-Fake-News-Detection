# factcheck_train_lora.py
import os
import argparse
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ------------------------
# Reproducibility
# ------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ------------------------
# Prompt templates (STANDARD)
# ------------------------
def prompt_standard(statement: str, subjects: str = "", use_subjects: bool = False) -> str:
    if use_subjects:
        return (
            f"Subjects: {subjects}\n"
            f"Statement: {statement}\n"
            f"Task: Verify the factual accuracy of the statement using only publicly available and objective information. "
            f"Use the subjects only as context clues; they may be incomplete. Do not guess.\n"
            f"Answer with only one word: True or False.\n"
            f"Answer:"
        )
    else:
        return (
            f"Statement: {statement}\n"
            f"Task: Verify the factual accuracy of the statement using only publicly available and objective information. "
            f"Do not guess.\n"
            f"Answer with only one word: True or False.\n"
            f"Answer:"
        )

def prompt_p1(statement: str, subjects: str = "", use_subjects: bool = False) -> str:
    base = (
        f"Statement: {statement}\n"
        f"Task: Determine whether the statement is factually correct or fake news. "
        f"Use only publicly verifiable and objective facts. Do not guess.\n"
        f"Answer with only one word: True or False.\n"
        f"Answer:"
    )
    return (f"Subjects: {subjects}\n" + base) if use_subjects else base

PROMPTS = {
    "standard": prompt_standard,
    "p1": prompt_p1,
}

# ------------------------
# Chat template (if available)
# ------------------------
def maybe_apply_chat_template(tokenizer, user_prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": user_prompt}]
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return user_prompt
    return user_prompt

# ------------------------
# Completion-only tokenization
# Labels = -100 for prompt tokens, normal ids for answer tokens
# ------------------------
def tokenize_completion_only(tokenizer, prompt_text: str, answer_text: str, max_len: int):
    full = prompt_text + " " + answer_text + (tokenizer.eos_token or "")
    enc_full = tokenizer(full, truncation=True, max_length=max_len, add_special_tokens=False)

    enc_prompt = tokenizer(prompt_text, truncation=True, max_length=max_len, add_special_tokens=False)
    prompt_len = len(enc_prompt["input_ids"])

    labels = [-100] * len(enc_full["input_ids"])
    for i in range(prompt_len, len(labels)):
        labels[i] = enc_full["input_ids"][i]

    enc_full["labels"] = labels
    return enc_full

def normalize_label(x):
    """
    Akzeptiert True/False als string/bool/int und gibt 1/0 zurück.
    """
    if x is None:
        raise ValueError("label is None")

    # bool direkt
    if isinstance(x, bool):
        return 1 if x else 0

    # int/float
    if isinstance(x, (int, float)):
        return 1 if int(x) == 1 else 0

    s = str(x).strip().lower()

    if s in ["true", "t", "yes", "y", "1"]:
        return 1
    if s in ["false", "f", "no", "n", "0"]:
        return 0

    raise ValueError(f"Unknown label value: {x}")
    
def sniff_delimiter(path, n=5):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [f.readline() for _ in range(n)]
    sample = "".join(lines)
    # simple heuristic: choose delimiter with most occurrences
    candidates = ["\t", ",", ";"]
    counts = {c: sample.count(c) for c in candidates}
    return max(counts, key=counts.get)


# ------------------------
# Guess LoRA target modules (robust across llama/gemma/deepseek)
# ------------------------
def guess_lora_targets(model) -> list[str]:
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj"]
    present = set()
    for name, _ in model.named_modules():
        for c in candidates:
            if name.endswith(c):
                present.add(c)
    return sorted(list(present)) if present else ["q_proj", "v_proj"]

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_name", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--prompt_id", choices=list(PROMPTS.keys()), default="standard")
    ap.add_argument("--use_subjects", action="store_true")

    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--val_split", type=float, default=0.05)

    # QLoRA
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")

    # LoRA params
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    args = ap.parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    delim = sniff_delimiter(args.train_csv)
    ds = load_dataset(
        "csv",
        data_files={"train": args.train_csv},
        delimiter=delim
    )["train"]
    print("Using delimiter:", repr(delim))

    if args.val_split and args.val_split > 0:
        split = ds.train_test_split(test_size=args.val_split, seed=args.seed)
        train_ds, val_ds = split["train"], split["test"]
    else:
        train_ds, val_ds = ds, None

    prompt_fn = PROMPTS[args.prompt_id]

    def map_fn(ex):
        statement = str(ex["statement"])
        subjects = str(ex.get("subjects", ex.get("subject", "")))
        label = normalize_label(ex["label"])
        answer = "True" if label == 1 else "False"


        user_prompt = prompt_fn(statement, subjects, args.use_subjects)
        rendered = maybe_apply_chat_template(tokenizer, user_prompt)

        return tokenize_completion_only(tokenizer, rendered, answer, args.max_len)

    train_tok = train_ds.map(map_fn, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(map_fn, remove_columns=val_ds.column_names) if val_ds is not None else None

    quant_cfg = None
    if args.use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_cfg,
        trust_remote_code=args.trust_remote_code,
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()


    if args.use_4bit:
        base_model = prepare_model_for_kbit_training(base_model)

    lora_targets = guess_lora_targets(base_model)
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_targets,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, peft_cfg)

    # Dynamic padding collator
    def data_collator(features):
        labels = [f["labels"] for f in features]
        for f in features:
            f.pop("labels")
    
        # input_ids/attention_mask padden
        batch = tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
    
        max_len = batch["input_ids"].shape[1]
    
        padded_labels = []
        for lab in labels:
            if len(lab) < max_len:
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            padded_labels.append(lab)
    
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=False,
        bf16=True,
        optim="adamw_torch",
        report_to="none",
        logging_steps=25,
    
        save_strategy="steps",
        save_steps=10,
        eval_strategy="steps" if val_tok is not None else "no",
        eval_steps=10 if val_tok is not None else None, 
    
        load_best_model_at_end=True if val_tok is not None else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    best_dir = os.path.join(args.output_dir, "best_adapter")
    os.makedirs(best_dir, exist_ok=True)
    trainer.model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    print("Saved BEST adapter to:", best_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Saved adapter + tokenizer to:", args.output_dir)

if __name__ == "__main__":
    main()
