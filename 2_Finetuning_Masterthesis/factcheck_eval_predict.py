# factcheck_eval_predict.py
import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # sichtbare GPUs

# ------------------------
# Same prompts as training
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

PROMPTS = {"standard": prompt_standard, "p1": prompt_p1}

def maybe_apply_chat_template(tokenizer, user_prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": user_prompt}]
        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return user_prompt
    return user_prompt

# ------------------------
# Helper: logprob of continuation tokens given prompt
# Robust method: one forward pass on (prompt + continuation)
# ------------------------
@torch.no_grad()
def continuation_logprob(model, tokenizer, prompt_text: str, continuation_tokens: list[int], max_len: int = 1024) -> float:
    # Build full ids = prompt_ids + continuation_ids
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = prompt_ids + continuation_tokens
    if len(full_ids) > max_len:
        # truncate from left (keep end) to avoid crash
        full_ids = full_ids[-max_len:]
        # prompt_len unknown after trunc; approximate by clipping prompt too
        # For stability, we re-derive prompt_len as len(full)-len(cont)
        prompt_len = len(full_ids) - len(continuation_tokens)
    else:
        prompt_len = len(prompt_ids)

    input_ids = torch.tensor([full_ids], device=model.device)
    attention_mask = torch.ones_like(input_ids)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits  # [1, seq, vocab]
    log_probs = torch.log_softmax(logits, dim=-1)

    # logprob of continuation token i is predicted at position (prompt_len-1 + i)
    total = 0.0
    for i, tok in enumerate(continuation_tokens):
        pos = (prompt_len - 1) + i
        if pos < 0 or pos >= log_probs.shape[1]:
            continue
        total += float(log_probs[0, pos, tok].item())
    return total

def encode_choice(tokenizer, word: str) -> list[int]:
    # prefer leading space for BPE tokenizers
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(ids) == 0:
        ids = tokenizer.encode(word, add_special_tokens=False)
    return ids

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


def softmax2(a, b):
    m = max(a, b)
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    s = ea + eb
    return ea / s, eb / s

def sniff_delimiter(path, n=5):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [f.readline() for _ in range(n)]
    sample = "".join(lines)
    # simple heuristic: choose delimiter with most occurrences
    candidates = ["\t", ",", ";"]
    counts = {c: sample.count(c) for c in candidates}
    return max(counts, key=counts.get)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--prompt_id", choices=list(PROMPTS.keys()), default="standard")
    ap.add_argument("--use_subjects", action="store_true")

    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--max_len_prompt", type=int, default=1024)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if args.use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant_cfg,
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    delim = sniff_delimiter(args.test_csv)
    ds = load_dataset("csv", data_files={"test": args.test_csv}, delimiter=delim)["test"]
    if args.max_items and args.max_items > 0:
        ds = ds.select(range(min(args.max_items, len(ds))))

    prompt_fn = PROMPTS[args.prompt_id]
    true_ids = encode_choice(tokenizer, "True")
    false_ids = encode_choice(tokenizer, "False")

    rows = []
    y_true, y_pred = [], []
    confs, confs_correct, confs_wrong = [], [], []

    for ex in ds:
        statement = str(ex["statement"])
        subjects = str(ex.get("subjects", ex.get("subject", "")))
        gold = normalize_label(ex["label"])

        user_prompt = prompt_fn(statement, subjects, args.use_subjects)
        rendered = maybe_apply_chat_template(tokenizer, user_prompt)

        lp_true = continuation_logprob(model, tokenizer, rendered, true_ids, max_len=args.max_len_prompt)
        lp_false = continuation_logprob(model, tokenizer, rendered, false_ids, max_len=args.max_len_prompt)

        p_true, p_false = softmax2(lp_true, lp_false)
        pred = 1 if p_true >= p_false else 0
        pred_str = "True" if pred == 1 else "False"

        confidence = max(p_true, p_false)
        margin = abs(p_true - p_false)
        correct = 1 if pred == gold else 0

        y_true.append(gold)
        y_pred.append(pred)

        confs.append(confidence)
        (confs_correct if correct else confs_wrong).append(confidence)

        rows.append({
            "statement": statement,
            "subjects": subjects,
            "label": gold,
            "pred_bin": pred,
            "pred": pred_str,
            "p_true": p_true,
            "p_false": p_false,
            "confidence": confidence,
            "margin": margin,
            "correct": correct,
            "logp_true": lp_true,
            "logp_false": lp_false,
        })

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "precision_pos": float(prec),
        "recall_pos": float(rec),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "mean_confidence": float(np.mean(confs)) if confs else None,
        "mean_conf_correct": float(np.mean(confs_correct)) if confs_correct else None,
        "mean_conf_wrong": float(np.mean(confs_wrong)) if confs_wrong else None,
        "prompt_id": args.prompt_id,
        "use_subjects": bool(args.use_subjects),
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "test_csv": args.test_csv,
    }

    # Save
    pred_path = os.path.join(args.out_dir, "predictions.csv")
    met_path = os.path.join(args.out_dir, "metrics.json")

    pd.DataFrame(rows).to_csv(pred_path, index=False)
    with open(met_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Pretty print
    print(json.dumps(metrics, indent=2))
    print("Saved predictions:", pred_path)
    print("Saved metrics:", met_path)

if __name__ == "__main__":
    main()
