# evaluate_router_all.py

import os
import re
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from jinja2.exceptions import TemplateError

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
# =========================
# CONFIG — edit these
# =========================

BASE_MODELS = {
    "llama":    "meta-llama/Llama-3.1-8B-Instruct",
    "gemma":    "google/gemma-7b-it",
    "deepseek": "deepseek-ai/deepseek-llm-7b-base",
}

LORA_PATHS = {
    "llama": {
        5:  "./adapters/llama-liar-statement-domain-lora_5Classes",
        8:  "./adapters/llama-liar-statement-domain-lora_8Classes",
        12: "./adapters/llama-liar-statement-domain-lora_12Classes",
    },
    "gemma": {
        5:  "./adapters/gemma-liar-statement-domain-lora_5Classes",
        8:  "./adapters/gemma-liar-statement-domain-lora_8Classes",
        12: "./adapters/gemma-liar-statement-domain-lora_12Classes",
    },
    "deepseek": {
        5:  "./adapters/deepseek-liar-statement-domain-lora_5Classes",
        8:  "./adapters/deepseek-liar-statement-domain-lora_8Classes",
        12: "./adapters/deepseek-liar-statement-domain-lora_12Classes",
    },
}

SUPER_LABELS_BY_K = {
    5: [
        "socioeconomic_policy",
        "foreign_security",
        "governance_law",
        "environment_science",
        "society_culture",
        "misc",
    ],
    8: [
        "economy",
        "health_social",
        "foreign_security",
        "law_rights",
        "politics_government",
        "environment_energy",
        "society_culture",
        "misc",
    ],
    12: [
        "macro_econ",
        "jobs_labor",
        "business_finance",
        "cost_of_living_housing",
        "healthcare_policy",
        "public_health_crises",
        "substances_gambling",
        "family_demographics",
        "law_crime_rights",
        "institutions_elections",
        "foreign_affairs_security",
        "environment_energy_infra",
        "media_meta",
    ],
}

TEST_PATHS = {
    5:  "Results/preprocessed_test_cleaned_binary_with_super_domain_5.csv",
    8:  "Results/preprocessed_test_cleaned_binary_with_super_domain_8.csv",
    12: "Results/preprocessed_test_cleaned_binary_with_super_domain_12.csv",
}
TEST_SEP = "\t"

GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

os.makedirs("Results", exist_ok=True)

# =========================
# Helpers
# =========================

def render_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    has_chat = hasattr(tokenizer, "apply_chat_template") and bool(
        getattr(tokenizer, "chat_template", None)
    )
    if has_chat:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
        except (TemplateError, Exception):
            merged = f"{system_prompt}\n\n{user_prompt}"
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": merged}],
                tokenize=False, add_generation_prompt=False,
            )
    return f"{system_prompt}\n\n{user_prompt}\n\n"


def build_system_prompt(labels) -> str:
    lines = (
        ["You are a strict domain classifier.\n",
         "Map the claim to exactly ONE label from this list:\n"]
        + [f"- {l}" for l in labels]
        + ["\nOutput ONLY the label. No explanation."]
    )
    return "\n".join(lines)


def classify(statement: str, model, tokenizer, labels, max_new_tokens: int = 8) -> str:
    system_prompt = build_system_prompt(labels)
    user_content = (
        f"Valid labels:\n{', '.join(labels)}\n\n"
        f"Claim:\n{statement}\n\n"
        "Return ONLY ONE label from the list above.\n"
        "Label:"
    )
    prompt = render_prompt(tokenizer, system_prompt, user_content)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        dev    = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    answer  = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()

    if not answer:
        return "misc" if "misc" in labels else labels[0]
    for L in labels:
        if re.search(r"\b" + re.escape(L.lower()) + r"\b", answer):
            return L
    for L in labels:
        if L.lower() in answer:
            return L
    return "misc" if "misc" in labels else labels[0]


def load_model(base_model_id: str, lora_dir: str):
    lora_dir = os.path.abspath(lora_dir)
    dtype = (
        torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16 if torch.cuda.is_available()
        else torch.float32
    )
    print(f"  Loading base:  {base_model_id}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    print(f"  Loading LoRA:  {lora_dir}")
    model = PeftModel.from_pretrained(base, lora_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(lora_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_test_df(k: int, labels) -> pd.DataFrame:
    df = pd.read_csv(TEST_PATHS[k], sep=TEST_SEP, quotechar='"', engine="python", dtype=str)
    df = df.dropna(subset=["statement", "super_domain"])
    df = df[df["statement"].str.strip() != ""]
    df = df[df["super_domain"].isin(labels)]
    return df.reset_index(drop=True)


def print_and_collect_metrics(df, labels, backbone, k, results_rows):
    y_true = df["super_domain"]
    y_pred = df["pred"]

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\n{'='*70}")
    print(f"  {backbone.upper()} | k={k}")
    print(f"{'='*70}")
    print(f"  Test set size : {len(df)}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"\nClassification report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print(f"Confusion matrix (rows=true, cols=pred):")
    print(f"Label order: {labels}")
    print(cm)

    # Save per-model predictions
    out_csv = f"Results/router_eval_{backbone}_k{k}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  💾 Predictions saved: {out_csv}")

    results_rows.append({
        "backbone": backbone,
        "k": k,
        "n_eval": len(df),
        "accuracy": round(acc, 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
    })


# =========================
# Main
# =========================

def main():
    results_rows = []   # one row per (backbone, k)
    all_preds    = []   # for backbone-level summary

    for backbone in ["llama", "gemma", "deepseek"]:
        backbone_rows = []  # collect all preds across k for this backbone

        for k in [5, 8, 12]:
            labels   = SUPER_LABELS_BY_K[k]
            lora_dir = LORA_PATHS[backbone][k]

            print(f"\n\n{'#'*70}")
            print(f"  Evaluating: backbone={backbone}  k={k}")
            print(f"{'#'*70}")

            model, tokenizer = load_model(BASE_MODELS[backbone], lora_dir)

            df = load_test_df(k, labels)
            print(f"  Test set size (filtered): {len(df)}")

            preds = []
            for row in tqdm(df.itertuples(), total=len(df),
                            desc=f"{backbone} k={k}", dynamic_ncols=True):
                pred = classify(row.statement, model, tokenizer, labels)
                preds.append(pred)

            df["pred"]     = preds
            df["backbone"] = backbone
            df["k"]        = k

            print_and_collect_metrics(df, labels, backbone, k, results_rows)

            backbone_rows.append(df)
            all_preds.append(df)

            del model
            torch.cuda.empty_cache()

        # ---- Per-backbone summary (across all k) ----
        df_bb = pd.concat(backbone_rows, ignore_index=True)
        acc_bb = accuracy_score(df_bb["super_domain"], df_bb["pred"])
        print(f"\n{'*'*70}")
        print(f"  BACKBONE SUMMARY: {backbone.upper()} (all k combined)")
        print(f"  Total samples : {len(df_bb)}")
        print(f"  Overall acc   : {acc_bb:.4f}")
        print(f"{'*'*70}")

    # ---- Final summary table ----
    summary = pd.DataFrame(results_rows)
    print(f"\n\n{'='*70}")
    print("FULL SUMMARY (all models)")
    print("="*70)
    print(summary.to_string(index=False))

    # Backbone-level aggregation
    bb_summary = (
        summary.groupby("backbone")
        .agg(
            total_samples=("n_eval", "sum"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
        )
        .round(4)
        .reset_index()
    )
    print(f"\nBACKBONE-LEVEL SUMMARY (mean across k=5,8,12):")
    print(bb_summary.to_string(index=False))

    # Save summaries
    summary.to_csv("Results/router_eval_summary_per_model.csv", index=False)
    bb_summary.to_csv("Results/router_eval_summary_per_backbone.csv", index=False)
    print("\n💾 Summaries saved to Results/")


if __name__ == "__main__":
    main()