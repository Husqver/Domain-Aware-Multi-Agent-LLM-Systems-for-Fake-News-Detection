# evaluate_experts_all.py

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
# CONFIG
# =========================

BASE_MODELS = {
    "llama":    "meta-llama/Llama-3.1-8B-Instruct",
    "gemma":    "google/gemma-7b-it",
    "deepseek": "deepseek-ai/deepseek-llm-7b-base",
}

EXPERT_ROOTS = {
    "llama": {
        5:  "./adapters/Llama_experts_5Classes",
        8:  "./adapters/Llama_experts_8Classes",
        12: "./adapters/Llama_experts_12Classes",
    },
    "gemma": {
        5:  "./adapters/gemma_experts_5Classes",
        8:  "./adapters/gemma_experts_8Classes",
        12: "./adapters/gemma_experts_12Classes",
    },
    "deepseek": {
        5:  "./adapters/deepseek_experts_5Classes",
        8:  "./adapters/deepseek_experts_8Classes",
        12: "./adapters/deepseek_experts_12Classes",
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
                msgs, tokenize=False, add_generation_prompt=True
            )
        except (TemplateError, Exception):
            merged = f"{system_prompt}\n\n{user_prompt}"
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": merged}],
                tokenize=False, add_generation_prompt=True,
            )
    return f"{system_prompt}\n\n{user_prompt}\n\nAnswer:"


def build_system_prompt(domain_name: str) -> str:
    return (
        f"You are a fact-checking expert specialized in the '{domain_name}' domain.\n"
        "You receive short political or public policy claims (statements) and must decide "
        "if they are factually correct.\n"
        "Answer STRICTLY with one of these two labels:\n"
        "- True  (the claim is factually correct)\n"
        "- False (the claim is factually incorrect)\n"
        "Do NOT explain. Do NOT add any extra text. Only output 'True' or 'False'."
    )


def norm_bool_label(answer: str) -> str:
    first = answer.split()[0].strip().strip(".,:;!?)(").lower() if answer.strip() else ""
    if first.startswith("true"):
        return "True"
    if first.startswith("false"):
        return "False"
    return "False"


def predict_verdict(statement: str, domain_name: str, model, tokenizer) -> str:
    system_prompt = build_system_prompt(domain_name)
    user_content  = (
        f"Claim:\n{statement}\n\n"
        "Is this claim factually correct? Answer strictly with 'True' or 'False'."
    )
    prompt = render_prompt(tokenizer, system_prompt, user_content)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)

    if torch.cuda.is_available():
        dev    = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    answer  = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return norm_bool_label(answer)


def load_expert(base_model_id: str, expert_dir: str):
    expert_dir = os.path.abspath(expert_dir)
    dtype = (
        torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16 if torch.cuda.is_available()
        else torch.float32
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, expert_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(expert_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def load_test_df(k: int) -> pd.DataFrame:
    df = pd.read_csv(TEST_PATHS[k], sep=TEST_SEP, quotechar='"', engine="python", dtype=str)
    df = df.dropna(subset=["statement", "label", "super_domain"])
    df = df[df["statement"].str.strip() != ""]
    df = df[df["label"].isin(["True", "False"])]
    return df.reset_index(drop=True)


# =========================
# Main
# =========================

def main():
    results_rows = []   # one row per (backbone, k, domain)
    summary_rows = []   # one row per (backbone, k) — aggregated

    for backbone in ["llama", "gemma", "deepseek"]:
        for k in [5, 8, 12]:
            domains      = SUPER_LABELS_BY_K[k]
            expert_root  = EXPERT_ROOTS[backbone][k]
            base_model   = BASE_MODELS[backbone]

            print(f"\n\n{'#'*70}")
            print(f"  Experts: backbone={backbone}  k={k}")
            print(f"  Root: {expert_root}")
            print(f"{'#'*70}")

            df_test = load_test_df(k)
            print(f"  Full test set size: {len(df_test)}")

            all_preds_this_k  = []   # collect all rows across domains for this (backbone, k)

            for domain in domains:
                expert_dir = os.path.join(expert_root, domain)
                if not os.path.isdir(os.path.abspath(expert_dir)):
                    print(f"\n  ⚠️  No expert found for '{domain}' at {expert_dir} — skipping.")
                    continue

                # Filter test set to this domain (ground truth)
                df_dom = df_test[df_test["super_domain"] == domain].copy()
                if df_dom.empty:
                    print(f"\n  ⚠️  No test samples for domain '{domain}' — skipping.")
                    continue

                print(f"\n  {'='*60}")
                print(f"  Domain: {domain}  |  n={len(df_dom)}")
                print(f"  {'='*60}")
                print(f"  Label distribution: {df_dom['label'].value_counts().to_dict()}")

                model, tokenizer = load_expert(base_model, expert_dir)

                preds = []
                for row in tqdm(df_dom.itertuples(), total=len(df_dom),
                                desc=f"{backbone} k={k} {domain}", dynamic_ncols=True):
                    pred = predict_verdict(row.statement, domain, model, tokenizer)
                    preds.append(pred)

                df_dom["pred"]     = preds
                df_dom["backbone"] = backbone
                df_dom["k"]        = k

                y_true = df_dom["label"]
                y_pred = df_dom["pred"]

                acc    = accuracy_score(y_true, y_pred)
                report = classification_report(
                    y_true, y_pred, labels=["True", "False"], zero_division=0, output_dict=True
                )
                cm     = confusion_matrix(y_true, y_pred, labels=["True", "False"])

                print(f"\n  Accuracy: {acc:.4f}")
                print(classification_report(y_true, y_pred, labels=["True", "False"], zero_division=0))
                print(f"  Confusion matrix [True, False]:\n{cm}")

                results_rows.append({
                    "backbone":        backbone,
                    "k":               k,
                    "domain":          domain,
                    "n":               len(df_dom),
                    "accuracy":        round(acc, 4),
                    "precision_true":  round(report["True"]["precision"], 4),
                    "recall_true":     round(report["True"]["recall"], 4),
                    "f1_true":         round(report["True"]["f1-score"], 4),
                    "macro_f1":        round(report["macro avg"]["f1-score"], 4),
                })

                all_preds_this_k.append(df_dom)

                del model
                torch.cuda.empty_cache()

           # ---- Aggregated metrics for this (backbone, k) ----
            if all_preds_this_k:
                df_all = pd.concat(all_preds_this_k, ignore_index=True)
                acc_all = accuracy_score(df_all["label"], df_all["pred"])
                rep_all = classification_report(
                    df_all["label"], df_all["pred"],
                    labels=["True", "False"], zero_division=0
                )
                cm_all = confusion_matrix(df_all["label"], df_all["pred"], labels=["True", "False"])

                print(f"\n{'='*60}")
                print(f"===== Gesamt-Performance über alle Domains =====")
                print(f"Backbone={backbone.upper()}  k={k}  |  total_samples={len(df_all)}")
                print(f"\nOverall accuracy: {acc_all:.3f}")
                print(f"\nOverall classification report:")
                print(rep_all)
                print(f"Overall confusion matrix (True, False):")
                print(cm_all)

                print(f"\nPer-domain accuracy summary:")
                for row in results_rows:
                    if row["backbone"] == backbone and row["k"] == k:
                        print(f"  {row['domain']:25s} n={row['n']:4d}  acc={row['accuracy']:.3f}")
                print(f"{'='*60}")
                summary_rows.append({
                    "backbone":       backbone,
                    "k":              k,
                    "n_eval":         len(df_all),
                    "accuracy":       round(acc_all, 4),
                    "macro_f1":       round(pd.DataFrame(classification_report(
                                          df_all["label"], df_all["pred"],
                                          labels=["True", "False"], zero_division=0, output_dict=True
                                      )).loc["f1-score", "macro avg"], 4),
                })

                # Save per-domain predictions for this (backbone, k)
                out_csv = f"Results/expert_eval_{backbone}_k{k}.csv"
                df_all.to_csv(out_csv, index=False)
                print(f"  💾 Saved: {out_csv}")

    # ---- Final summary tables ----
    df_results = pd.DataFrame(results_rows)
    df_summary = pd.DataFrame(summary_rows)

    # Backbone-level aggregation (mean across k)
    df_backbone = (
        df_summary.groupby("backbone")
        .agg(
            total_samples=("n_eval", "sum"),
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
        )
        .round(4)
        .reset_index()
    )

    print(f"\n\n{'='*70}")
    print("FULL SUMMARY — per (backbone, k)")
    print("="*70)
    print(df_summary.to_string(index=False))

    print(f"\n\nBACKBONE-LEVEL SUMMARY (mean across k=5,8,12)")
    print(df_backbone.to_string(index=False))

    df_results.to_csv("Results/expert_eval_per_domain.csv",   index=False)
    df_summary.to_csv("Results/expert_eval_summary_per_model.csv", index=False)
    df_backbone.to_csv("Results/expert_eval_summary_per_backbone.csv", index=False)
    print("\n💾 All summaries saved to Results/")


if __name__ == "__main__":
    main()