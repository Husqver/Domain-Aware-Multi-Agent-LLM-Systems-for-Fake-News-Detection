import os
import re
import glob
import pandas as pd
from typing import Any, Dict, List, Optional

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def norm_bool_label(x: Any) -> str:
    if not isinstance(x, str):
        return "False"
    s = x.strip().strip(".,:;!?)(").lower()
    if s.startswith("true"):
        return "True"
    if s.startswith("false"):
        return "False"
    return "False"


def safe_read_csv(path: str) -> pd.DataFrame:
    # auto-detect separator
    return pd.read_csv(
        path,
        sep=None,
        engine="python",
        dtype=str,
        keep_default_na=False,
    )


def infer_k_from_filename(name: str) -> Optional[int]:
    # *_5.csv, *_8.csv, *_12.csv
    m = re.search(r"(?:_|-)(5|8|12)(?:\.csv$)", name)
    if m:
        return int(m.group(1))
    # *5Classes*, *8Classes*, *12Classes*
    m = re.search(r"(5|8|12)\s*classes", name, flags=re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"(5|8|12)Classes", name, flags=re.I)
    if m:
        return int(m.group(1))
    return None


def infer_model_from_filename(name: str) -> str:
    n = name.lower()
    if "llama" in n:
        return "llama"
    if "gemma" in n:
        return "gemma"
    if "deepseek" in n:
        return "deepseek"
    return "unknown"


def print_header(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def evaluate_predictions_df_verbose(
    df: pd.DataFrame,
    super_labels: Optional[List[str]],
    text_col: str = "statement",
    label_col: str = "label",
    domain_col: str = "super_domain",
    domain_pred_col: str = "domain_pred",
    domain_router_col: str = "domain_pred_router",
    verdict_pred_col: str = "verdict_pred",
) -> Dict[str, Any]:
    # ---- Basic filtering (matches your pipeline)
    df = df.copy()
    df = df.dropna(subset=[text_col, label_col])
    df = df[df[text_col].astype(str).str.strip() != ""]
    df = df[df[label_col].isin(["True", "False"])]

    print("Test set size:", len(df))

    if domain_col in df.columns:
        print("True domain distribution:")
        print(df[domain_col].value_counts())

    if domain_router_col in df.columns:
        print("\nPredicted domain distribution (router):")
        print(df[domain_router_col].value_counts())

    # If domain_pred missing, fallback to router
    if domain_pred_col not in df.columns and domain_router_col in df.columns:
        df[domain_pred_col] = df[domain_router_col]

    # --------------------------
    # Domain routing metrics
    # --------------------------
    domain_acc = None
    cm_domain = None
    n_domain_eval = 0
    domain_labels_used = None

    if domain_col in df.columns and domain_pred_col in df.columns:
        domain_labels_used = super_labels if super_labels is not None else sorted(
            set(df[domain_col].dropna().unique()).union(set(df[domain_pred_col].dropna().unique()))
        )

        mask_valid = df[domain_col].isin(domain_labels_used) & df[domain_pred_col].isin(domain_labels_used)
        n_domain_eval = int(mask_valid.sum())

        if n_domain_eval > 0:
            y_true_dom = df.loc[mask_valid, domain_col]
            y_pred_dom = df.loc[mask_valid, domain_pred_col]

            domain_acc = float(accuracy_score(y_true_dom, y_pred_dom))
            cm_domain = confusion_matrix(y_true_dom, y_pred_dom, labels=domain_labels_used)

            print(f"\n🌍 Domain routing accuracy: {domain_acc:.3f}")
            print("Confusion matrix (rows=true, cols=pred):")
            print(cm_domain)
            print("Label order:", domain_labels_used)
        else:
            print("\n⚠️ Domain routing accuracy skipped (no valid domain labels).")

    # --------------------------
    # Verdict metrics
    # --------------------------
    verdict_acc = None
    cm_verdict = None
    verdict_rep = None
    n_eval = len(df)

    precision_true = None
    recall_true = None
    f1_true = None

    if verdict_pred_col in df.columns:
        y_true = df[label_col].map(norm_bool_label)
        y_pred = df[verdict_pred_col].map(norm_bool_label)

        verdict_acc = float(accuracy_score(y_true, y_pred))
        cm_verdict = confusion_matrix(y_true, y_pred, labels=["True", "False"])
        verdict_rep = classification_report(y_true, y_pred, labels=["True", "False"], zero_division=0)

        # metrics for class "True"
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=["True"], average=None, zero_division=0
        )
        precision_true = float(p[0])
        recall_true = float(r[0])
        f1_true = float(f1[0])

        print(f"\n✅ Verdict accuracy (overall): {verdict_acc:.3f}\n")
        print("Classification report (True/False):")
        print(verdict_rep)
        print("Confusion matrix [rows=true, cols=pred] (True, False):")
        print(cm_verdict)
    else:
        print("\n⚠️ Verdict accuracy skipped (missing verdict_pred column).")

    # --------------------------
    # Per-domain verdict accuracy
    # --------------------------
    per_domain = {}
    if domain_col in df.columns and verdict_pred_col in df.columns:
        labels_for_per_domain = super_labels if super_labels is not None else sorted(df[domain_col].dropna().unique())
        print("\n===== Per-domain verdict accuracy (by true domain) =====")
        for dom in labels_for_per_domain:
            df_dom = df[df[domain_col] == dom]
            if df_dom.empty:
                continue
            yt = df_dom[label_col].map(norm_bool_label)
            yp = df_dom[verdict_pred_col].map(norm_bool_label)
            acc_dom = float(accuracy_score(yt, yp))
            per_domain[dom] = {"n": int(len(df_dom)), "acc": acc_dom}
            print(f"{dom:22s} n={len(df_dom):4d} acc={acc_dom:.3f}")

    return {
        "n_eval": int(n_eval),
        "domain_acc": domain_acc,
        "n_domain_eval": int(n_domain_eval),
        "domain_confusion_matrix": cm_domain,
        "domain_labels_used": domain_labels_used,
        "verdict_acc": verdict_acc,
        "precision_true": precision_true,
        "recall_true": recall_true,
        "f1_true": f1_true,
        "verdict_confusion_matrix": cm_verdict,
        "verdict_report": verdict_rep,
        "per_domain": per_domain,
    }


def evaluate_all_verbose(
    folder: str = "Results",
    pattern: str = "*test_predictions_with_experts*.csv",
    super_labels_by_k: Optional[Dict[int, List[str]]] = None,
    only_model: Optional[str] = None,   # e.g. "llama"
    only_k: Optional[int] = None,       # e.g. 8
    save_summary_csv: Optional[str] = "Results/_summary_by_model_and_k.csv",
) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    paths = [
        p for p in paths
        if re.search(r"test_predictions_with_experts_\d+\.csv$", os.path.basename(p))
    ]
    if not paths:
        raise FileNotFoundError(f"No files found in '{folder}' with pattern '{pattern}'")

    rows = []

    for p in paths:
        fname = os.path.basename(p)
        model = infer_model_from_filename(fname)
        k = infer_k_from_filename(fname)

        if only_model is not None and model != only_model.lower():
            continue
        if only_k is not None and k != only_k:
            continue

        df = safe_read_csv(p)

        super_labels = None
        if super_labels_by_k is not None and k in super_labels_by_k:
            super_labels = super_labels_by_k[k]

        print_header(f"{fname}   |   model={model}   |   k={k}")
        res = evaluate_predictions_df_verbose(df, super_labels=super_labels)

        rows.append({
            # deine bisherigen Spalten
            "file": fname,
            "model": model,
            "k": k,

            # Neu: verdict task summary wie in deinem Beispiel
            "mode": f"{model}_k{k}",
            "n_eval": res["n_eval"],
            "accuracy": res["verdict_acc"],
            "precision_true": res["precision_true"],
            "recall_true": res["recall_true"],
            "f1_true": res["f1_true"],
            "confusion_matrix": str(res["verdict_confusion_matrix"]),

            # Optional: Domain routing ebenfalls mit Confusion Matrix
            "domain_acc": res["domain_acc"],
            "n_domain_eval": res["n_domain_eval"],
            "domain_confusion_matrix": (
                None if res["domain_confusion_matrix"] is None
                else str(res["domain_confusion_matrix"])
            ),
        })

    summary = pd.DataFrame(rows)
    summary["k"] = pd.to_numeric(summary["k"], errors="coerce")
    summary = summary.sort_values(["model", "k", "file"], ascending=[True, True, True])

    print_header("SUMMARY (all evaluated files)")
    with pd.option_context("display.max_rows", 300, "display.max_colwidth", 200):
        print(summary.to_string(index=False))

    if save_summary_csv:
        out_dir = os.path.dirname(save_summary_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        summary.to_csv(save_summary_csv, index=False, encoding="utf-8")
        print(f"\n💾 Saved summary to: {save_summary_csv}")

    return summary

