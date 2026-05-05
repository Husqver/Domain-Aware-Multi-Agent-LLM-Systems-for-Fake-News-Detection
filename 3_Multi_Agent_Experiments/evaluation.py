import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# -------------------------
# Config / Helpers
# -------------------------
LABEL_ALIASES = ["label_true", "label"]
STATEMENT_ALIASES = ["statement"]
SUBJECT_ALIASES = ["subjects", "subject"]

# Default expert names 
EXPERT_NAMES_DEFAULT = ["politics", "economy", "health_science"]


def _norm_bool_str(x: Any) -> Optional[str]:
    """Normalize model outputs/labels to 'True'/'False' or None."""
    if x is None:
        return None
    s = str(x).strip().strip('"').strip("'")
    if s == "":
        return None
    s_low = s.lower()
    if s_low in ("true", "t", "1", "yes"):
        return "True"
    if s_low in ("false", "f", "0", "no"):
        return "False"
    return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    """Robust float conversion: handles '', 'nan', None, non-finite."""
    if x is None:
        return default
    s = str(x).strip().strip('"').strip("'")
    if s == "" or s.lower() in ("nan", "none", "null"):
        return default
    try:
        v = float(s)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _find_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _pick_label_col(df: pd.DataFrame) -> str:
    col = _find_first_existing_col(df, LABEL_ALIASES)
    if not col:
        raise ValueError(f"Kein Label-Column gefunden. Vorhanden: {list(df.columns)}")
    return col


def _pick_statement_col(df: pd.DataFrame) -> Optional[str]:
    return _find_first_existing_col(df, STATEMENT_ALIASES)


def _pick_subject_col(df: pd.DataFrame) -> Optional[str]:
    return _find_first_existing_col(df, SUBJECT_ALIASES)


def _pick_decision_col(df: pd.DataFrame) -> Optional[str]:
    if "verdict_final" in df.columns:
        return "verdict_final"
    if "verdict_pred__final_decision" in df.columns:
        return "verdict_pred__final_decision"
    verdict_cols = [c for c in df.columns if c.startswith("verdict_pred__")]
    if verdict_cols:
        return verdict_cols[0]
    return None


def _expert_verdict_cols(df: pd.DataFrame, expert_names: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Returns mapping expert_name -> verdict_column
    Expects columns like verdict_pred__politics etc.
    """
    expert_names = expert_names or EXPERT_NAMES_DEFAULT
    mapping = {}
    for name in expert_names:
        col = f"verdict_pred__{name}"
        if col in df.columns:
            mapping[name] = col

    if not mapping:
        cols = [c for c in df.columns if c.startswith("verdict_pred__")]
        cols = [c for c in cols if "final" not in c.lower() and "decision" not in c.lower()]
        for c in cols:
            name = c.split("__", 1)[-1]
            mapping[name] = c

    return mapping


def _compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "n_eval": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_true": float(precision_score(y_true, y_pred, pos_label="True", zero_division=0)),
        "recall_true": float(recall_score(y_true, y_pred, pos_label="True", zero_division=0)),
        "f1_true": float(f1_score(y_true, y_pred, pos_label="True", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["True", "False"]).tolist(),
        "report": classification_report(y_true, y_pred, labels=["True", "False"], zero_division=0),
    }


def _majority_vote_row(expert_votes: List[Optional[str]]) -> str:
    votes = [v for v in expert_votes if v in ("True", "False")]
    if not votes:
        return "False"
    t = sum(v == "True" for v in votes)
    f = sum(v == "False" for v in votes)
    if t > f:
        return "True"
    return "False"  # tie -> False


def _weighted_vote_row(
    expert_votes_by_name: Dict[str, Optional[str]],
    weights_by_name: Dict[str, float],
) -> str:
    items = []
    for name, v in expert_votes_by_name.items():
        if v in ("True", "False") and name in weights_by_name:
            items.append((name, v, float(weights_by_name[name])))

    if not items:
        return "False"

    w_sum = sum(w for _, _, w in items)
    if w_sum <= 1e-12:
        return _majority_vote_row([v for _, v, _ in items])

    true_mass = sum(w for _, v, w in items if v == "True") / w_sum
    return "True" if true_mass > 0.5 else "False"


# -------------------------
# Router weights pairing
# -------------------------
def _infer_backbone_and_setting(filename: str) -> Tuple[Optional[str], Optional[str]]:
    fn = filename.lower()

    backbone = None
    for b in ["gemma", "llama", "deepseek"]:
        if fn.startswith(b + "_") or fn.startswith("router_weights_" + b):
            backbone = b
            break

    setting = None
    if "withsubjects" in fn or "with_subjects" in fn:
        setting = "withsubjects"
    if "withoutsubjects" in fn:
        setting = "withoutsubjects"
    if "no_subjects" in fn:
        setting = "withoutsubjects"
    if "with_subjects" in fn:
        setting = "withsubjects"

    return backbone, setting


def _find_router_weights_file(results_dir: str, result_filename: str) -> Optional[str]:
    backbone, setting = _infer_backbone_and_setting(result_filename)
    if not backbone or not setting:
        return None

    target1 = f"router_weights_{backbone}_{setting}.csv"
    path1 = os.path.join(results_dir, target1)
    if os.path.exists(path1):
        return path1

    for f in os.listdir(results_dir):
        lf = f.lower()
        if lf.startswith("router_weights_") and backbone in lf and setting in lf and lf.endswith(".csv"):
            return os.path.join(results_dir, f)

    return None


# -------------------------
# Main evaluation per file
# -------------------------
@dataclass
class EvalResult:
    filename: str
    modes: Dict[str, Dict[str, Any]]  # mode -> metrics
    notes: List[str]


def evaluate_one_file(
    file_path: str,
    results_dir: str,
    expert_names: Optional[List[str]] = None,
    enable_weighted: bool = True,
) -> EvalResult:
    df = pd.read_csv(file_path, dtype=str)
    notes: List[str] = []

    # --- Labels ---
    label_col = _pick_label_col(df)
    df["_y_true"] = df[label_col].apply(_norm_bool_str)
    df = df[df["_y_true"].isin(["True", "False"])].copy()
    if df.empty:
        return EvalResult(os.path.basename(file_path), modes={}, notes=[f"Keine auswertbaren Labels in {label_col}."])

    # --- Decision column ---
    decision_col = _pick_decision_col(df)
    if not decision_col:
        notes.append("Kein Decision-Verdict gefunden (verdict_final / verdict_pred__...).")

    # --- Experts ---
    expert_map = _expert_verdict_cols(df, expert_names=expert_names)
    if not expert_map:
        notes.append("Keine Expert-Verdict-Spalten gefunden (verdict_pred__*).")

    modes: Dict[str, Dict[str, Any]] = {}

    # ---- Decision metrics ----
    if decision_col and decision_col in df.columns:
        df["_y_decision"] = df[decision_col].apply(_norm_bool_str)
        mask = df["_y_decision"].isin(["True", "False"])
        y_true = df.loc[mask, "_y_true"].tolist()
        y_pred = df.loc[mask, "_y_decision"].tolist()
        if y_true:
            modes["decision_agent"] = _compute_metrics(y_true, y_pred)
        else:
            notes.append(f"Decision column {decision_col} hat keine validen True/False Werte.")

    # ---- Majority vote + per-expert ----
    if expert_map:
        for name, col in expert_map.items():
            df[f"_ex_{name}"] = df[col].apply(_norm_bool_str)

        expert_cols_norm = [f"_ex_{n}" for n in expert_map.keys()]

        df["_y_majority"] = df[expert_cols_norm].apply(
            lambda r: _majority_vote_row([r[c] for c in expert_cols_norm]),
            axis=1,
        )
        modes["majority_vote"] = _compute_metrics(df["_y_true"].tolist(), df["_y_majority"].tolist())

        for name in expert_map.keys():
            mask = df[f"_ex_{name}"].isin(["True", "False"])
            if mask.any():
                modes[f"expert__{name}"] = _compute_metrics(
                    df.loc[mask, "_y_true"].tolist(),
                    df.loc[mask, f"_ex_{name}"].tolist(),
                )

    # ---- Weighted vote (experts + router weights) ----
    if enable_weighted and expert_map:
        router_path = _find_router_weights_file(results_dir, os.path.basename(file_path))
        if not router_path:
            notes.append("Keine passende Router-Weights Datei gefunden -> weighted_vote übersprungen.")
        else:
            wdf = pd.read_csv(router_path, dtype=str)

            # weight column detection
            col_candidates = {
                "politics": ["w_politics", "politics", "weight_politics", "w__politics"],
                "economy": ["w_economy", "economy", "weight_economy", "w__economy"],
                "health_science": ["w_health_science", "health_science", "weight_health_science", "w__health_science"],
            }
            wcols: Dict[str, str] = {}
            for k, cands in col_candidates.items():
                c = _find_first_existing_col(wdf, cands)
                if c:
                    wcols[k] = c

            if len(wcols) < 3:
                notes.append(f"Router file {os.path.basename(router_path)} hat nicht alle Weight-Spalten -> skip.")
            else:
                # numeric + NaN->0
                for k, col in wcols.items():
                    wdf[col] = pd.to_numeric(wdf[col], errors="coerce").fillna(0.0)

                # Merge by statement (+ subject if available); fallback to index alignment
                stmt_col_res = _pick_statement_col(df)
                stmt_col_w = _pick_statement_col(wdf)
                subj_col_res = _pick_subject_col(df)
                subj_col_w = _pick_subject_col(wdf)

                # Trim keys on both sides (avoids whitespace/quote mismatch)
                if stmt_col_res:
                    df[stmt_col_res] = df[stmt_col_res].astype(str).str.strip()
                if subj_col_res:
                    df[subj_col_res] = df[subj_col_res].astype(str).str.strip()
                if stmt_col_w:
                    wdf[stmt_col_w] = wdf[stmt_col_w].astype(str).str.strip()
                if subj_col_w:
                    wdf[subj_col_w] = wdf[subj_col_w].astype(str).str.strip()

                if stmt_col_res and stmt_col_w:
                    left_on = [stmt_col_res]
                    right_on = [stmt_col_w]
                    if subj_col_res and subj_col_w:
                        left_on.append(subj_col_res)
                        right_on.append(subj_col_w)

                    merged = df.merge(
                        wdf,
                        how="left",
                        left_on=left_on,
                        right_on=right_on,
                        suffixes=("", "__w"),
                    )
                else:
                    # fallback: index alignment
                    df2 = df.copy()
                    wdf2 = wdf.copy()
                    df2["_idx"] = np.arange(len(df2))
                    wdf2["_idx"] = np.arange(len(wdf2))
                    merged = df2.merge(wdf2, on="_idx", how="left", suffixes=("", "__w"))

                # Ensure normalized expert votes exist on merged
                for name in expert_map.keys():
                    col_norm = f"_ex_{name}"
                    if col_norm not in merged.columns:
                        merged[col_norm] = merged[expert_map[name]].apply(_norm_bool_str)

                def row_weights(r) -> Dict[str, float]:
                    out: Dict[str, float] = {}
                    for name in expert_map.keys():
                        if name == "politics":
                            out[name] = _safe_float(r.get(wcols["politics"], 0.0))
                        elif name == "economy":
                            out[name] = _safe_float(r.get(wcols["economy"], 0.0))
                        elif name == "health_science":
                            out[name] = _safe_float(r.get(wcols["health_science"], 0.0))
                        else:
                            out[name] = 0.0

                    s = sum(out.values())
                    if s <= 1e-12:
                        n = max(1, len(out))
                        return {k: 1.0 / n for k in out.keys()}
                    return {k: v / s for k, v in out.items()}

                merged["_y_weighted"] = merged.apply(
                    lambda r: _weighted_vote_row(
                        expert_votes_by_name={n: r[f"_ex_{n}"] for n in expert_map.keys()},
                        weights_by_name=row_weights(r),
                    ),
                    axis=1,
                )

                modes["weighted_vote"] = _compute_metrics(
                    merged["_y_true"].tolist(),
                    merged["_y_weighted"].tolist(),
                )

                # Optional note if many weights missing after merge
                w_missing = merged[wcols["politics"]].isna().sum() if wcols["politics"] in merged.columns else 0
                if w_missing > 0:
                    notes.append(f"Hinweis: {w_missing} Zeilen ohne gemergte Weights (check merge keys).")

    return EvalResult(os.path.basename(file_path), modes=modes, notes=notes)


def evaluate_results_dir(
    results_dir: str,
    output_summary_csv: Optional[str] = None,
    output_details_json: Optional[str] = None,
    expert_names: Optional[List[str]] = None,
    enable_weighted: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    files = sorted([f for f in os.listdir(results_dir) if f.lower().endswith(".csv")])

    result_files = [f for f in files if not f.lower().startswith("router_weights_")]

    all_rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}

    for f in result_files:
        path = os.path.join(results_dir, f)
        try:
            res = evaluate_one_file(
                file_path=path,
                results_dir=results_dir,
                expert_names=expert_names,
                enable_weighted=enable_weighted,
            )
        except Exception as e:
            details[f] = {"error": str(e)}
            all_rows.append({
                "file": f,
                "mode": "_error",
                "n_eval": None,
                "accuracy": None,
                "precision_true": None,
                "recall_true": None,
                "f1_true": None,
                "confusion_matrix": json.dumps(str(e)),
            })
            continue

        details[f] = {"notes": res.notes, "modes": res.modes}

        for mode, met in res.modes.items():
            all_rows.append({
                "file": f,
                "mode": mode,
                "n_eval": met.get("n_eval"),
                "accuracy": met.get("accuracy"),
                "precision_true": met.get("precision_true"),
                "recall_true": met.get("recall_true"),
                "f1_true": met.get("f1_true"),
                "confusion_matrix": json.dumps(met.get("confusion_matrix")),
            })

        if res.notes:
            all_rows.append({
                "file": f,
                "mode": "_notes",
                "n_eval": None,
                "accuracy": None,
                "precision_true": None,
                "recall_true": None,
                "f1_true": None,
                "confusion_matrix": json.dumps(res.notes, ensure_ascii=False),
            })

    summary_df = pd.DataFrame(all_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["file", "mode"]).reset_index(drop=True)

    if output_summary_csv:
        os.makedirs(os.path.dirname(output_summary_csv) or ".", exist_ok=True)
        summary_df.to_csv(output_summary_csv, index=False, encoding="utf-8")
        print(f"✅ Summary saved: {output_summary_csv}")

    if output_details_json:
        os.makedirs(os.path.dirname(output_details_json) or ".", exist_ok=True)
        with open(output_details_json, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
        print(f"✅ Details saved: {output_details_json}")

    return summary_df, details
