# ===== Env & Logging (muss ganz oben stehen) =====
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2")

import logging, json, math, time, torch
from typing import Literal, List, Dict, Optional, Callable
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import json as _json

from langchain_experimental.pydantic_v1 import BaseModel, Field
from langchain_experimental.llms import LMFormatEnforcer

logging.basicConfig(level=logging.ERROR)

# ========== 1) JSON-Schema ==========
class ClaimVerdict(BaseModel):
    verdict: Literal["True", "False"] = Field(..., description="Binary verdict")
    explanation: str = Field(..., description="2-4 sentences, brief, no stepwise reasoning")

JSON_SCHEMA = ClaimVerdict.schema()

# ========== 2) Helper: Modell-Cache & Prompt ==========
@dataclass
class Expert:
    name: str
    system: str
    model_id: Optional[str] = None
    max_new_tokens: int = 176

class ModelBundle:
    def __init__(self, model_id: str):
        self.model_id = model_id
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = AutoConfig.from_pretrained(model_id)
        if device.type == "cuda":
            config.pretraining_tp = 1

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            do_sample=False,
            return_full_text=False,
        )

        self.enforced_llm = LMFormatEnforcer(
            pipeline=self.pipeline,
            json_schema=JSON_SCHEMA,
        )

    def build_expert_prompt(self, system_prompt: str, statement: str, subject: str) -> str:
        schema_text = json.dumps(JSON_SCHEMA, ensure_ascii=False)
        subj = (subject or "").strip()
        subject_block = f"Subjects: {subj}" if subj else "Subjects: (none)"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":
                f"Use this JSON Schema:\n{schema_text}\n\n"
                f"Claim: {statement}\n{subject_block}\nJSON:"},
        ]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return (f"[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n"
                    f"Use this JSON Schema:\n{schema_text}\n\n"
                    f"Claim: {statement}\n{subject_block}\nJSON:")

def _normalize_labels(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().map({
        "True": "True", "False": "False",
        "true": "True", "false": "False",
        "TRUE": "True", "FALSE": "False",
        "T": "True", "F": "False", "1": "True", "0": "False"
    })

def compute_binary_metrics(df: pd.DataFrame, label_col: str, pred_col: str) -> Dict:
    y_true_all = _normalize_labels(df[label_col].dropna())
    y_pred_all = _normalize_labels(df[pred_col].dropna())
    idx = y_true_all.index.intersection(y_pred_all.index)
    y_true, y_pred = y_true_all.loc[idx], y_pred_all.loc[idx]
    if y_true.empty:
        return {"counts": {"n_total": int(df.shape[0]), "n_eval": 0},
                "note": f"No evaluable rows for label_col='{label_col}' and pred_col='{pred_col}'."}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_true": float(precision_score(y_true, y_pred, pos_label="True", zero_division=0)),
        "recall_true": float(recall_score(y_true, y_pred, pos_label="True", zero_division=0)),
        "f1_true": float(f1_score(y_true, y_pred, pos_label="True", zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=["True", "False"], zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["True", "False"]).tolist(),
        "counts": {"n_total": int(df.shape[0]), "n_eval": int(len(idx))},
        "columns": {"label_col": label_col, "pred_col": pred_col},
    }

def extract_first_json(text: str):
    dec = json.JSONDecoder()
    s = text.strip()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(s[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("No JSON object found")

_MODEL_CACHE: Dict[str, ModelBundle] = {}
def get_model_bundle(model_id: str) -> ModelBundle:
    if model_id not in _MODEL_CACHE:
        _MODEL_CACHE[model_id] = ModelBundle(model_id)
    return _MODEL_CACHE[model_id]

def run_with_retries(bundle: ModelBundle, prompt_text: str, base_max_new_tokens: int = 176, tries: int = 3, bump: int = 48):
    last_err = None
    for t in range(tries):
        try_tokens = base_max_new_tokens + t * bump
        try:
            raw = bundle.enforced_llm.invoke(prompt_text, max_new_tokens=try_tokens)
            return extract_first_json(raw)
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("Unknown generation error")

# ========== 3) Hauptfunktion (Mehrheitsvotum) ==========
def process_claims_multi_experts(
    csv_path: str,
    experts_cfg: List[Dict],
    nrows: Optional[int] = None,
    default_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    show_progress: bool = True,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    # --- neue Optionen für Majority ---
    majority_tiebreak: str = "best_expert",   # "best_expert" | "first_non_null" | "default_false"
    save_path: Optional[str] = None
) -> pd.DataFrame:
    # --- CSV einlesen ---
    df = pd.read_csv(
        csv_path,
        sep="\t",
        quotechar='"',
        engine="python",
        dtype=str,
        nrows=nrows,
    )

    # --- Expert:innen normalisieren ---
    experts: List[Expert] = []
    for ec in experts_cfg:
        if "name" not in ec or "system" not in ec:
            raise ValueError("Jeder Expert benötigt mindestens 'name' und 'system'.")
        experts.append(Expert(
            name=str(ec["name"]),
            system=str(ec["system"]),
            model_id=ec.get("model_id") or default_model_id,
            max_new_tokens=int(ec.get("max_new_tokens", 176)),
        ))

    # --- Modelle vorbereiten ---
    for ex in experts:
        get_model_bundle(ex.model_id)

    # --- Fortschritt ---
    total_tasks = len(df)
    iterator = tqdm(
        total=total_tasks,
        desc="🔎 Fact-checking (majority vote)",
        unit="statement",
        dynamic_ncols=True,
        mininterval=0.25,
        smoothing=0.1,
        leave=True,
    ) if show_progress else None

    rows_out = []
    done = 0
    t0 = time.perf_counter()

    # --- Hauptlauf: Expert:innen generieren ---
    expert_colnames: List[str] = []
    for _, row in df.iterrows():
        out_row = {
            "label_true": row.get("label") or row.get("label_true"),
            "statement": row.get("statement"),
            "subject": row.get("subject") or row.get("subjects"),
        }

        for ex in experts:
            bundle = get_model_bundle(ex.model_id)
            subject_val = out_row["subject"] or ""
            prompt_text = bundle.build_expert_prompt(
                system_prompt=ex.system,
                statement=out_row["statement"],
                subject=subject_val,
            )

            verdict, explanation = None, None
            try:
                parsed = run_with_retries(
                    bundle,
                    prompt_text,
                    base_max_new_tokens=ex.max_new_tokens,
                    tries=3,
                    bump=48,
                )
                verdict = parsed.get("verdict")
                explanation = parsed.get("explanation")
            except Exception as e:
                explanation = f"Error: {e}"

            col_v = f"verdict_pred__{ex.name}"
            col_e = f"explanation__{ex.name}"
            out_row[col_v] = verdict
            out_row[col_e] = explanation
            if col_v not in expert_colnames:
                expert_colnames.append(col_v)

        rows_out.append(out_row)

        done += 1
        if iterator:
            iterator.update(1)
        if progress_callback:
            elapsed = time.perf_counter() - t0
            eta = (elapsed / max(done, 1)) * (total_tasks - done)
            progress_callback(done, total_tasks, eta)

    if iterator:
        iterator.close()

    result_df = pd.DataFrame(rows_out)

    # --- Mehrheitsvotum berechnen ---
    def row_majority(series: pd.Series, expert_cols: List[str], best_col: Optional[str]) -> Dict[str, str]:
        votes = []
        for c in expert_cols:
            v = series.get(c)
            if isinstance(v, str):
                vv = v.strip()
                if vv in {"True", "False", "true", "false", "TRUE", "FALSE", "T", "F", "1", "0"}:
                    # robust normalisieren
                    vv = _normalize_labels(pd.Series([vv])).iloc[0]
                    votes.append((c, vv))
        if not votes:
            # fallback
            if majority_tiebreak == "best_expert" and best_col and isinstance(series.get(best_col), str):
                vv = _normalize_labels(pd.Series([series.get(best_col)])).iloc[0]
                return {"verdict": vv, "rule": f"fallback_best_expert:{best_col}", "n_true": 0, "n_false": 0}
            if majority_tiebreak == "first_non_null" and votes:
                return {"verdict": votes[0][1], "rule": f"fallback_first:{votes[0][0]}", "n_true": 0, "n_false": 0}
            return {"verdict": "False", "rule": "fallback_default_false", "n_true": 0, "n_false": 0}

        n_true = sum(1 for _, v in votes if v == "True")
        n_false = sum(1 for _, v in votes if v == "False")

        if n_true > n_false:
            return {"verdict": "True", "rule": "majority", "n_true": n_true, "n_false": n_false}
        if n_false > n_true:
            return {"verdict": "False", "rule": "majority", "n_true": n_true, "n_false": n_false}

        # Tie
        if majority_tiebreak == "best_expert" and best_col and isinstance(series.get(best_col), str):
            vv = _normalize_labels(pd.Series([series.get(best_col)])).iloc[0]
            return {"verdict": vv, "rule": f"tiebreak_best_expert:{best_col}", "n_true": n_true, "n_false": n_false}
        if majority_tiebreak == "first_non_null":
            c0, v0 = votes[0]
            return {"verdict": v0, "rule": f"tiebreak_first:{c0}", "n_true": n_true, "n_false": n_false}
        return {"verdict": "False", "rule": "tiebreak_default_false", "n_true": n_true, "n_false": n_false}

    # --- Best-Expert zur Tie-Breaker-Nutzung bestimmen (falls Labels existieren) ---
    best_expert_col: Optional[str] = None
    if "label_true" in result_df.columns and result_df["label_true"].notna().any() and majority_tiebreak == "best_expert":
        label_norm = _normalize_labels(result_df["label_true"].dropna())
        best_acc, best_col = -1.0, None
        for c in expert_colnames:
            pred_norm = _normalize_labels(result_df[c].dropna())
            idx = label_norm.index.intersection(pred_norm.index)
            if len(idx) == 0:
                continue
            acc = accuracy_score(label_norm.loc[idx], pred_norm.loc[idx])
            if acc > best_acc:
                best_acc, best_col = acc, c
        best_expert_col = best_col

    maj = result_df.apply(lambda r: row_majority(r, expert_colnames, best_expert_col), axis=1, result_type="expand")
    result_df["majority_true_votes"] = maj["n_true"]
    result_df["majority_false_votes"] = maj["n_false"]
    result_df["majority_decision_rule"] = maj["rule"]
    result_df["verdict_final"] = maj["verdict"]
    result_df["explanation_final"] = (
        "Panel majority: True=" + result_df["majority_true_votes"].astype(str) +
        ", False=" + result_df["majority_false_votes"].astype(str) +
        " | rule=" + result_df["majority_decision_rule"]
    )

    # --- optional speichern ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result_df.to_csv(save_path, index=False, encoding="utf-8")
        print(f"✅ Ergebnisse gespeichert unter: {save_path}")

    # --- Evaluation (auf 'verdict_final') ---
    try:
        if "label_true" in result_df.columns and "verdict_final" in result_df.columns:
            y_true = _normalize_labels(result_df["label_true"].dropna())
            y_pred = _normalize_labels(result_df["verdict_final"].dropna())
            idx = y_true.index.intersection(y_pred.index)
            y_true, y_pred = y_true.loc[idx], y_pred.loc[idx]
            if not y_true.empty:
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, pos_label="True", zero_division=0)
                rec = recall_score(y_true, y_pred, pos_label="True", zero_division=0)
                f1 = f1_score(y_true, y_pred, pos_label="True", zero_division=0)
                cm = confusion_matrix(y_true, y_pred, labels=["True", "False"]).tolist()
                report = classification_report(y_true, y_pred, labels=["True", "False"], zero_division=0)
                print("\n📊 Evaluation (Majority Verdict)")
                print(f"accuracy: {acc:.3f}")
                print(f"precision_true: {prec:.3f}")
                print(f"recall_true: {rec:.3f}")
                print(f"f1_true: {f1:.3f}")
                print(f"confusion_matrix:")
                for row in cm:
                    print(row)
                print("Classification report:\n" + report)
            else:
                print("⚠️ Keine auswertbaren Labels/Vorhersagen gefunden.")
        else:
            print("⚠️ Evaluation übersprungen (fehlende Spalten 'label_true' oder 'verdict_final').")
    except Exception as e:
        print(f"⚠️ Evaluation fehlgeschlagen: {e}")

    return result_df
