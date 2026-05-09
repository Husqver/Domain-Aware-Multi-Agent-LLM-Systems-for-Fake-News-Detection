# ===== Env & Logging (muss ganz oben stehen) =====
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")           # TF-Backend vermeiden
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3,4,7")       # wähle deine GPU(s)

import logging, json, math, time, torch
from typing import Literal, List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import json as _json  # falls oben schon 'json' importiert ist

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
    model_id: Optional[str] = None         # None => default_model_id
    base_model_id: Optional[str] = None    # Nur nötig, wenn model_id ein PEFT-Adapter ist
    max_new_tokens: int = 176

def _is_peft_adapter(path_or_repo: str) -> bool:
    """
    Heuristik für lokale Ordner: adapter_config.json vorhanden?
    Für Hub-IDs können wir es erst beim Laden sicher erkennen; hier genügt die lokale Erkennung.
    """
    try:
        return os.path.isdir(path_or_repo) and os.path.isfile(os.path.join(path_or_repo, "adapter_config.json"))
    except Exception:
        return False

class ModelBundle:
    """
    Lädt entweder ein vollständiges HF-Modell (mit config.json) ODER
    einen PEFT/LoRA-Adapter + Basis-Modell.
    """
    def __init__(self, model_id: str, base_model_id: Optional[str] = None, trust_remote_code: bool = False):
        self.model_id = model_id
        self.base_model_id = base_model_id
        self.trust_remote_code = trust_remote_code

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        device_map = "auto" if device.type == "cuda" else None

        is_adapter_local = _is_peft_adapter(model_id)

        if is_adapter_local and not base_model_id:
            raise ValueError(
                f"'{model_id}' sieht nach einem PEFT-Adapter aus (adapter_config.json gefunden), "
                "aber 'base_model_id' wurde nicht angegeben."
            )

        # Versuche: Vollmodell laden. Falls das fehlschlägt und base_model_id vorhanden ist,
        # interpretieren wir es als Adapter und laden Basis + Adapter.
        try:
            if not is_adapter_local and base_model_id is None:
                # Vollmodellpfad/Hub-ID
                config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
                if device.type == "cuda":
                    # manche Llama-Gewichte brauchen das, sonst warnen sie
                    try:
                        config.pretraining_tp = 1
                    except Exception:
                        pass

                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    torch_dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                ).eval()
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            else:
                # PEFT: Basis + Adapter
                from peft import PeftModel
                base_cfg = AutoConfig.from_pretrained(base_model_id, trust_remote_code=trust_remote_code)
                if device.type == "cuda":
                    try:
                        base_cfg.pretraining_tp = 1
                    except Exception:
                        pass

                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    config=base_cfg,
                    torch_dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                )
                model = PeftModel.from_pretrained(base_model, model_id)
                # Optional: Merge, wenn du reine Inferenz willst (spart etwas Overhead)
                try:
                    model = model.merge_and_unload()
                except Exception:
                    pass
                model = model.eval()
                tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

        except ValueError as e:
            # Klassischer Fehler: "Unrecognized model ... Should have a `model_type` key"
            if base_model_id:
                # Letzter Versuch als Adapter
                from peft import PeftModel
                base_cfg = AutoConfig.from_pretrained(base_model_id, trust_remote_code=trust_remote_code)
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    config=base_cfg,
                    torch_dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=trust_remote_code,
                )
                model = PeftModel.from_pretrained(base_model, model_id)
                try:
                    model = model.merge_and_unload()
                except Exception:
                    pass
                model = model.eval()
                tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
            else:
                raise

        self.model = model
        self.tokenizer = tokenizer
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

# ---- Label-Normalisierung (robust)
def _normalize_labels(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().map({
        "True": "True", "False": "False",
        "true": "True", "false": "False",
        "TRUE": "True", "FALSE": "False",
        "T": "True", "F": "False",
        "1": "True", "0": "False"
    })

def compute_binary_metrics(df: pd.DataFrame, label_col: str, pred_col: str) -> Dict:
    y_true_all = _normalize_labels(df[label_col].dropna())
    y_pred_all = _normalize_labels(df[pred_col].dropna())
    idx = y_true_all.index.intersection(y_pred_all.index)
    y_true = y_true_all.loc[idx]
    y_pred = y_pred_all.loc[idx]

    if y_true.empty:
        return {
            "counts": {"n_total": int(df.shape[0]), "n_eval": 0},
            "note": f"No evaluable rows for label_col='{label_col}' and pred_col='{pred_col}'.",
        }

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_true": float(precision_score(y_true, y_pred, pos_label="True", zero_division=0)),
        "recall_true": float(recall_score(y_true, y_pred, pos_label="True", zero_division=0)),
        "f1_true": float(f1_score(y_true, y_pred, pos_label="True", zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=["True", "False"], zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["True", "False"]).tolist(),
        "counts": {"n_total": int(df.shape[0]), "n_eval": int(len(idx))},
        "columns": {"label_col": label_col, "pred_col": pred_col},
    }
    return metrics

# ---- Robust JSON extractor (kein Regex nötig)
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

# ===== Modell-Cache =====
# Cache-Key muss base_model_id berücksichtigen (Adapter-Fall!)
_MODEL_CACHE: Dict[Tuple[str, Optional[str]], ModelBundle] = {}

def get_model_bundle(model_id: str, base_model_id: Optional[str] = None) -> ModelBundle:
    key = (model_id, base_model_id)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = ModelBundle(model_id, base_model_id=base_model_id)
    return _MODEL_CACHE[key]

# ---- Retry-Wrapper mit dynamischem max_new_tokens
def run_with_retries(
    bundle: ModelBundle,
    prompt_text: str,
    base_max_new_tokens: int = 176,
    tries: int = 3,
    bump: int = 48,
):
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

# ========== 3) Hauptfunktion mit Majority Vote (kein Decision-Modell mehr) ==========
def process_claims_multi_experts(
    csv_path: str,
    experts_cfg: List[Dict],
    nrows: Optional[int] = None,
    default_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    show_progress: bool = True,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
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
            base_model_id=ec.get("base_model_id"),  # nur setzen, wenn Adapter
            max_new_tokens=int(ec.get("max_new_tokens", 176)),
        ))

    # --- Modelle vorbereiten ---
    for ex in experts:
        get_model_bundle(ex.model_id, ex.base_model_id)

    # --- Fortschritt: 1 Tick pro Statement ---
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

    expert_colnames: List[str] = []

    for _, row in df.iterrows():
        out_row = {
            "label_true": row.get("label") or row.get("label_true"),
            "statement": row.get("statement"),
            "subject": row.get("subject") or row.get("subjects"),
        }

        # --- Experten laufen lassen ---
        for ex in experts:
            bundle = get_model_bundle(ex.model_id, ex.base_model_id)
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

        # --- Fortschritt jetzt EINMAL pro Statement updaten ---
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

    # --- Mehrheitsvotum-Helfer ---
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

        # Keine gültigen Votes
        if not votes:
            if majority_tiebreak == "best_expert" and best_col and isinstance(series.get(best_col), str):
                vv = _normalize_labels(pd.Series([series.get(best_col)])).iloc[0]
                return {"verdict": vv, "rule": f"fallback_best_expert:{best_col}", "n_true": 0, "n_false": 0}
            # "first_non_null" macht ohne Votes keinen Sinn → default_false
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
    if (
        "label_true" in result_df.columns
        and result_df["label_true"].notna().any()
        and majority_tiebreak == "best_expert"
    ):
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

    # --- Majority-Verdict pro Zeile berechnen ---
    maj = result_df.apply(
        lambda r: row_majority(r, expert_colnames, best_expert_col),
        axis=1,
        result_type="expand"
    )

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
        
    # --- Evaluation NUR für finales Ergebnis (Majority) ---
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
                for row_ in cm:
                    print(row_)
                print("Classification report:\n" + report)
            else:
                print("⚠️ Keine auswertbaren Labels/Vorhersagen gefunden.")
        else:
            print("⚠️ Evaluation übersprungen (fehlende Spalten 'label_true' oder 'verdict_final').")
    except Exception as e:
        print(f"⚠️ Evaluation fehlgeschlagen: {e}")

    return result_df
