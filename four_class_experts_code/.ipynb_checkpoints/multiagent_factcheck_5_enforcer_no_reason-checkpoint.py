# ===== Env & Logging (muss ganz oben stehen) =====
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")           # TF-Backend vermeiden
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")         # wähle deine GPU(s)

import logging, json, math, time, torch
from typing import Literal, List, Dict, Optional, Callable
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

from sklearn.metrics import accuracy_score, precision_score,  recall_score, f1_score,  classification_report, confusion_matrix
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
    max_new_tokens: int = 176              # etwas großzügiger als 128

@dataclass
class DecisionConfig:
    name: str                               # z.B. "final_decision"
    system: str                             # System-Prompt fürs Decision-Modell
    model_id: Optional[str] = None          # None => default_model_id
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

    def build_decision_prompt(
        self,
        system_prompt: str,
        statement: str,
        subject: str,
        panel: List[Dict[str, str]]
    ) -> str:
        """
        panel: Liste von dicts mit keys: name, verdict
        WICHTIG: Es werden KEINE Expert:innen-Erklärungen mehr an das Decision-Modell übergeben.
        """
        schema_text = json.dumps(JSON_SCHEMA, ensure_ascii=False)
        subj = (subject or "").strip()
        subject_block = f"Subjects: {subj}" if subj else "Subjects: (none)"
        # Panel als einfache Textliste (nur Namen + Verdict)
        panel_lines = []
        for p in panel:
            n = p.get("name", "unknown")
            v = p.get("verdict", "None")
            panel_lines.append(f"- Expert: {n} | verdict: {v}")

        panel_text = "\n".join(panel_lines) if panel_lines else "- (no expert outputs available)"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":
                f"Use this JSON Schema:\n{schema_text}\n\n"
                f"Claim: {statement}\n{subject_block}\n\n"
                f"Expert Panel (only verdicts):\n{panel_text}\n\n"
                f"JSON:"},
        ]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return (f"[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n"
                    f"Use this JSON Schema:\n{schema_text}\n\n"
                    f"Claim: {statement}\n{subject_block}\n\n"
                    f"Expert Panel (only verdicts):\n{panel_text}\n\nJSON:")

def _normalize_labels(series):
    """Bringt Labels in {'True','False'}-Strings; ignoriert NaN."""
    return series.astype(str).str.strip().map(
        {"True": "True", "False": "False", "true": "True", "false": "False"}
    )

def compute_binary_metrics(df: pd.DataFrame, label_col: str, pred_col: str) -> Dict:
    y_true_all = _normalize_labels(df[label_col].dropna())
    y_pred_all = _normalize_labels(df[pred_col].dropna())
    # nur gemeinsame Indizes werten
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

# globaler Cache für geladene Modelle
_MODEL_CACHE: Dict[str, ModelBundle] = {}

def get_model_bundle(model_id: str) -> ModelBundle:
    if model_id not in _MODEL_CACHE:
        _MODEL_CACHE[model_id] = ModelBundle(model_id)
    return _MODEL_CACHE[model_id]

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

# ========== 3) Hauptfunktion ==========
def process_claims_multi_experts(
    csv_path: str,
    experts_cfg: List[Dict],
    nrows: Optional[int] = None,
    default_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    show_progress: bool = True,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    decision_cfg: Optional[Dict] = None,
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

    # --- Decision normalisieren (optional) ---
    decision: Optional[DecisionConfig] = None
    if decision_cfg:
        if "name" not in decision_cfg or "system" not in decision_cfg:
            raise ValueError("Decision benötigt mindestens 'name' und 'system'.")
        decision = DecisionConfig(
            name=str(decision_cfg["name"]),
            system=str(decision_cfg["system"]),
            model_id=decision_cfg.get("model_id") or default_model_id,
            max_new_tokens=int(decision_cfg.get("max_new_tokens", 176)),
        )

    # --- Modelle vorbereiten ---
    for ex in experts:
        get_model_bundle(ex.model_id)
    if decision:
        get_model_bundle(decision.model_id)

    # --- Fortschritt: 1 Tick pro Statement ---
    total_tasks = len(df)
    iterator = tqdm(
        total=total_tasks,
        desc="🔎 Fact-checking",
        unit="statement",
        dynamic_ncols=True,
        mininterval=0.25,
        smoothing=0.1,
        leave=True,
    ) if show_progress else None

    rows_out = []
    done = 0
    t0 = time.perf_counter()

    for _, row in df.iterrows():
        out_row = {
            "label_true": row.get("label") or row.get("label_true"),
            "statement": row.get("statement"),
            "subject": row.get("subject") or row.get("subjects"),
        }

        # --- Experten laufen lassen ---
        panel_for_decision: List[Dict[str, str]] = []
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

            # In Ausgabe behalten wir weiterhin die Expert:innen-Erklärungen (für Logging/Analyse)
            out_row[f"verdict_pred__{ex.name}"] = verdict
            out_row[f"explanation__{ex.name}"] = explanation

            # WICHTIG: Für das Decision-Modell NUR die Verdicts weitergeben
            panel_for_decision.append({
                "name": ex.name,
                "verdict": verdict if verdict is not None else "None",
            })

        # --- Decision-Modell (optional) ---
        if decision:
            bundle_dec = get_model_bundle(decision.model_id)
            subject_val = out_row["subject"] or ""
            dec_prompt = bundle_dec.build_decision_prompt(
                system_prompt=decision.system,
                statement=out_row["statement"],
                subject=subject_val,
                panel=panel_for_decision,
            )
            dec_verdict, dec_expl = None, None
            try:
                parsed_dec = run_with_retries(
                    bundle_dec,
                    dec_prompt,
                    base_max_new_tokens=decision.max_new_tokens,
                    tries=3,
                    bump=48,
                )
                dec_verdict = parsed_dec.get("verdict")
                dec_expl = parsed_dec.get("explanation")
            except Exception as e:
                dec_expl = f"Error: {e}"

            out_row[f"verdict_pred__{decision.name}"] = dec_verdict
            out_row[f"explanation__{decision.name}"] = dec_expl
            out_row["verdict_final"] = dec_verdict
            out_row["explanation_final"] = dec_expl

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
        # --- optional speichern ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result_df.to_csv(save_path, index=False, encoding="utf-8")
        print(f"✅ Ergebnisse gespeichert unter: {save_path}")
        
    # --- Evaluation NUR für finales Ergebnis ---
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

                print("\n📊 Evaluation (Final Verdict)")
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
