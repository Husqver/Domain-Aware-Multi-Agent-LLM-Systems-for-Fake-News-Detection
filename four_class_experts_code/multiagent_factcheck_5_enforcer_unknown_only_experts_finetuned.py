# ===== Env & Logging (muss ganz oben stehen) =====
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3,4,7")  # ggf. anpassen

import io, re
import logging, json, time, torch
from typing import Literal, List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import json as _json  # falls oben schon 'json' importiert ist

from langchain_experimental.pydantic_v1 import BaseModel, Field
from langchain_experimental.llms import LMFormatEnforcer

logging.basicConfig(level=logging.ERROR)


# ======================= 1) JSON-Schema =======================
class ClaimVerdict(BaseModel):
    verdict: Literal["True", "False", "Unknown"] = Field(
        ...,
        description="Use 'Unknown' if evidence is insufficient, unverifiable, contradictory, or out-of-domain."
    )
    explanation: str = Field(
        ...,
        description="2-4 sentences, brief, no stepwise reasoning"
    )
    # wichtig: einfach halten -> kein Optional/anyOf, damit LMFormatEnforcer nicht ausrastet
    confidence: float = Field(
        0.0,
        description="Confidence in [0,1]."
    )

JSON_SCHEMA = ClaimVerdict.schema()
_ALLOWED = {"True", "False", "Unknown"}


# =================== 2) Helper: Cache & Prompt =================
@dataclass
class Expert:
    name: str
    system: str
    model_id: Optional[str] = None           # HF-ID oder lokaler Pfad (Vollmodell ODER Adapter)
    base_model_id: Optional[str] = None      # nur nötig, wenn model_id ein PEFT/LoRA-Adapter ist
    trust_remote_code: bool = False
    max_new_tokens: int = 176
    weight: float = 1.0                      # Gewicht für Decision-Modell

@dataclass
class DecisionConfig:
    name: str
    system: str
    model_id: Optional[str] = None
    base_model_id: Optional[str] = None
    trust_remote_code: bool = False
    max_new_tokens: int = 176


def _is_peft_adapter(path_or_repo: str) -> bool:
    """
    Heuristik: lokaler Ordner mit 'adapter_config.json' => PEFT/LoRA-Adapter.
    (Für reine Hub-IDs erkennt man es erst beim Laden sicher.)
    """
    try:
        return os.path.isdir(path_or_repo) and os.path.isfile(
            os.path.join(path_or_repo, "adapter_config.json")
        )
    except Exception:
        return False


class ModelBundle:
    """
    Lädt entweder ein vollständiges HF-Modell ODER Basis-Modell + PEFT/LoRA-Adapter.
    - GPU-Fall: device_map='auto' beim Laden -> automatisches Sharding
    - CPU-Fall: alles auf CPU, pipeline kümmert sich automatisch
    Kein manuelles model.to(...) -> kein Device-Mismatch.
    """
    def __init__(self, model_id: str, base_model_id: Optional[str] = None, trust_remote_code: bool = False):
        self.model_id = model_id
        self.base_model_id = base_model_id
        self.trust_remote_code = trust_remote_code

        use_cuda = torch.cuda.is_available()
        self.on_gpu = use_cuda
        self.dtype = torch.float16 if use_cuda else torch.float32

        is_adapter_local = False
        if model_id:
            is_adapter_local = _is_peft_adapter(model_id)

        if is_adapter_local and not base_model_id:
            raise ValueError(
                f"'{model_id}' sieht nach einem PEFT-Adapter aus (adapter_config.json gefunden), "
                "aber 'base_model_id' wurde nicht angegeben."
            )

        model = None
        tokenizer = None

        # =========== 1) Vollmodell versuchen ===========
        if not is_adapter_local and base_model_id is None:
            try:
                config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
                try:
                    config.pretraining_tp = 1
                except Exception:
                    pass

                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    torch_dtype=self.dtype,
                    device_map="auto" if use_cuda else None,
                    trust_remote_code=trust_remote_code,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            except Exception as e:
                print(f"[ModelBundle] Vollmodell-Laden für '{model_id}' fehlgeschlagen: {e}")
                model = None

        # =========== 2) Basis + Adapter (PEFT/LoRA) ===========
        if model is None:
            if base_model_id is None:
                raise ValueError(
                    f"Konnte Vollmodell '{model_id}' nicht laden. "
                    f"Wenn dies ein PEFT/LoRA-Adapter ist, musst du 'base_model_id' angeben."
                )
            from peft import PeftModel

            base_cfg = AutoConfig.from_pretrained(base_model_id, trust_remote_code=trust_remote_code)
            try:
                base_cfg.pretraining_tp = 1
            except Exception:
                pass

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                config=base_cfg,
                torch_dtype=self.dtype,
                device_map="auto" if use_cuda else None,
                trust_remote_code=trust_remote_code,
            )

            model = PeftModel.from_pretrained(base_model, model_id)
            # optional: Merge für reine Inferenz
            try:
                model = model.merge_and_unload()
            except Exception:
                pass

            tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

        self.model = model.eval()
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ---------- Pipeline ----------
        # WICHTIG: kein device_map hier – das Modell ist schon richtig verteilt
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

    def _chat_fallback(self, system_prompt: str, user_content: str) -> str:
        # Llama-3-kompatible Chat-Template-Fallback
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{user_content}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
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
            return self._chat_fallback(
                system_prompt,
                f"Use this JSON Schema:\n{schema_text}\n\nClaim: {statement}\n{subject_block}\nJSON:",
            )

    def build_decision_prompt(
        self, system_prompt: str, statement: str, subject: str, panel: List[Dict[str, str]]
    ) -> str:
        schema_text = json.dumps(JSON_SCHEMA, ensure_ascii=False)
        subj = (subject or "").strip()
        subject_block = f"Subjects: {subj}" if subj else "Subjects: (none)"
        panel_lines = []
        for p in panel:
            n = p.get("name", "unknown")
            v = p.get("verdict", "Unknown")
            e = (p.get("explanation") or "").replace("\n", " ").strip()
            w = p.get("weight", 1.0)
            panel_lines.append(f"- Expert: {n} | weight: {w} | verdict: {v} | explanation: {e}")
        panel_text = "\n".join(panel_lines) if panel_lines else "- (no expert outputs available)"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":
                f"Use this JSON Schema:\n{schema_text}\n\n"
                f"Claim: {statement}\n{subject_block}\n\n"
                f"Expert Panel:\n{panel_text}\n\n"
                f"JSON:"},
        ]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return self._chat_fallback(
                system_prompt,
                f"Use this JSON Schema:\n{schema_text}\n\nClaim: {statement}\n{subject_block}\n\n"
                f"Expert Panel:\n{panel_text}\n\nJSON:",
            )


# =================== 3) Datei-Einlese & Cleaning ===================
def _strip_outer_quotes_and_fix(lines: List[str]) -> List[str]:
    cleaned = []
    for i, line in enumerate(lines):
        s = line.strip()
        if i == len(lines) - 1 and s.endswith("'") and not s.startswith("'"):
            s = s[:-1].rstrip()
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            s = s[1:-1]
        cleaned.append(s)
    return cleaned

_TIDY_REGEXES = [
    (re.compile(r"\s+\."), "."),
    (re.compile(r"\s+,"), ","),
]

def _tidy_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    for rx, repl in _TIDY_REGEXES:
        s = rx.sub(repl, s)
    return s.strip()

def _read_messy_tsv_or_csv(path: str, nrows=None) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.read().splitlines()
    cleaned_lines = _strip_outer_quotes_and_fix(raw_lines)
    buf = io.StringIO("\n".join(cleaned_lines))
    try:
        df = pd.read_csv(
            buf, sep="\t", engine="python", dtype=str, nrows=nrows,
            keep_default_na=False, na_values=[], quotechar='"', on_bad_lines="skip",
        )
    except Exception:
        buf2 = io.StringIO("\n".join(cleaned_lines))
        df = pd.read_csv(
            buf2, sep=None, engine="python", dtype=str, nrows=nrows,
            keep_default_na=False, na_values=[], on_bad_lines="skip",
        )
    df.columns = [c.strip() for c in df.columns]
    for col in ("label", "label_true", "statement", "subject", "subjects"):
        if col in df.columns:
            df[col] = df[col].map(lambda x: _tidy_text(x) if isinstance(x, str) else x)
    return df


# =================== 4) Label & Metrics ===================
def _normalize_labels(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series([], dtype=object)
    s = series.astype(str).str.strip()
    s = s.replace({"": "Unknown", "none": "Unknown", "None": "Unknown", "nan": "Unknown", "NaN": "Unknown"})
    s = s.replace({"true": "True", "false": "False", "unknown": "Unknown"})
    return s

def compute_ternary_metrics(df: pd.DataFrame, label_col: str, pred_col: str) -> Dict:
    labels = ["True", "False", "Unknown"]
    y_true_all = _normalize_labels(df[label_col])
    y_pred_all = _normalize_labels(df[pred_col])

    idx = y_true_all.index.intersection(y_pred_all.index)
    y_true, y_pred = y_true_all.loc[idx], y_pred_all.loc[idx]

    mask = y_true.isin(labels) & y_pred.isin(labels)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if y_true.empty:
        return {"note": "No evaluable rows for ternary metrics."}

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "counts": {"n_eval": int(len(y_true))},
        "columns": {"label_col": label_col, "pred_col": pred_col},
        "labels": labels,
    }


# =================== 5) JSON & Postprocessing ===================
_CODEFENCE_START_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*")
_CODEFENCE_END_RE   = re.compile(r"\s*```\s*$")

def _strip_codefences(s: str) -> str:
    if isinstance(s, str) and s.strip().startswith("```"):
        s = _CODEFENCE_START_RE.sub("", s.strip(), count=1)
        s = _CODEFENCE_END_RE.sub("", s)
    return s

def extract_first_json(text: str):
    """
    Relativ einfacher, robuster JSON-Extractor:
    - entfernt Codefences
    - sucht ab der ersten '{' ein parsebares JSON-Objekt
    """
    if not isinstance(text, str):
        raise ValueError("Model output is not a string")

    s = _strip_codefences(text.strip())
    dec = json.JSONDecoder()

    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(s[i:])
                return obj
            except json.JSONDecodeError:
                continue

    raise ValueError("No JSON object found")

def _postprocess_verdict(verdict: Optional[str], confidence: Optional[float]) -> str:
    v = (str(verdict) if verdict is not None else "").strip().title()
    if v not in _ALLOWED:
        v = "Unknown"
    try:
        if confidence is not None and float(confidence) < 0.6:
            v = "Unknown"
    except Exception:
        pass
    return v


# =================== 6) Model Cache & Invoke ===================
# Cache-Key berücksichtigt (model_id, base_model_id, trust_remote_code)
_MODEL_CACHE: Dict[Tuple[str, Optional[str], bool], ModelBundle] = {}

def get_model_bundle(model_id: str, base_model_id: Optional[str] = None, trust_remote_code: bool = False) -> ModelBundle:
    key = (model_id, base_model_id, bool(trust_remote_code))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = ModelBundle(model_id, base_model_id=base_model_id, trust_remote_code=trust_remote_code)
    return _MODEL_CACHE[key]

def run_with_retries(
    bundle: 'ModelBundle',
    prompt_text: str,
    base_max_new_tokens: int = 176,
    tries: int = 3,
    bump: int = 48,
):
    last_err = None
    raw_last = None
    for t in range(tries):
        try_tokens = base_max_new_tokens + t * bump
        try:
            raw = bundle.enforced_llm.invoke(prompt_text, max_new_tokens=try_tokens)
            raw_last = raw
            return extract_first_json(raw)
        except Exception as e:
            last_err = e
            continue

    # HARD FALLBACK ohne Enforcer
    try_tokens = base_max_new_tokens + tries * bump
    try:
        fallback_prompt = prompt_text + "\n\nReturn exactly ONE valid JSON object, no prose, no code fences."
        gen = bundle.pipeline(
            fallback_prompt,
            max_new_tokens=try_tokens,
            do_sample=False,
        )
        text = gen[0]["generated_text"]
        return extract_first_json(text)
    except Exception as e2:
        raise RuntimeError(
            f"JSON generation failed. last_enforcer_err={last_err}; fallback_err={e2}; raw_last={raw_last!r}"
        )


# ======================== 7) Hauptfunktion ========================
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
    # --- Datei robust einlesen ---
    try:
        df = _read_messy_tsv_or_csv(csv_path, nrows=nrows)
    except Exception as e:
        print(f"Warnung: Fallback auf Standard-Parser wegen {e}")
        df = pd.read_csv(
            csv_path, sep=None, engine="python", dtype=str, nrows=nrows,
            keep_default_na=False, na_values=[], on_bad_lines="skip",
        )
        df.columns = [c.strip() for c in df.columns]

    # --- Expert:innen normalisieren ---
    experts: List[Expert] = []
    for ec in experts_cfg:
        if "name" not in ec or "system" not in ec:
            raise ValueError("Jeder Expert benötigt mindestens 'name' und 'system'.")
        experts.append(Expert(
            name=str(ec["name"]),
            system=str(ec["system"]),
            model_id=ec.get("model_id") or default_model_id,
            base_model_id=ec.get("base_model_id"),
            trust_remote_code=bool(ec.get("trust_remote_code", False)),
            max_new_tokens=int(ec.get("max_new_tokens", 176)),
            weight=float(ec.get("weight", 1.0)),
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
            base_model_id=decision_cfg.get("base_model_id"),
            trust_remote_code=bool(decision_cfg.get("trust_remote_code", False)),
            max_new_tokens=int(decision_cfg.get("max_new_tokens", 176)),
        )

    # --- Modelle vorbereiten ---
    for ex in experts:
        get_model_bundle(ex.model_id, ex.base_model_id, ex.trust_remote_code)
    if decision:
        get_model_bundle(decision.model_id, decision.base_model_id, decision.trust_remote_code)

    # --- Fortschritt ---
    total_tasks = len(df)
    iterator = tqdm(
        total=total_tasks, desc="🔎 Fact-checking", unit="statement",
        dynamic_ncols=True, mininterval=0.25, smoothing=0.1, leave=True,
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
            bundle = get_model_bundle(ex.model_id, ex.base_model_id, ex.trust_remote_code)
            subject_val = out_row["subject"] or ""
            prompt_text = bundle.build_expert_prompt(
                system_prompt=ex.system, statement=out_row["statement"], subject=subject_val,
            )
            verdict, explanation, confidence = None, None, None
            try:
                parsed = run_with_retries(
                    bundle, prompt_text,
                    base_max_new_tokens=ex.max_new_tokens, tries=3, bump=48,
                )
                conf_val = float(parsed.get("confidence", 0.0))
                confidence = conf_val
                verdict = _postprocess_verdict(parsed.get("verdict"), conf_val)
                explanation = parsed.get("explanation")
            except Exception as e:
                verdict = "Unknown"
                explanation = f"Error: {e}"
                confidence = 0.0

            out_row[f"verdict_pred__{ex.name}"] = verdict
            out_row[f"explanation__{ex.name}"] = explanation
            out_row[f"confidence__{ex.name}"] = confidence

            panel_for_decision.append({
                "name": ex.name,
                "verdict": verdict if verdict is not None else "Unknown",
                "explanation": explanation or "",
                "weight": ex.weight,
            })

        # --- Decision-Modell (optional) ---
        if decision:
            bundle_dec = get_model_bundle(decision.model_id, decision.base_model_id, decision.trust_remote_code)
            subject_val = out_row["subject"] or ""
            dec_prompt = bundle_dec.build_decision_prompt(
                system_prompt=decision.system, statement=out_row["statement"],
                subject=subject_val, panel=panel_for_decision,
            )
            dec_verdict, dec_expl, dec_conf = None, None, None
            try:
                parsed_dec = run_with_retries(
                    bundle_dec, dec_prompt,
                    base_max_new_tokens=decision.max_new_tokens, tries=3, bump=48,
                )
                conf_val = float(parsed_dec.get("confidence", 0.0))
                dec_conf = conf_val
                dec_verdict = _postprocess_verdict(parsed_dec.get("verdict"), conf_val)
                dec_expl = parsed_dec.get("explanation")
            except Exception as e:
                dec_verdict = "Unknown"
                dec_expl = f"Error: {e}"
                dec_conf = 0.0

            out_row[f"verdict_pred__{decision.name}"] = dec_verdict
            out_row[f"explanation__{decision.name}"] = dec_expl
            out_row[f"confidence__{decision.name}"] = dec_conf

            out_row["verdict_final"] = dec_verdict
            out_row["explanation_final"] = dec_expl

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

    # --- optional speichern ---
    if save_path:
        dirn = os.path.dirname(save_path) or "."
        os.makedirs(dirn, exist_ok=True)
        result_df.to_csv(save_path, index=False, encoding="utf-8")
        print(f"✅ Ergebnisse gespeichert unter: {save_path}")

    # ===================== Evaluation =====================
    # 1) Ternary (inkl. Unknown)
    try:
        if {"label_true", "verdict_final"}.issubset(result_df.columns):
            tern = compute_ternary_metrics(result_df, "label_true", "verdict_final")
            if "note" not in tern:
                print("\n📊 Ternary evaluation (includes Unknown)")
                print(f"n_evaluable: {tern['counts']['n_eval']}")
                print(f"accuracy: {tern['accuracy']:.3f}")
                print("confusion_matrix (rows=true, cols=pred):")
                for row_cm in tern["confusion_matrix"]:
                    print(row_cm)
                print("Classification report:\n" + tern["report"])
            else:
                print("⚠️ " + tern["note"])
        else:
            print("⚠️ Ternary eval skipped (missing 'label_true' or 'verdict_final').")
    except Exception as e:
        print(f"⚠️ Ternary evaluation failed: {e}")

    # 2) Binary (Unknown ignorieren)
    try:
        if "label_true" in result_df.columns and "verdict_final" in result_df.columns:
            y_true = _normalize_labels(result_df["label_true"])
            y_pred = _normalize_labels(result_df["verdict_final"])
            mask_bin = y_true.isin({"True", "False"}) & y_pred.isin({"True", "False"})
            y_true_bin = y_true[mask_bin]
            y_pred_bin = y_pred[mask_bin]
            if not y_true_bin.empty:
                acc = accuracy_score(y_true_bin, y_pred_bin)
                prec = precision_score(y_true_bin, y_pred_bin, pos_label="True", zero_division=0)
                rec  = recall_score(y_true_bin, y_pred_bin, pos_label="True", zero_division=0)
                f1   = f1_score(y_true_bin, y_pred_bin, pos_label="True", zero_division=0)
                cm   = confusion_matrix(y_true_bin, y_pred_bin, labels=["True", "False"]).tolist()
                report = classification_report(y_true_bin, y_pred_bin, labels=["True", "False"], zero_division=0)
                print("\n📊 Binary evaluation (Unknown ignored)")
                print(f"n_total: {len(result_df)} | n_scored: {len(y_true_bin)} | n_ignored: {len(result_df) - len(y_true_bin)}")
                print(f"accuracy: {acc:.3f}")
                print(f"precision_true: {prec:.3f}")
                print(f"recall_true: {rec:.3f}")
                print(f"f1_true: {f1:.3f}")
                print("confusion_matrix (rows=true, cols=pred):")
                for row_cm in cm:
                    print(row_cm)
                print("Classification report:\n" + report)
            else:
                print("⚠️ Keine auswertbaren (nicht-Unknown) Paare für die Binary-Evaluation.")
        else:
            print("⚠️ Evaluation übersprungen (fehlende Spalten 'label_true' oder 'verdict_final').")
    except Exception as e:
        print(f"⚠️ Evaluation fehlgeschlagen: {e}")

    return result_df
