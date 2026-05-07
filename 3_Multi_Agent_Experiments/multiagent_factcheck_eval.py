import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import logging, json, time, torch
from typing import Literal, List, Dict, Optional, Callable
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

from peft import PeftModel, PeftConfig

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
    max_new_tokens: int = 176

@dataclass
class DecisionConfig:
    name: str
    system: str
    model_id: Optional[str] = None         # None => default_model_id
    max_new_tokens: int = 176

# ---------- Robust JSON extractor ----------
def extract_first_json(text: str):
    dec = json.JSONDecoder()
    s = (text or "").strip()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(s[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("No JSON object found")

def _normalize_labels(series):
    return series.astype(str).str.strip().map(
        {"True": "True", "False": "False", "true": "True", "false": "False"}
    )

# ---------- ModelBundle (HF oder LoRA-Adapter-Ordner) ----------
class ModelBundle:
    def __init__(self, model_id: str):
        self.model_id = model_id

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_cuda = device.type == "cuda"

        torch_dtype = torch.float16 if use_cuda else torch.float32
        device_map = {"": 0} if use_cuda else None

        # CASE A: model_id ist ein PEFT/LoRA-Adapter-Ordner
        adapter_cfg_path = os.path.join(model_id, "adapter_config.json")
        if os.path.exists(adapter_cfg_path):
            peft_cfg = PeftConfig.from_pretrained(model_id)
            base_id = peft_cfg.base_model_name_or_path

            # tokenizer: erst Adapter-Ordner (falls gespeichert), sonst Base
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)

            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            config = AutoConfig.from_pretrained(base_id)
            if use_cuda:
                config.pretraining_tp = 1

            base_model = AutoModelForCausalLM.from_pretrained(
                base_id,
                config=config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )

            lora_model = PeftModel.from_pretrained(base_model, model_id)

            try:
                if use_cuda and torch.cuda.device_count() <= 1:
                    lora_model = lora_model.merge_and_unload()
            except Exception:
                pass

            self.model = lora_model.eval()

        else:
            # CASE B: normales HF Modell
            config = AutoConfig.from_pretrained(model_id)
            if use_cuda:
                config.pretraining_tp = 1

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
            ).eval()

            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            do_sample=False,
            return_full_text=False,
        )


        # LMFE wrapper
        self.enforced_llm = LMFormatEnforcer(
            pipeline=self.pipeline,
            json_schema=JSON_SCHEMA,
        )

    def build_expert_prompt(self, system_prompt: str, statement: str, subject: str) -> str:
        subj = (subject or "").strip()
        subject_block = f"Subjects: {subj}" if subj else "Subjects: (none)"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":
                f"Return ONLY a JSON object with keys verdict and explanation.\n"
                f'Valid verdict values: "True" or "False".\n\n'
                f"Claim: {statement}\n{subject_block}\nJSON:"},
        ]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return (f"[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n"
                    f"Return ONLY a JSON object with keys verdict and explanation.\n"
                    f'Valid verdict values: "True" or "False".\n\n'
                    f"Claim: {statement}\n{subject_block}\nJSON:")

    def build_decision_prompt(
        self,
        system_prompt: str,
        statement: str,
        subject: str,
        panel: List[Dict[str, str]]
    ) -> str:
        subj = (subject or "").strip()
        subject_block = f"Subjects: {subj}" if subj else "Subjects: (none)"

        panel_lines = []
        for p in panel:
            n = p.get("name", "unknown")
            v = p.get("verdict", "None")
            e = (p.get("explanation", "") or "").replace("\n", " ").strip()
            panel_lines.append(f"- Expert: {n} | verdict: {v} | explanation: {e}")
        panel_text = "\n".join(panel_lines) if panel_lines else "- (no expert outputs available)"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":
                f"Return ONLY a JSON object with keys verdict and explanation.\n"
                f'Valid verdict values: "True" or "False".\n\n'
                f"Claim: {statement}\n{subject_block}\n\n"
                f"Expert Panel:\n{panel_text}\n\nJSON:"},
        ]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return (f"[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n"
                    f"Return ONLY a JSON object with keys verdict and explanation.\n"
                    f'Valid verdict values: "True" or "False".\n\n'
                    f"Claim: {statement}\n{subject_block}\n\n"
                    f"Expert Panel:\n{panel_text}\n\nJSON:")

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

            # Wenn LMFE sauber greift, sollte das JSON direkt loadbar sein
            try:
                obj = json.loads((raw or "").strip())
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

            # Fallback: erstes JSON aus Text extrahieren
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
    save_path: Optional[str] = None,
    use_subjects: bool = True
) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep="\t",
        quotechar='"',
        engine="python",
        dtype=str,
        nrows=nrows,
    )

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

    for ex in experts:
        get_model_bundle(ex.model_id)
    if decision:
        get_model_bundle(decision.model_id)

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

        panel_for_decision: List[Dict[str, str]] = []
        for ex in experts:
            bundle = get_model_bundle(ex.model_id)
            subject_val = (out_row["subject"] or "") if use_subjects else ""
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

            out_row[f"verdict_pred__{ex.name}"] = verdict
            out_row[f"explanation__{ex.name}"] = explanation

            panel_for_decision.append({
                "name": ex.name,
                "verdict": verdict if verdict is not None else "None",
                "explanation": explanation or "",
            })

        # Decision
        if decision:
            bundle_dec = get_model_bundle(decision.model_id)
            subject_val = (out_row["subject"] or "") if use_subjects else ""

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


    if save_path:
        dirn = os.path.dirname(save_path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        result_df.to_csv(save_path, index=False, encoding="utf-8")
        print(f"✅ Ergebnisse gespeichert unter: {save_path}")

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
                print("confusion_matrix:")
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
