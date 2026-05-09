# ===== Env & Logging (muss ganz oben stehen) =====
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import logging, json, math, time, torch
from typing import Literal, List, Dict, Optional, Callable, Tuple
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

# ========== 1) JSON-Schemata ==========
class ClaimVerdict(BaseModel):
    verdict: Literal["True", "False"] = Field(..., description="Binary verdict")
    explanation: str = Field(..., description="2-4 sentences, brief, no stepwise reasoning")

JSON_SCHEMA_VERDICT = ClaimVerdict.schema()

class RoutingWeights(BaseModel):
    subject_detected: Literal["politics","economy","health","other"] = Field(..., description="Detected coarse subject")
    weights: Dict[str, float] = Field(..., description="Keys: politics,economy,health. Values in [0,1], sum≈1")
    rationale: str = Field(..., description="1-2 short sentences why these weights")

JSON_SCHEMA_ROUTER = RoutingWeights.schema()

# ========== 2) Helper: Modell-Cache & Prompt ==========

@dataclass
class Expert:
    name: str
    system: str
    model_id: Optional[str] = None
    base_model_id: Optional[str] = None   # für PEFT/Adapter
    max_new_tokens: int = 176

@dataclass
class DecisionConfig:
    name: str
    system: str
    model_id: Optional[str] = None
    base_model_id: Optional[str] = None   # für PEFT/Adapter
    max_new_tokens: int = 176

@dataclass
class RouterConfig:
    name: str
    system: str
    model_id: Optional[str] = None
    base_model_id: Optional[str] = None   # für PEFT/Adapter
    max_new_tokens: int = 128

def _is_peft_adapter(path_or_repo: str) -> bool:
    """
    Heuristik für lokale Ordner: adapter_config.json vorhanden?
    Für Hub-IDs sicher erst beim Laden erkennbar; hier reicht lokale Erkennung.
    """
    try:
        return os.path.isdir(path_or_repo) and os.path.isfile(os.path.join(path_or_repo, "adapter_config.json"))
    except Exception:
        return False

class ModelBundle:
    """
    Lädt entweder ein vollständiges HF-Modell ODER einen PEFT/LoRA-Adapter + Basis-Modell.
    Nutzt bei CUDA float16 und device_map='auto'. Unterstützt trust_remote_code bei Bedarf.
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
                # Vollmodell
                config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
                if device.type == "cuda":
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
                # Adapter: Basis + Adapter
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
                # Für reine Inferenz ist Merge oft sinnvoll (schneller, weniger Overhead)
                try:
                    model = model.merge_and_unload()
                except Exception:
                    pass
                model = model.eval()
                tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

        except ValueError as e:
            # Fallback: Wenn Vollmodell-Laden fehlschlug, aber base_model_id vorhanden ist, als Adapter interpretieren
            if base_model_id:
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
        # Default Enforcer für ClaimVerdict; für Router nutzen wir dynamisch ein anderes Schema
        self.enforced_llm_verdict = LMFormatEnforcer(pipeline=self.pipeline, json_schema=JSON_SCHEMA_VERDICT)

    def _apply_template(self, system_prompt: str, user_content: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n{user_content}\n"

    def build_expert_prompt(self, system_prompt: str, statement: str, subject: str) -> str:
        schema_text = json.dumps(JSON_SCHEMA_VERDICT, ensure_ascii=False)
        subj = (subject or "").strip()
        subject_block = f"Subjects: {subj}" if subj else "Subjects: (none)"
        user = f"Use this JSON Schema:\n{schema_text}\n\nClaim: {statement}\n{subject_block}\nJSON:"
        return self._apply_template(system_prompt, user)

    def build_router_prompt(self, system_prompt: str, statement: str, subject: str) -> str:
        schema_text = json.dumps(JSON_SCHEMA_ROUTER, ensure_ascii=False)
        subj = (subject or "").strip()
        subject_block = f"Subjects(meta): {subj}" if subj else "Subjects(meta): (none)"
        user = (
            f"Infer a coarse subject and weights for domain experts.\n"
            f"Use this JSON Schema:\n{schema_text}\n\n"
            f"Claim: {statement}\n{subject_block}\nJSON:"
        )
        return self._apply_template(system_prompt, user)

def _normalize_labels(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().map({
        "True":"True","False":"False","true":"True","false":"False",
        "TRUE":"True","FALSE":"False","T":"True","F":"False","1":"True","0":"False"
    })

def compute_binary_metrics(df: pd.DataFrame, label_col: str, pred_col: str) -> Dict:
    y_true_all = _normalize_labels(df[label_col].dropna())
    y_pred_all = _normalize_labels(df[pred_col].dropna())
    idx = y_true_all.index.intersection(y_pred_all.index)
    y_true, y_pred = y_true_all.loc[idx], y_pred_all.loc[idx]
    if y_true.empty:
        return {"counts":{"n_total":int(df.shape[0]),"n_eval":0},
                "note":f"No evaluable rows for label_col='{label_col}' and pred_col='{pred_col}'."}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_true": float(precision_score(y_true, y_pred, pos_label="True", zero_division=0)),
        "recall_true": float(recall_score(y_true, y_pred, pos_label="True", zero_division=0)),
        "f1_true": float(f1_score(y_true, y_pred, pos_label="True", zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=["True","False"], zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["True","False"]).tolist(),
        "counts":{"n_total":int(df.shape[0]),"n_eval":int(len(idx))},
        "columns":{"label_col":label_col,"pred_col":pred_col},
    }

def extract_first_json(text: str):
    dec = json.JSONDecoder(); s = text.strip()
    for i,ch in enumerate(s):
        if ch=="{":
            try:
                obj,_ = dec.raw_decode(s[i:]); return obj
            except json.JSONDecodeError: continue
    raise ValueError("No JSON object found")

# Cache-Key muss base_model_id berücksichtigen (Adapter-Fall!)
_MODEL_CACHE: Dict[Tuple[str, Optional[str]], ModelBundle] = {}

def get_model_bundle(model_id: str, base_model_id: Optional[str] = None) -> ModelBundle:
    key = (model_id, base_model_id)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = ModelBundle(model_id, base_model_id=base_model_id)
    return _MODEL_CACHE[key]

def run_with_retries_json(
    bundle: ModelBundle,
    prompt_text: str,
    schema: Dict,
    base_max_new_tokens: int = 160,
    tries: int = 3,
    bump: int = 40,
):
    """
    Erzwingt JSON gemäß `schema` (dynamisch). Nutzt pro Call einen frischen Enforcer,
    um unterschiedliche Schemata (Verdict vs Router) zu unterstützen.
    """
    last_err = None
    for t in range(tries):
        try_tokens = base_max_new_tokens + t*bump
        try:
            enforced = LMFormatEnforcer(pipeline=bundle.pipeline, json_schema=schema)
            raw = enforced.invoke(prompt_text, max_new_tokens=try_tokens)
            return extract_first_json(raw)
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("Unknown generation error")

# ========== 3) Mapping & Gewichtete Mehrheit ==========
def map_expert_to_domain(expert_name: str) -> str:
    name = expert_name.lower()
    if "politic" in name:
        return "politics"
    if "econom" in name:
        return "economy"
    if "health" in name or "science" in name or "med" in name:
        return "health"
    # Default: gleichmäßig werten
    return "economy"  # (oder 'politics') – hier festlegen, wenn unbekannt

def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    # ensure keys exist + nonneg + sum>0
    keys = ["politics","economy","health"]
    vals = {k: max(0.0, float(w.get(k, 0.0))) for k in keys}
    s = sum(vals.values())
    if s <= 0:
        # fallback: uniform
        return {k: 1.0/3.0 for k in keys}
    return {k: vals[k]/s for k in keys}

def weighted_majority(votes: List[Dict[str,str]], weights: Dict[str,float]) -> Dict[str, float]:
    """
    votes: [{name: expert_name, verdict: "True"/"False"}]
    weights: domain->weight
    return: dict with sum_true, sum_false, label, margin
    """
    sum_true = 0.0; sum_false = 0.0
    for v in votes:
        verdict = str(v.get("verdict","")).strip()
        if verdict.lower() not in ("true","false"):
            continue
        domain = map_expert_to_domain(v.get("name",""))
        w = float(weights.get(domain, 0.0))
        if verdict.lower()=="true":
            sum_true += w
        else:
            sum_false += w
    label = "True" if sum_true > sum_false else ("False" if sum_false > sum_true else "tie")
    margin = abs(sum_true - sum_false)
    return {"sum_true": sum_true, "sum_false": sum_false, "label": label, "margin": margin}

# ========== 4) Hauptfunktion (Router + Weights im Decision-Prompt; Guardrails optional) ==========
def process_claims_multi_experts(
    csv_path: str,
    experts_cfg: List[Dict],
    nrows: Optional[int] = None,
    default_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    show_progress: bool = True,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    router_cfg: Optional[Dict] = None,      # Router-Experte für subject/weights
    decision_cfg: Optional[Dict] = None,    # Decision-LLM
    save_path: Optional[str] = None,
    MAJ_THRESH: float = 0.25,               # Margin für optionale Guardrails (gewichtete Mehrheit)
    enforce_guardrails: bool = False,       # False => Decision-LLM ist Chef; True => Majority/Unanimity kann finalisieren
) -> pd.DataFrame:

    # --- CSV einlesen ---
    df = pd.read_csv(
        csv_path, sep="\t", quotechar='"', engine="python", dtype=str, nrows=nrows,
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
            base_model_id=ec.get("base_model_id"),  # Adapter-Unterstützung
            max_new_tokens=int(ec.get("max_new_tokens", 176)),
        ))

    # --- Router + Decision normalisieren ---
    router: Optional[RouterConfig] = None
    if router_cfg:
        if "name" not in router_cfg or "system" not in router_cfg:
            raise ValueError("Router benötigt mindestens 'name' und 'system'.")
        router = RouterConfig(
            name=str(router_cfg["name"]),
            system=str(router_cfg["system"]),
            model_id=router_cfg.get("model_id") or default_model_id,
            base_model_id=router_cfg.get("base_model_id"),
            max_new_tokens=int(router_cfg.get("max_new_tokens", 128)),
        )

    decision: Optional[DecisionConfig] = None
    if decision_cfg:
        if "name" not in decision_cfg or "system" not in decision_cfg:
            raise ValueError("Decision benötigt mindestens 'name' und 'system'.")
        decision = DecisionConfig(
            name=str(decision_cfg["name"]),
            system=str(decision_cfg["system"]),
            model_id=decision_cfg.get("model_id") or default_model_id,
            base_model_id=decision_cfg.get("base_model_id"),
            max_new_tokens=int(decision_cfg.get("max_new_tokens", 176)),
        )

    # --- Modelle vorbereiten ---
    for ex in experts:
        get_model_bundle(ex.model_id, ex.base_model_id)
    if router:
        get_model_bundle(router.model_id, router.base_model_id)
    if decision:
        get_model_bundle(decision.model_id, decision.base_model_id)

    # --- Fortschritt ---
    total_tasks = len(df)
    iterator = tqdm(total=total_tasks, desc="🔎 Fact-checking (router + weighted context)", unit="statement",
                    dynamic_ncols=True, mininterval=0.25, smoothing=0.1, leave=True) if show_progress else None

    rows_out = []; done = 0; t0 = time.perf_counter()

    for _, row in df.iterrows():
        out_row = {
            "label_true": row.get("label") or row.get("label_true"),
            "statement": row.get("statement"),
            "subject": row.get("subject") or row.get("subjects"),
        }

        # --- 1) Router (subject + weights) ---
        weights = {"politics": 1/3, "economy": 1/3, "health": 1/3}
        router_subject = "other"
        router_rationale = ""
        if router:
            bundle_r = get_model_bundle(router.model_id, router.base_model_id)
            r_prompt = bundle_r.build_router_prompt(
                system_prompt=router.system,
                statement=out_row["statement"],
                subject=out_row["subject"] or "",
            )
            try:
                parsed_r = run_with_retries_json(
                    bundle=bundle_r,
                    prompt_text=r_prompt,
                    schema=JSON_SCHEMA_ROUTER,
                    base_max_new_tokens=router.max_new_tokens,
                    tries=3, bump=32,
                )
                weights = normalize_weights(parsed_r.get("weights") or {})
                router_subject = parsed_r.get("subject_detected") or "other"
                router_rationale = parsed_r.get("rationale") or ""
            except Exception as e:
                router_rationale = f"RouterError: {e}"
        out_row["router_subject"] = router_subject
        out_row["router_weight_politics"] = weights["politics"]
        out_row["router_weight_economy"]  = weights["economy"]
        out_row["router_weight_health"]   = weights["health"]
        out_row["router_rationale"]       = router_rationale

        # --- 2) Experten laufen lassen ---
        panel_for_decision: List[Dict[str, str]] = []
        expert_verdict_cols: List[str] = []
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
                parsed = run_with_retries_json(
                    bundle=bundle,
                    prompt_text=prompt_text,
                    schema=JSON_SCHEMA_VERDICT,
                    base_max_new_tokens=ex.max_new_tokens,
                    tries=3, bump=48,
                )
                verdict = parsed.get("verdict")
                explanation = parsed.get("explanation")
            except Exception as e:
                explanation = f"Error: {e}"

            col_v = f"verdict_pred__{ex.name}"
            col_e = f"explanation__{ex.name}"
            out_row[col_v] = verdict
            out_row[col_e] = explanation
            expert_verdict_cols.append(col_v)

            panel_for_decision.append({
                "name": ex.name,
                "verdict": verdict if verdict is not None else "None",
                "explanation": explanation or "",
            })

        # --- 3) Stimmen sammeln (für Logging/Fallback) ---
        votes_clean = [{"name": p["name"], "verdict": p["verdict"]} for p in panel_for_decision
                       if str(p["verdict"]).lower() in ("true","false")]
        n_votes = len(votes_clean)
        n_true = sum(1 for v in votes_clean if str(v["verdict"]).lower()=="true")
        n_false = n_votes - n_true

        def finalize(label: str, rule: str, extra: str = ""):
            out_row["verdict_final"] = label
            out_row["explanation_final"] = (
                f"{rule} | votes True={n_true}, False={n_false}, total={n_votes} | "
                f"weights(p,e,h)=({weights['politics']:.2f},{weights['economy']:.2f},{weights['health']:.2f})"
                + (f" | {extra}" if extra else "")
            )
            return True

        finalized = False

        # --- OPTIONAL: Guardrails (nur wenn enforce_guardrails=True) ---
        if enforce_guardrails:
            # Einstimmigkeit
            if n_votes > 0 and (n_true == n_votes or n_false == n_votes):
                label = "True" if n_true == n_votes else "False"
                finalized = finalize(label, "guardrail_unanimity")

            # Gewichtete Mehrheit mit Margin
            if not finalized and n_votes >= 2:
                wmaj = weighted_majority(votes_clean, weights)
                if wmaj["label"] in ("True","False") and wmaj["margin"] >= MAJ_THRESH:
                    finalized = finalize(wmaj["label"], "guardrail_weighted_majority", f"margin={wmaj['margin']:.2f}")

        # --- 4) Decision-LLM (Chef) ---
        if decision and not finalized:
            bundle_dec = get_model_bundle(decision.model_id, decision.base_model_id)
            subject_val = out_row["subject"] or ""

            # Kompaktes, gewichtet annotiertes Panel
            def short(text, n=220):
                if not isinstance(text, str):
                    return ""
                text = " ".join(text.split())
                return text if len(text) <= n else (text[:n-1] + "…")

            panel_lines = []
            for p in panel_for_decision:
                domain = map_expert_to_domain(p["name"])
                w = weights.get(domain, 0.0)
                v = p.get("verdict", "None")
                e = short(p.get("explanation", ""))
                panel_lines.append(f"- {p['name']} | domain={domain} | weight={w:.2f} | verdict={v} | note={e}")
            panel_summary = "\n".join(panel_lines)

            dec_system = (
                decision.system.rstrip() +
                "\n\nCONTEXT RULES:\n"
                "- Consider the provided domain weights when aggregating expert opinions.\n"
                "- Do not assume unobserved evidence. Prefer well-supported, domain-relevant arguments.\n"
                "- Output strictly valid JSON per schema.\n"
            )

            dec_prompt = bundle_dec._apply_template(
                dec_system,
                user_content=(
                    f"Use this JSON Schema:\n{json.dumps(JSON_SCHEMA_VERDICT, ensure_ascii=False)}\n\n"
                    f"Claim: {out_row['statement']}\n"
                    f"Subjects: {subject_val or '(none)'}\n\n"
                    f"Router:\n"
                    f"- subject_detected: {router_subject}\n"
                    f"- weights: politics={weights['politics']:.2f}, economy={weights['economy']:.2f}, health={weights['health']:.2f}\n\n"
                    f"Expert Panel (name | domain | weight | verdict | note):\n{panel_summary}\n\n"
                    f"JSON:"
                )
            )

            dec_verdict, dec_expl = None, None
            try:
                parsed_dec = run_with_retries_json(
                    bundle=bundle_dec,
                    prompt_text=dec_prompt,
                    schema=JSON_SCHEMA_VERDICT,
                    base_max_new_tokens=decision.max_new_tokens,
                    tries=3, bump=48,
                )
                dec_verdict = parsed_dec.get("verdict")
                dec_expl = parsed_dec.get("explanation")
            except Exception as e:
                dec_expl = f"Error: {e}"

            out_row[f"verdict_pred__{decision.name}"] = dec_verdict
            out_row[f"explanation__{decision.name}"] = dec_expl

            # Failsafe falls Decision-LLM kein valides Verdict liefert
            if dec_verdict not in ("True","False"):
                wmaj = weighted_majority(votes_clean, weights)
                if wmaj["label"] in ("True","False"):
                    finalize(wmaj["label"], "fallback_weighted_majority", f"margin={wmaj['margin']:.2f}")
                else:
                    finalize("False", "fallback_default_false", "no_valid_decision_output")
            else:
                out_row["verdict_final"] = dec_verdict
                out_row["explanation_final"] = dec_expl

        rows_out.append(out_row)

        # Fortschritt
        done += 1
        if iterator: iterator.update(1)
        if progress_callback:
            elapsed = time.perf_counter() - t0
            eta = (elapsed / max(done,1)) * (total_tasks - done)
            progress_callback(done, total_tasks, eta)

    if iterator: iterator.close()

    result_df = pd.DataFrame(rows_out)

    # --- optional speichern ---
    if save_path:
        import os; os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result_df.to_csv(save_path, index=False, encoding="utf-8")
        print(f"✅ Ergebnisse gespeichert unter: {save_path}")

    # --- Evaluation (falls vorhanden) ---
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
                cm = confusion_matrix(y_true, y_pred, labels=["True","False"]).tolist()
                report = classification_report(y_true, y_pred, labels=["True","False"], zero_division=0)
                print("\n📊 Evaluation (Final Verdict, router-weights context)")
                print(f"accuracy: {acc:.3f}")
                print(f"precision_true: {prec:.3f}")
                print(f"recall_true: {rec:.3f}")
                print(f"f1_true: {f1:.3f}")
                print("confusion_matrix:")
                for r in cm: print(r)
                print("Classification report:\n" + report)
            else:
                print("⚠️ Keine auswertbaren Labels/Vorhersagen gefunden.")
        else:
            print("⚠️ Evaluation übersprungen (fehlende Spalten 'label_true' oder 'verdict_final').")
    except Exception as e:
        print(f"⚠️ Evaluation fehlgeschlagen: {e}")

    return result_df
