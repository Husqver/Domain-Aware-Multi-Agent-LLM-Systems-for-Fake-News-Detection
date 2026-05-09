import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import torch
import pandas as pd
from typing import Literal, Dict, Any, List, Optional
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline as hf_pipeline,
)
from peft import PeftModel

from langchain_experimental.pydantic_v1 import BaseModel, Field
from langchain_experimental.llms import LMFormatEnforcer

# === Environment-Settings (optional wie bei dir) ===
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # sichtbare GPUs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================
# 1) Hilfsfunktionen & Schemas
# =========================

def norm_bool_label(x: str) -> str:
    if not isinstance(x, str):
        return "False"
    s = x.strip().strip(".,:;!?)(").lower()
    if s.startswith("true"):
        return "True"
    if s.startswith("false"):
        return "False"
    return "False"


def extract_first_json(text_or_obj):
    """
    Versucht robust, ein JSON-Objekt zu extrahieren.
    """
    if isinstance(text_or_obj, dict):
        return text_or_obj

    if isinstance(text_or_obj, list):
        for el in text_or_obj:
            if isinstance(el, dict):
                return el
            if isinstance(el, str):
                try:
                    dec = json.JSONDecoder()
                    s = el.strip()
                    for i, ch in enumerate(s):
                        if ch == "{":
                            try:
                                obj, _ = dec.raw_decode(s[i:])
                                return obj
                            except json.JSONDecodeError:
                                continue
                except Exception:
                    continue

    if not isinstance(text_or_obj, str):
        raise ValueError(f"Cannot parse JSON from type {type(text_or_obj)}")

    dec = json.JSONDecoder()
    s = text_or_obj.strip()
    for i, ch in enumerate(s):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(s[i:])
                return obj
            except json.JSONDecodeError:
                continue

    raise ValueError("No JSON object found in model output")


# =========================
# 2) Checkability-Gate (schnell, ohne LMFormatEnforcer)
# =========================

CHECKABILITY_SYSTEM_PROMPT = """
You are a strict classifier for factual checkability of claims.

You receive:
- a political or public policy claim (statement)
- optionally some metadata (speaker, place, topics, context)

Your task: Decide whether this statement can be objectively fact-checked
OR whether it should be excluded from automated fact-checking.

You MUST return exactly one of these categories:

- "fact_checkable":
  A concrete, factual claim that can be verified with objective evidence
  (e.g., numbers, votes, events, dates, laws).

- "non_claim":
  No clear factual proposition. Examples: topic headings, fragments,
  descriptions like "On residency requirements for public workers".

- "opinion_or_ambiguous":
  Mainly opinions, value judgements, vague or causal blame
  (e.g., "we have this mess because of X", "this is the worst ever"),
  where no clear factual benchmark exists.

- "needs_additional_context":
  The statement could in principle be checked, but only if we know
  exactly who, when, where, or which entity it refers to.

- "sensitive_selfharm":
  Statements primarily about suicide or self-harm, especially generalizations
  like "if someone is determined to commit suicide, X doesn't matter".
  These are ethically sensitive and should be excluded from automatic
  true/false classification.

Return ONLY a JSON object with fields:
{
  "category": "..."
}
No markdown, no extra keys, no stepwise reasoning, no explanation.
"""


# ----- Verdict-Schema für Experten -----

class ClaimVerdict(BaseModel):
    verdict: Literal["True", "False"] = Field(..., description="Binary verdict")
    explanation: str = Field(..., description="2–4 sentences, brief, no stepwise reasoning")

JSON_SCHEMA = ClaimVerdict.schema()


def build_expert_system_prompt(domain_name: str) -> str:
    return f"""You are a fact-checking expert specialized in the '{domain_name}' domain.

You receive a political or public policy claim (statement) and must decide if it is factually correct.

You must output a JSON object that follows this schema:

{json.dumps(JSON_SCHEMA, indent=2)}

The field "verdict" MUST be either "True" or "False".
The field "explanation" MUST be 2–4 concise sentences. 
Do NOT include step-by-step reasoning or lists.

Reply with JSON only. No markdown, no extra text.
"""


ROUTER_SYSTEM_PROMPT = """You are a strict domain classifier.

You receive a political or public policy claim (statement) and must map it
to exactly ONE high-level domain label from a given list.

You MUST answer with EXACTLY ONE label string from the list.
Do NOT explain. Only output the label.
"""


# =========================
# 2) Checkability-Gate
# =========================

def run_checkability_gate(
    df: pd.DataFrame,
    base_model_id: str,
    text_col: str = "statement",
    domain_col: str = "super_domain",
) -> pd.DataFrame:
    """
    Läuft einmal über das DataFrame und erzeugt:
    - checkability_category
    - checkability_explanation

    Nutzt nur das Base-Modell (kein LoRA), auf GPU falls verfügbar.
    """
    print("\n=== Checkability-Gate: Welche Claims sind überhaupt prüfbar? ===")

    use_cuda = torch.cuda.is_available()
    gate_device = torch.device("cuda" if use_cuda else "cpu")
    torch_dtype = torch.float16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
    ).to(gate_device)

    gen_pipe = hf_pipeline(
        "text-generation",
        model=base,
        tokenizer=tokenizer,
        device=0 if use_cuda else -1,
        do_sample=False,
        return_full_text=False,
    )

    cats = []
    exps = []

    for row in tqdm(
        df.itertuples(),
        total=len(df),
        desc="🔎 Checkability",
        dynamic_ncols=True,
    ):
        statement = getattr(row, text_col)

        user_content = (
            f"Statement:\n{statement}\n\n"
            "Return the JSON object now."
        )

        messages = [
            {"role": "system", "content": CHECKABILITY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt_text = (
                f"[SYSTEM]\n{CHECKABILITY_SYSTEM_PROMPT}\n[/SYSTEM]\n"
                f"User:\n{user_content}\nAssistant (JSON only):"
            )

        try:
            out = gen_pipe(prompt_text, max_new_tokens=120)[0]["generated_text"]
            obj = extract_first_json(out)
            cat = obj.get("category", "opinion_or_ambiguous")
            exp = obj.get("explanation", "").strip()
        except Exception as e:
            print(f"[WARN] Checkability JSON parse failed: {e}")
            cat = "opinion_or_ambiguous"
            exp = f"Fallback due to parsing error: {str(e)[:200]}"

        if not exp:
            exp = "No explanation was provided."

        cats.append(cat)
        exps.append(exp)

    del base
    if use_cuda:
        torch.cuda.empty_cache()

    return pd.DataFrame(
        {
            "checkability_category": cats,
            "checkability_explanation": exps,
        },
        index=df.index,
    )


# =========================
# 3) Router: Statement -> super_domain
# =========================

def run_router(
    df: pd.DataFrame,
    base_model_id: str,
    router_lora_dir: str,
    super_labels: List[str],
    text_col: str = "statement",
) -> pd.Series:
    print("\n=== Phase 1: Routing (Statement -> super_domain) ===")

    try:
        tokenizer = AutoTokenizer.from_pretrained(router_lora_dir, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    model = PeftModel.from_pretrained(base, router_lora_dir)
    model = model.to(device)
    model.eval()

    import re

    preds = []
    label_list = ", ".join(super_labels)

    for row in tqdm(df.itertuples(), total=len(df),
                   desc="🌍 Routing-only", dynamic_ncols=True):
        statement = getattr(row, text_col)

        user_content = (
            f"Valid labels:\n{label_list}\n\n"
            f"Statement:\n{statement}\n\n"
            "Return ONLY ONE label from the list above.\n"
            "Respond ONLY with the label.\n\n"
            "Label:"
        )

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            gen = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=8,
                do_sample=False,
            )

        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

        if not answer:
            preds.append("misc")
            continue

        pred_dom = "misc"
        for L in super_labels:
            pattern = r"\b" + re.escape(L.lower()) + r"\b"
            if re.search(pattern, answer):
                pred_dom = L
                break
        else:
            for L in super_labels:
                if L.lower() in answer:
                    pred_dom = L
                    break

        preds.append(pred_dom)

    del model
    del base
    torch.cuda.empty_cache()

    return pd.Series(preds, index=df.index, name="domain_pred_router")


# =========================
# 4) Experten-LoRAs + LMFormatEnforcer
# =========================

def run_expert_for_domain(
    df: pd.DataFrame,
    domain_name: str,
    base_model_id: str,
    expert_root: str,
    text_col: str = "statement",
):
    expert_dir = os.path.join(expert_root, domain_name)
    if not os.path.isdir(expert_dir):
        raise FileNotFoundError(f"Expert directory for domain '{domain_name}' not found: {expert_dir}")

    print(f"\n=== Experten-Fact-Check für Domain '{domain_name}' (n={len(df)}) ===")

    try:
        tokenizer = AutoTokenizer.from_pretrained(expert_dir, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    model = PeftModel.from_pretrained(base, expert_dir)
    model = model.to(device)
    model.eval()

    gen_pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device.type == "cuda" else -1,
        do_sample=False,
        return_full_text=False,
    )

    enforced_llm = LMFormatEnforcer(
        pipeline=gen_pipe,
        json_schema=JSON_SCHEMA,
    )

    def fact_check_one(statement: str) -> Dict[str, Any]:
        system_prompt = build_expert_system_prompt(domain_name)
        user_content = f"Claim:\n{statement}\n\nReturn the JSON object now."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt_text = (
                f"[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n"
                f"User:\n{user_content}\nAssistant (JSON only):"
            )

        try:
            raw = enforced_llm.invoke(prompt_text, max_new_tokens=192)
            parsed = extract_first_json(raw)
            verdict = parsed.get("verdict")
            expl = parsed.get("explanation", "").strip()
        except Exception as e:
            print(f"[WARN] JSON parse failed for domain '{domain_name}': {e}")
            verdict = "False"
            expl = f"Fallback due to parsing error: {str(e)[:200]}"

        if verdict not in ["True", "False"]:
            verdict = "False"
        if not expl:
            expl = "No explanation was provided."

        return {"verdict": verdict, "explanation": expl}

    verdicts = []
    explanations = []

    for row in tqdm(df.itertuples(), total=len(df),
                   desc=f"🧠 {domain_name}", dynamic_ncols=True):
        statement = getattr(row, text_col)
        res = fact_check_one(statement)
        verdicts.append(norm_bool_label(res["verdict"]))
        explanations.append(res["explanation"])

    del model
    del base
    torch.cuda.empty_cache()

    return pd.Series(verdicts, index=df.index), pd.Series(explanations, index=df.index)


# =========================
# 5) Komplettpipeline mit Checkability-Gate
# =========================

def run_full_pipeline_with_gate(
    base_model_id: str,
    router_lora_dir: str,
    expert_root: str,
    super_labels: List[str],
    test_path: str,
    out_path: Optional[str] = None,
    text_col: str = "statement",
    label_col: str = "label",
    domain_col: str = "super_domain",
    sep: str = "\t",
) -> pd.DataFrame:
    """
    Vollständige Pipeline:
      1) Testdaten laden
      2) Checkability-Gate -> fact_checkable vs. Rest
      3) Router nur auf fact_checkable
      4) Experten nur auf fact_checkable
      5) Metriken (Routing + Verdict) nur für fact_checkable
      6) Optional alles in CSV schreiben (inkl. Checkability-Spalten)
    """

    # --- Testdaten laden ---
    df = pd.read_csv(
        test_path,
        sep=sep,
        quotechar='"',
        engine="python",
        dtype=str,
    )

    df = df.dropna(subset=[text_col, label_col])
    df = df[df[text_col].str.strip() != ""]
    df = df[df[label_col].isin(["True", "False"])]

    print("Testset size (nach Initial-Filtern):", len(df))
    if domain_col in df.columns:
        print("Super-Domain-Verteilung (true):")
        print(df[domain_col].value_counts())

    # --- Checkability-Gate ---
    gate_df = run_checkability_gate(
        df=df,
        base_model_id=base_model_id,
        text_col=text_col,
        domain_col=domain_col,
    )
    df = pd.concat([df, gate_df], axis=1)

    print("\nCheckability-Kategorien:")
    print(df["checkability_category"].value_counts())

    mask_fact = df["checkability_category"] == "fact_checkable"
    df_fact = df[mask_fact].copy()

    print(f"\nBewertbare Claims (fact_checkable): {len(df_fact)}")
    print(f"Ausgeschlossene Claims (nicht fact_checkable): {len(df) - len(df_fact)}")

    if df_fact.empty:
        raise RuntimeError("Keine fact_checkable Claims gefunden – Checkability-Gate ist zu streng oder Datenproblem.")

    # --- Router nur auf fact_checkable ---
    df_fact["domain_pred_router"] = run_router(
        df=df_fact,
        base_model_id=base_model_id,
        router_lora_dir=router_lora_dir,
        super_labels=super_labels,
        text_col=text_col,
    )

    print("\nRouting-only Verteilung (nur fact_checkable):")
    print(df_fact["domain_pred_router"].value_counts())

    df_fact["domain_pred"] = df_fact["domain_pred_router"]
    df_fact["verdict_pred"] = "False"
    df_fact["explanation_pred"] = ""

    # --- Experten pro Domain ---
    for dom in super_labels:
        idxs = df_fact.index[df_fact["domain_pred_router"] == dom].tolist()
        if not idxs:
            print(f"\n⚠️ Keine vom Router zugewiesenen Beispiele für Domain '{dom}'.")
            continue

        df_dom = df_fact.loc[idxs]
        v_dom, e_dom = run_expert_for_domain(
            df=df_dom,
            domain_name=dom,
            base_model_id=base_model_id,
            expert_root=expert_root,
            text_col=text_col,
        )

        df_fact.loc[idxs, "verdict_pred"] = v_dom
        df_fact.loc[idxs, "explanation_pred"] = e_dom

    # --- Metriken nur auf fact_checkable ---
    print("\nBeispiel-Zeilen mit Predictions (fact_checkable):")
    cols_show = [text_col, domain_col, "domain_pred", label_col, "verdict_pred", "checkability_category"]
    print(df_fact[[c for c in cols_show if c in df_fact.columns]].head())

    # Routing-Accuracy
    if domain_col in df_fact.columns:
        mask_dom_eval = (
            df_fact[domain_col].isin(super_labels)
            & df_fact["domain_pred"].isin(super_labels)
        )
        df_dom_eval = df_fact[mask_dom_eval].copy()

        if not df_dom_eval.empty:
            true_dom = df_dom_eval[domain_col]
            pred_dom = df_dom_eval["domain_pred"]
            dom_acc = accuracy_score(true_dom, pred_dom)
            print(f"\n🌍 Domain-Routing Accuracy (fact_checkable + gültige Labels): {dom_acc:.3f}")
            print("Domain-Routing Confusion Matrix (rows=true, cols=pred):")
            print(confusion_matrix(true_dom, pred_dom, labels=super_labels))
            print("Domain label order:", super_labels)
        else:
            print("\n⚠️ Keine gültigen Domain-Labels für Routing-Accuracy vorhanden.")

    # Verdict-Accuracy
    y_true = df_fact[label_col].map(norm_bool_label)
    y_pred = df_fact["verdict_pred"].map(norm_bool_label)

    verdict_acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Verdict Accuracy (gesamt, fact_checkable): {verdict_acc:.3f}\n")

    print("Classification report (True/False, fact_checkable):")
    print(classification_report(y_true, y_pred, labels=["True", "False"], zero_division=0))

    print("Confusion matrix [rows=true, cols=pred] (True, False, fact_checkable):")
    print(confusion_matrix(y_true, y_pred, labels=["True", "False"]))

    # per Domain
    if domain_col in df_fact.columns:
        print("\n===== Per-Domain Verdict-Accuracy (nach true super_domain, fact_checkable) =====")
        for dom in super_labels:
            df_dom_true = df_fact[df_fact[domain_col] == dom]
            if df_dom_true.empty:
                continue
            yt = df_dom_true[label_col].map(norm_bool_label)
            yp = df_dom_true["verdict_pred"].map(norm_bool_label)
            acc_dom = accuracy_score(yt, yp)
            print(f"{dom:22s} n={len(df_dom_true):4d} acc={acc_dom:.3f}")

    # Vorhersagen zurück in kompletten df schreiben
    for col in ["domain_pred_router", "domain_pred", "verdict_pred", "explanation_pred"]:
        if col in df_fact.columns:
            df.loc[df_fact.index, col] = df_fact[col]

    # speichern
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\n💾 Predictions gespeichert unter:\n{out_path}")

    return df
