# expert_eval.py

import os
from typing import List, Dict, Any, Tuple, Union

import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
from peft import PeftModel

from langchain_experimental.pydantic_v1 import BaseModel, Field
from langchain_experimental.llms import LMFormatEnforcer

# === Environment-Settings (optional wie bei dir) ===
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # sichtbare GPUs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("expert_eval :: using device:", device)


# ===== 1) Kleines Schema: nur Verdict =====

class VerdictOnly(BaseModel):
    verdict: str = Field(..., description="Must be either 'True' or 'False'")

VERDICT_SCHEMA = VerdictOnly.schema()


def norm_bool_label(x: str) -> str:
    if not isinstance(x, str):
        return "False"
    s = x.strip().strip(".,:;!?)(").lower()
    if s.startswith("true"):
        return "True"
    if s.startswith("false"):
        return "False"
    return "False"


# ===== 2) Expert-Systemprompt =====

def build_expert_system_prompt(domain_name: str) -> str:
    return f"""You are a fact-checking expert specialized in the '{domain_name}' domain.

You receive a short political or public policy claim (statement) and must decide if it is factually correct.

You must output a JSON object that follows this schema:

{VERDICT_SCHEMA}

The field "verdict" MUST be either "True" or "False".

Reply with JSON only. No markdown, no extra text.
"""


# ===== 3) Laden eines Domain-Experten (Base + LoRA + Tokenizer) =====

def load_expert_model(
    domain_name: str,
    base_model_id: str,
    expert_root: str,
):
    """
    Lädt Base-Modell, LoRA-Adapter + Tokenizer für eine bestimmte Domain.
    """
    expert_dir = os.path.join(expert_root, domain_name)
    if not os.path.isdir(expert_dir):
        raise FileNotFoundError(f"Expert directory not found for domain '{domain_name}': {expert_dir}")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, expert_dir)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(expert_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


# ===== 4) Ein Claim -> Verdict (True/False) mit LMFormatEnforcer =====

def predict_with_expert_lmfe(
    statement: str,
    domain_name: str,
    tokenizer,
    enforced_llm: LMFormatEnforcer,
) -> str:
    """
    Nutzt LMFormatEnforcer, um ein JSON mit 'verdict' zu erzeugen und normalisiert das auf 'True'/'False'.
    """
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
        raw = enforced_llm.invoke(prompt_text, max_new_tokens=64)
        # LMFormatEnforcer sollte hier schon ein Dict liefern (json_schema)
        if isinstance(raw, dict):
            verdict = raw.get("verdict", "")
        else:
            # Fallback: versuchen, wenn doch String kommt
            import json as _json
            if isinstance(raw, str):
                obj = _json.loads(raw)
                verdict = obj.get("verdict", "")
            else:
                verdict = ""
    except Exception as e:
        print(f"[WARN] LMFormatEnforcer invoke/parsing error for domain '{domain_name}': {e}")
        verdict = ""

    return norm_bool_label(verdict)


# ===== 5) Hauptfunktion: alle Experten testen =====

def evaluate_experts_with_lmfe(
    super_labels: List[str],
    base_model_id: str,
    expert_root: str,
    test_paths: Union[str, List[str]],
    text_col: str = "statement",
    label_col: str = "label",
    domain_col: str = "super_domain",
    sep: str = "\t",
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Evaluiert NUR die Experten-Modelle (ohne Router), d.h. es wird für jede Domain
    das passende LoRA-Modell geladen und auf alle Testbeispiele dieser Domain angewendet.

    Parameter:
      - super_labels: alle Domains, für die Experten existieren (z.B. 5 oder 8 Klassen)
      - base_model_id: Basismodell-ID (z.B. 'meta-llama/Llama-3.1-8B-Instruct')
      - expert_root: Pfad, unter dem die Domain-Unterordner liegen
      - test_paths: Pfad oder Liste von Pfaden zu TSV/CSV-Dateien
      - text_col: Spalte mit Claim
      - label_col: Spalte mit True/False Label
      - domain_col: Spalte mit Superdomain (true Labels)
      - sep: Separator (Standard: '\t')

    Rückgabe:
      - df_all_pred: DataFrame mit zusätzlicher Spalte 'pred'
      - all_results: Liste von Dicts mit pro-Domain-Accuracy
    """

    # --- Testdaten laden (eine oder mehrere Dateien) ---
    if isinstance(test_paths, str):
        test_paths = [test_paths]

    dfs = []
    for path in test_paths:
        df_part = pd.read_csv(
            path,
            sep=sep,
            quotechar='"',
            engine="python",
            dtype=str,
        )
        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True)

    # Aufräumen
    df = df.dropna(subset=[text_col, label_col, domain_col])
    df = df[df[text_col].str.strip() != ""]
    df = df[df[label_col].isin(["True", "False"])]

    print("Testset size:", len(df))
    print("Super-Domain-Verteilung (Test):")
    print(df[domain_col].value_counts())

    all_results: List[Dict[str, Any]] = []
    all_rows_with_pred: List[pd.DataFrame] = []

    # --- Pro Domain: Modell laden, Pipeline + LMFE bauen, alle Claims predicten ---
    for domain in super_labels:
        df_dom = df[df[domain_col] == domain].copy()
        if df_dom.empty:
            print(f"\n⚠️ Domain '{domain}': keine Testbeispiele – wird übersprungen.")
            continue

        print(f"\n===== Evaluating expert for domain: {domain} =====")
        print(f"# Testbeispiele: {len(df_dom)}")
        print("Label-Verteilung (true):")
        print(df_dom[label_col].value_counts())

        # Modell + Tokenizer laden
        model, tokenizer = load_expert_model(
            domain_name=domain,
            base_model_id=base_model_id,
            expert_root=expert_root,
        )

        # HF-Pipeline + LMFormatEnforcer (einmal pro Domain)
        gen_pipe = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            do_sample=False,
            return_full_text=False,
        )
        enforced_llm = LMFormatEnforcer(
            pipeline=gen_pipe,
            json_schema=VERDICT_SCHEMA,
        )

        preds = []
        for _, row in tqdm(
            df_dom.iterrows(),
            total=len(df_dom),
            desc=f"🧠 {domain}",
            dynamic_ncols=True,
        ):
            statement = row[text_col]
            pred = predict_with_expert_lmfe(
                statement=statement,
                domain_name=domain,
                tokenizer=tokenizer,
                enforced_llm=enforced_llm,
            )
            preds.append(pred)

        df_dom["pred"] = preds
        all_rows_with_pred.append(df_dom)

        y_true = df_dom[label_col].map(norm_bool_label)
        y_pred = df_dom["pred"].map(norm_bool_label)

        acc = accuracy_score(y_true, y_pred)
        print(f"\nAccuracy ({domain}): {acc:.3f}")

        report = classification_report(
            y_true, y_pred,
            labels=["True", "False"],
            zero_division=0,
        )
        print("Classification report:")
        print(report)

        cm = confusion_matrix(y_true, y_pred, labels=["True", "False"])
        print("Confusion matrix [rows=true, cols=pred] (True, False):")
        print(cm)

        all_results.append({
            "domain": domain,
            "n": int(len(df_dom)),
            "accuracy": float(acc),
        })

        # Speicher freigeben
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_rows_with_pred:
        print("\n⚠️ Keine Predictions erzeugt – stimmen die Superlabels mit der Spalte "
              f"'{domain_col}' in den Testdaten überein?")
        return pd.DataFrame(), all_results

    df_all_pred = pd.concat(all_rows_with_pred, ignore_index=True)

    # --- Gesamtmetriken ---
    print("\n===== Gesamt-Performance über alle Domains =====")
    y_true_all = df_all_pred[label_col].map(norm_bool_label)
    y_pred_all = df_all_pred["pred"].map(norm_bool_label)

    acc_all = accuracy_score(y_true_all, y_pred_all)
    print(f"Overall accuracy: {acc_all:.3f}")

    print("Overall classification report:")
    print(classification_report(y_true_all, y_pred_all, labels=["True", "False"], zero_division=0))

    print("Overall confusion matrix (True, False):")
    print(confusion_matrix(y_true_all, y_pred_all, labels=["True", "False"]))

    print("\nPer-domain accuracy summary:")
    for r in all_results:
        print(f"{r['domain']:25s}  n={r['n']:4d}  acc={r['accuracy']:.3f}")

    return df_all_pred, all_results
