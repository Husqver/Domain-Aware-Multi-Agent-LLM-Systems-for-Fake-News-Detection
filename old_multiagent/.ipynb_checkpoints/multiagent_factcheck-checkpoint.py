# multiagent_factcheck.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import torch
import json
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------- Default Prompts (du kannst eigene übergeben) ----------

DEFAULT_EXPERT_PROMPT = """You are a domain expert and careful fact-checker.
Task: Verify the claim strictly with objective, publicly verifiable facts. If evidence is insufficient, choose False.
Return ONLY a compact JSON object with fields:
- verdict: "True" or "False"
- explanation: short rationale (2-4 sentences, cite known facts/sources generically)

Input:
Claim: {statement}
{subject_block}
JSON:"""

DEFAULT_DECISION_PROMPT = """You are the final decision maker. You receive multiple expert votes with weights.
Aggregate their analyses and decide the final verdict.

Instructions:
- Consider expert weights explicitly.
- Prefer well-justified analyses; break ties with higher weight and stronger reasoning quality.
- Output a single JSON with fields:
  - verdict: "True" or "False"
  - explanation: 2-4 sentences explaining why this verdict follows from the experts.

Input JSON (array of expert objects):
{experts_json}

Return ONLY JSON, nothing else.
JSON:"""

# ---------- Configs ----------

@dataclass
class ExpertConfig:
    name: str
    model_name: str
    prompt_template: str = DEFAULT_EXPERT_PROMPT
    weight: float = 1.0
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    device_map: str = "auto"
    torch_dtype: Optional[str] = "float16"  # "float16" | "bfloat16" | None

@dataclass
class DecisionConfig:
    model_name: str
    prompt_template: str = DEFAULT_DECISION_PROMPT
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    device_map: str = "auto"
    torch_dtype: Optional[str] = "float16"

# ---------- Loader Cache (einfach & minimal) ----------

_MODEL_CACHE: Dict[str, Tuple[AutoTokenizer, AutoModelForCausalLM]] = {}

def _get_model(model_name: str, device_map="auto", torch_dtype: Optional[str]="float16"):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    dtype = None
    if torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype
    )
    mdl.eval()
    _MODEL_CACHE[model_name] = (tok, mdl)
    return tok, mdl

# ---------- Small helpers ----------

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}\s*$")

# --- ERSETZEN deiner bisherigen _safe_json_extract ---
def _safe_json_extract(text: str) -> Dict[str, Any]:
    """
    Nimmt den Modell-Output, kappt typische Nachträge (Code/JSON/Beispiele),
    findet das letzte balancierte JSON-Objekt und parst es robust.
    Fallback ist konservativ (verdict=False).
    """
    s = _truncate_at_markers(text.strip())

    # Balancierte-Klammern-Scan statt Regex-am-Ende
    last_json = None
    stack, start = 0, -1
    for i, ch in enumerate(s):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start != -1:
                    last_json = s[start:i+1]  # Kandidat

    if last_json is not None:
        try:
            obj = json.loads(last_json)
            verdict = "True" if str(obj.get("verdict", "")).strip() == "True" else "False"
            expl = _clean_explanation(str(obj.get("explanation", "")))
            return {"verdict": verdict, "explanation": expl}
        except Exception:
            pass

    # konservativer Fallback
    verdict = "True" if re.search(r'\b"verdict"\s*:\s*"True"\b', s) else "False"
    return {"verdict": verdict, "explanation": _clean_explanation(s)}


def _generate_json(
    tokenizer, model, prompt: str, max_new_tokens=256, temperature=0.2, top_p=0.9
) -> Dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Nur den Modell-Zusatz hinter dem Prompt extrahieren
    completion = text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
    return _safe_json_extract(completion)

# --- Zusatz-Helfer: Ausgaben säubern ---
_CODE_FENCE_RE = re.compile(r"```.*?```", re.S)

def _truncate_at_markers(s: str) -> str:
    for m in ["\n```", "```", "\nJSON:", "\nCode:", "\nExample:", "\nBeispiel:", "\nOutput:"]:
        i = s.find(m)
        if i != -1:
            return s[:i]
    return s

def _clean_explanation(s: str) -> str:
    if not s:
        return "No explanation provided."
    s = _CODE_FENCE_RE.sub("", s)                 # ganze ```…```-Blöcke raus
    s = re.sub(r"`{1,3}", "", s)                  # einzelne Backticks raus
    s = re.sub(r"(?:^|\s)(JSON:|Note:|Hinweis:|Beispiel:).*", "", s, flags=re.I|re.S)
    s = re.sub(r"\s+", " ", s).strip()
    # auf max. 4 Sätze kürzen
    parts = re.split(r'(?<=[.!?])\s+', s)
    s = " ".join(parts[:4]).strip()
    return s or "No explanation provided."


# ---------- Public API ----------

def classify_claim_multiagent(
    statement: str,
    subject: Optional[str],
    experts: List[ExpertConfig],
    decision: DecisionConfig,
    return_intermediates: bool = False,
) -> Dict[str, Any]:
    """
    Führt den 2-Schicht-Workflow aus:
      1) Reasoning Layer: mehrere Expert:innen erzeugen je ein JSON mit {verdict, explanation}
      2) Decision Layer: ein Modell aggregiert alle Expert-JSONs + Gewichte und gibt finale {verdict, explanation}
    """
    # --- Reasoning Layer ---
    expert_outputs: List[Dict[str, Any]] = []
    subject_block = f"Subject(s): {subject}" if subject else "Subject(s): (none/provided)"
    for cfg in experts:
        tok, mdl = _get_model(cfg.model_name, cfg.device_map, cfg.torch_dtype)
        prompt = cfg.prompt_template.format(
            statement=statement,
            subject_block=subject_block
        )
        result = _generate_json(
            tok, mdl, prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p
        )
        verdict = str(result.get("verdict", "False")).strip()
        explanation = str(result.get("explanation", "")).strip()
        expert_outputs.append({
            "name": cfg.name,
            "weight": float(cfg.weight),
            "verdict": "True" if verdict == "True" else "False",
            "explanation": explanation
        })

    # --- Decision Layer ---
    tok_d, mdl_d = _get_model(decision.model_name, decision.device_map, decision.torch_dtype)
    experts_json = json.dumps(expert_outputs, ensure_ascii=False, indent=2)
    dec_prompt = decision.prompt_template.format(experts_json=experts_json)
    final_json = _generate_json(
        tok_d, mdl_d, dec_prompt,
        max_new_tokens=decision.max_new_tokens,
        temperature=decision.temperature,
        top_p=decision.top_p
    )

    # Normalize final verdict field
    final_verdict = "True" if str(final_json.get("verdict", "False")).strip() == "True" else "False"
    final_expl = _clean_explanation(str(final_json.get("explanation", "")).strip())

    result = {
        "statement": statement,
        "subject": subject,
        "final_verdict": final_verdict,
        "final_explanation": final_expl,
    }
    if return_intermediates:
        result["experts"] = expert_outputs
        result["decision_raw"] = final_json
    return result

# ---------- Optional: simple weighted fallback (ohne Decision-LLM) ----------

def weighted_vote_fallback(expert_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Nützlich, falls du mal ohne Decision-LLM aggregieren willst.
    """
    score = 0.0
    for e in expert_outputs:
        score += e.get("weight", 1.0) * (1 if e.get("verdict") == "True" else -1)
    verdict = "True" if score > 0 else "False"
    explanation = f"Weighted vote score={score:.2f} from {len(expert_outputs)} experts."
    return {"verdict": verdict, "explanation": explanation}



import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def run_multiagent_on_csv(
    csv_path: str,
    experts: list,
    decision: DecisionConfig,
    statement_col: str = "statement",
    subject_col: str = "subjects",   # falls nicht vorhanden, wird None genutzt
    label_col: str = "label",        # Ground Truth (für Metriken)
    limit: int | None = None,
    save_path: str | None = None,
    show_progress: bool = True,
    return_intermediates: bool = False,
):
    """
    Liest eine CSV, ruft für jede Zeile classify_claim_multiagent auf,
    speichert Prediction + Erklärung und berechnet Metriken (falls label_col vorhanden).
    """
    df = pd.read_csv(csv_path,
                     sep="\t",          # Tab als Trenner
                     quotechar='"',     # Äußere Anführungszeichen entfernen
                     engine="python",   # flexiblerer Parser
                     dtype=str
                    )
    if limit:
        df = df.head(limit).copy()

    preds, expls, expert_dumps = [], [], []

    it = tqdm(df.iterrows(), total=len(df), desc="🔎 Fact-checking") if show_progress else df.iterrows()
    for _, row in it:
        statement = row[statement_col]
        subject = row[subject_col] if subject_col in df.columns else None

        res = classify_claim_multiagent(
            statement=statement,
            subject=subject,
            experts=experts,
            decision=decision,
            return_intermediates=return_intermediates
        )
        preds.append(res["final_verdict"])
        expls.append(res["final_explanation"])
        if return_intermediates:
            expert_dumps.append(res["experts"])
        else:
            expert_dumps.append(None)

    # Ergebnisse anhängen
    df["prediction"] = preds
    df["explanation"] = expls
    if return_intermediates:
        df["experts"] = expert_dumps

    # Falls Ground-Truth vorhanden → Metriken
    metrics = {}
    if label_col in df.columns:
        y_true = df[label_col]
        y_pred = df["prediction"]

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_true": precision_score(y_true, y_pred, pos_label="True"),
            "recall_true": recall_score(y_true, y_pred, pos_label="True"),
            "f1_true": f1_score(y_true, y_pred, pos_label="True"),
            "report": classification_report(y_true, y_pred, labels=["True","False"], target_names=["True","False"],
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["True", "False"])
)
        }

        print("\n📊 Evaluation")
        for k, v in metrics.items():
            if k != "report":
                print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
        print("\nClassification report:\n", metrics["report"])

    # Optional speichern
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"💾 Ergebnisse gespeichert unter: {save_path}")

    return df, metrics