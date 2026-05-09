# multiagent_factcheck.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import torch
import json
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
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
    Schneidet nichts Hartes weg: entpackt Codefences, findet letztes balanciertes JSON,
    extrahiert 'explanation' auch aus verschachtelten Strings/Objekten
    und fällt sonst auf Kontext zurück.
    """
    # Unwrap first, damit JSON in Codefences erkannt wird
    raw = _unwrap_code_fences(text or "")
    s = raw.strip()

    # Balancierte-Klammern-Suche
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
                    last_json = s[start:i+1]

    if last_json is not None:
        try:
            obj = json.loads(last_json)

            # Verdict robust normalisieren
            verdict = "True" if str(obj.get("verdict", "")).strip() == "True" else "False"

            # Explanation robust extrahieren (auch verschachtelt)
            explanation = obj.get("explanation", "")
            if isinstance(explanation, dict):
                # ggf. nested explanation/rationale
                for k in ("explanation", "rationale", "reason", "why", "text"):
                    if k in explanation and isinstance(explanation[k], str) and explanation[k].strip():
                        explanation = explanation[k]
                        break
                else:
                    explanation = json.dumps(explanation, ensure_ascii=False)
            elif isinstance(explanation, list):
                explanation = " ".join(str(x) for x in explanation if isinstance(x, (str, int, float)))
            elif not isinstance(explanation, str):
                explanation = str(explanation)

            explanation = explanation.strip()

            # Falls leer, versuche aus dem restlichen JSON/Umfeld zu gewinnen
            if not explanation:
                # 1) Vielleicht steckt sie als JSON-String drin
                inner = _maybe_parse_json_string(obj.get("explanation", ""))
                if isinstance(inner, dict):
                    for k in ("explanation", "rationale", "reason", "why", "text"):
                        if k in inner and isinstance(inner[k], str) and inner[k].strip():
                            explanation = inner[k].strip()
                            break

            if not explanation:
                # 2) Kontext-Fallback (Text vor/nach dem JSON)
                explanation = _extract_contextual_explanation(s, last_json)

            return {
                "verdict": verdict,
                "explanation": _clean_explanation(explanation)
            }
        except Exception:
            pass

    # Kein parsebares JSON → heuristischer Fallback
    verdict = "True" if re.search(r'\b"verdict"\s*:\s*"True"\b', s) else "False"
    explanation = _extract_contextual_explanation(s, None)
    return {"verdict": verdict, "explanation": _clean_explanation(explanation)}


# --- Zusatz-Helfer: Ausgaben säubern ---

_CODE_FENCE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\s*\n([\s\S]*?)```", re.M)
_CODE_FENCE_INLINE_RE = re.compile(r"`([^`]+)`")

def _unwrap_code_fences(s: str) -> str:
    # Ersetze ```lang\n...``` durch den reinen Inhalt (keine harte Löschung mehr)
    def _block_sub(m):
        return m.group(2).strip()
    s = _CODE_FENCE_BLOCK_RE.sub(_block_sub, s)
    # Inline-Backticks entfernen, Inhalt behalten
    s = _CODE_FENCE_INLINE_RE.sub(r"\1", s)
    return s

def _maybe_parse_json_string(s: str) -> Optional[Dict[str, Any]]:
    """Versucht, einen String, der wie JSON aussieht, zu parsen."""
    if not s:
        return None
    txt = s.strip()
    if (txt.startswith("{") and txt.endswith("}")) or (txt.startswith("[") and txt.endswith("]")):
        try:
            return json.loads(txt)
        except Exception:
            return None
    return None

def _extract_contextual_explanation(raw_text: str, json_segment: Optional[str]) -> str:
    """
    Fallback: baue eine Kurzbegründung aus Kontext, wenn die Explanation leer war.
    Nimmt bevorzugt Text vor dem JSON; sonst nach dem JSON.
    """
    s = raw_text or ""
    if not s:
        return "No explanation provided."
    prefix, suffix = s, ""
    if json_segment and json_segment in s:
        parts = s.split(json_segment, 1)
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""
    cand = prefix if len(prefix.strip()) >= 40 else suffix
    return cand.strip() or "No explanation provided."


def _truncate_at_markers(s: str) -> str:
    for m in ["\n```", "```", "\nJSON:", "\nCode:", "\nExample:", "\nBeispiel:", "\nOutput:"]:
        i = s.find(m)
        if i != -1:
            return s[:i]
    return s

def _clean_explanation(s: str) -> str:
    if not s:
        return "No explanation provided."

    # 1) Codefences/Inline-Code ENTWENDEN (Inhalt behalten)
    s = _unwrap_code_fences(s)

    # 2) Verschachteltes JSON herauslösen (falls Explanation fälschlich JSON enthält)
    maybe = _maybe_parse_json_string(s)
    if isinstance(maybe, dict):
        # Präferiere typische Schlüssel
        for k in ("explanation", "rationale", "reason", "why"):
            if k in maybe and isinstance(maybe[k], str) and maybe[k].strip():
                s = maybe[k]
                break

    # 3) Offensichtliche Marker abkappen, aber nicht alles danach löschen
    #    (nur die Marker selbst entfernen)
    s = re.sub(r"\b(JSON:|Note:|Hinweis:|Beispiel:|Example:)\s*", "", s, flags=re.I)

    # 4) Überschriften/Listenmarker entschärfen (Inhalt behalten)
    s = re.sub(r"^#{1,6}\s+", "", s, flags=re.M)     # "# Heading" -> "Heading"
    s = re.sub(r"^\s*[-*]\s+", "", s, flags=re.M)    # "- item" -> "item"

    # 5) Whitespace normalisieren
    s = re.sub(r"\s+", " ", s).strip()

    # 6) Auf 2–4 Sätze begrenzen
    parts = re.split(r'(?<=[.!?])\s+', s)
    if len(parts) > 4:
        s = " ".join(parts[:4]).strip()

    return s or "No explanation provided."

def _generate_json(
    tokenizer, model, prompt: str, max_new_tokens=256, temperature=0.2, top_p=0.9
) -> Dict[str, Any]:
    """
    Schickt den Prompt an das Modell, dekodiert die Antwort
    und gibt geparstes JSON via _safe_json_extract zurück.
    """
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

    # Kompletten Output dekodieren
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    # Prompt-Teil wegschneiden, nur das Modell-Completion behalten
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    completion = text[len(prompt_text):]

    return _safe_json_extract(completion)




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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

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
            "report": classification_report(y_true, y_pred, labels=["True","False"], target_names=["True","False"]),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["True", "False"])

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






import os, json, torch, pandas as pd
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

TRUE_LABELS = {"true", "mostly-true", "half-true"}
FALSE_LABELS = {"barely-true", "false", "pants-fire", "pants-on-fire"}

def map_label_to_binary(label: str) -> str:
    l = str(label).strip().lower()
    if l in TRUE_LABELS:
        return "True"
    if l in FALSE_LABELS:
        return "False"
    return "False"

def build_prompt(prompt_template: str, statement: str, subjects: str | None, use_subjects: bool) -> str:
    subject_block = f"Subjects: {subjects}\n" if (use_subjects and subjects) else ""
    try:
        return prompt_template.format(statement=statement, subject_block=subject_block)
    except KeyError:
        # falls ein altes Template {subjects} nutzt
        return prompt_template.format(statement=statement, subjects=(subjects or ""))

def _balanced_slice_with_verdict(s: str, start_idx: int) -> str | None:
    """Nimmt Start bei '{' und liefert die kleinstmögliche balancierte JSON-Scheibe zurück, die '\"verdict\"' enthält."""
    depth = 0
    seen_verdict = False
    end_idx = None
    for i in range(start_idx, len(s)):
        c = s[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
        else:
            # grober Check auf '"verdict"' im Fenster
            if not seen_verdict and s[max(start_idx, i-10): i+10].find('"verdict"') != -1:
                seen_verdict = True
    if end_idx and seen_verdict:
        return s[start_idx:end_idx]
    return None

def extract_json_verdict(text: str) -> str:
    """
    Robust: Sucht von rechts nach '"verdict"', findet das nächstliegende '{' links davon,
    extrahiert die balancierte Klammer-Scheibe und parst sie.
    Fallback: iteriert über alle '{' und versucht dasselbe.
    """
    # 1) von rechts nahe '"verdict"'
    k = text.rfind('"verdict"')
    if k != -1:
        start = text.rfind('{', 0, k)
        if start != -1:
            chunk = _balanced_slice_with_verdict(text, start)
            if chunk:
                try:
                    obj = json.loads(chunk)
                    v = obj.get("verdict")
                    if v in ("True", "False"):
                        return v
                except Exception:
                    pass
    # 2) Fallback: alle '{' durchgehen
    for i, c in enumerate(text):
        if c == '{':
            chunk = _balanced_slice_with_verdict(text, i)
            if chunk:
                try:
                    obj = json.loads(chunk)
                    v = obj.get("verdict")
                    if v in ("True", "False"):
                        return v
                except Exception:
                    continue
    return "False"

class StopOnValidJson(StoppingCriteria):
    """Stoppt, sobald ab Prompt-Ende ein balanciertes JSON mit '\"verdict\"' erkannt wurde."""
    def __init__(self, tokenizer, input_len):
        self.tok = tokenizer
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        new_tokens = input_ids[0][self.input_len:]
        txt = self.tok.decode(new_tokens, skip_special_tokens=True)
        # schneller Vor-Check:
        if '"verdict"' not in txt or '{' not in txt or '}' not in txt:
            return False
        # prüfen, ob eine balancierte JSON-Scheibe mit "verdict" bereits vorliegt
        v = extract_json_verdict(txt)
        return v in ("True", "False")

def evaluate_model_json(
    model_name: str,
    test_data_path: str,
    prompt_template: str,
    cuda_device: str = "0",
    limit: int = 2000,
    use_subjects: bool = False,
    max_new_tokens: int = 120,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    save_csv_path: str | None = None,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(test_data_path)
    if limit and limit > 0:
        df = df.iloc[:limit].copy()

    print(f"\n📦 Lade Modell: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    preds: List[str] = []
    gens: List[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="🧾 JSON-Gen"):
        statement = row["statement"]
        subjects = row["subjects"] if (use_subjects and "subjects" in df.columns) else None
        prompt = build_prompt(prompt_template, statement, subjects, use_subjects)
        inputs = tok(prompt, return_tensors="pt").to(device)

        stop = StoppingCriteriaList([StopOnValidJson(tok, inputs["input_ids"].shape[1])])

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tok.eos_token_id,
                stopping_criteria=stop,
            )

        gen_text = tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gens.append(gen_text)
        verdict = extract_json_verdict(gen_text)
        preds.append(verdict)

    y_true = df["label"].apply(map_label_to_binary)
    y_pred = pd.Series(preds).apply(lambda v: "True" if v == "True" else "False")

    print("\n📊 Evaluation (JSON):")
    print("Confusion Matrix (True, False):")
    print(confusion_matrix(y_true, y_pred, labels=["True", "False"]))
    print(f"\nAccuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred, pos_label='True'):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred, pos_label='True'):.3f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, pos_label='True'):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["True", "False"]))

    suffix = model_name.split("/")[-1]
    df[f"label_{suffix}"] = y_pred.values
    df[f"gen_{suffix}"] = gens

    if save_csv_path:
        df.to_csv(save_csv_path, index=False, encoding="utf-8")
        print(f"💾 Ergebnisse gespeichert unter: {save_csv_path}")

    return df




import pandas as pd
import json
from pathlib import Path

TRUE_LABELS = {"true", "mostly-true", "half-true"}
FALSE_LABELS = {"barely-true", "false", "pants-fire", "pants-on-fire"}

def map_label_to_binary(label: str) -> str:
    l = str(label).strip().lower()
    if l in TRUE_LABELS:
        return "True"
    if l in FALSE_LABELS:
        return "False"
    return "False"

def make_explanation(verdict: str, subjects: str | None) -> str:
    dom = f" in {subjects}" if subjects else ""
    if verdict == "True":
        return f"Independent public records and reputable analyses support the claim{dom} within the stated timeframe and definitions."
    else:
        return f"Available public records and reliable sources do not support the claim{dom} as stated or omit key context."

def build_training_jsonl(
    csv_path: str,
    prompt_template: str,   # dein JSON-Style-Template mit "Claim: {statement}\n{subject_block}JSON:"
    output_path: str,
    use_subjects: bool = True,
):
    df = pd.read_csv(csv_path, header=0)
    out = []

    for _, row in df.iterrows():
        statement = row["statement"]
        subjects = row.get("subjects", "")
        label = row["label"]
        verdict = map_label_to_binary(label)
        subject_block = f"Subjects: {subjects}\n" if (use_subjects and subjects) else ""

        # Prompt-Teil (endet mit "JSON:")
        prompt = prompt_template.format(statement=statement, subject_block=subject_block)

        # Ziel-JSON
        explanation = make_explanation(verdict, subjects if use_subjects else None)
        target_json = json.dumps(
            {"verdict": verdict, "explanation": explanation},
            ensure_ascii=False
        )

        text = prompt + target_json
        out.append({"text": text})

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ {len(out)} Beispiele gespeichert in {output_path}")

