# multiagent_factcheck.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import torch
import json
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------- GPU -----------------
os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4, 5, 6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Prompts -----------------
DEFAULT_EXPERT_PROMPT = """You are a domain expert and careful fact-checker.
Task: Verify the claim strictly with objective, publicly verifiable facts. If evidence is insufficient, choose False.

Return ONLY a compact JSON with:
- verdict: "True" or "False"
- explanation: 30–90 words, specific, with 1–2 generic source mentions in parentheses (e.g., (GAO 2018; CRS)), no placeholders, no meta-instructions.

Rules:
- Do NOT write things like "2-4 sentences", "short rationale", "No explanation provided", "TBD", or template text.
- The explanation must stand on its own and reference concrete facts (dates, actors, scope). If evidence is missing, say why.

Input:
Claim: {statement}
{subject_block}
JSON:"""

# Strenger, anti-Echo, eindeutige Tags und Kopierverbot
DEFAULT_DECISION_PROMPT = """You are the final decision maker. You receive multiple expert votes with weights.
Aggregate their analyses and decide the final verdict.

OUTPUT REQUIREMENTS (read carefully):
- Output MUST be EXACTLY one JSON object with keys "verdict" and "explanation".
- DO NOT repeat or quote the input JSON. DO NOT add prose, markdown, code fences, or extra keys.
- DO NOT copy any sentence from INPUT. Use your own words.
- "explanation": 40–100 words, summarize the strongest concrete, checkable reasons (dates, actors, scope).

Tie-breaking & policy:
- Consider expert weights explicitly.
- Prefer well-justified analyses; break ties with higher weight and stronger reasoning quality.
- If most experts give no reasons, default to "False" because insufficient evidence → False, and explain that lack of evidence is the reason.

INPUT:
<EXPERTS_JSON>
{experts_json}
</EXPERTS_JSON>

Return your answer between these exact tags, on a single line JSON:
BEGIN_DECISION_JSON
{"verdict": "True" or "False", "explanation": "<your own 40–100 word synthesis>"}
END_DECISION_JSON
"""

# ----------------- Configs -----------------
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
    torch_dtype: Optional[str] = "float16"

@dataclass
class DecisionConfig:
    model_name: str
    prompt_template: str = DEFAULT_DECISION_PROMPT
    max_new_tokens: int = 256
    temperature: float = 0.0    # deterministischer, weniger Echo
    top_p: float = 1.0
    device_map: str = "auto"
    torch_dtype: Optional[str] = "float16"

# ----------------- Loader Cache -----------------
_MODEL_CACHE: Dict[str, Tuple[AutoTokenizer, AutoModelForCausalLM]] = {}

def _get_model(model_name: str, device_map="auto", torch_dtype: Optional[str] = "float16"):
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
    # --- WICHTIG: device_map NICHT 'auto' benutzen, um Offload-Hooks zu vermeiden ---
    if device_map in (None, "none"):
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            mdl.to(device)
    elif device_map in ("cuda", "cuda:0"):
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map={"": 0},
        )
    elif isinstance(device_map, dict):
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
    mdl.eval()
    _MODEL_CACHE[model_name] = (tok, mdl)
    return tok, mdl

# ----------------- Helpers -----------------
_CODE_FENCE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\s*\n([\s\S]*?)```", re.M)
_CODE_FENCE_INLINE_RE = re.compile(r"`([^`]+)`")

PLACEHOLDER_PATTERNS = [
    r"\b2-4\b.*\bsentences?\b",
    r"\bshort rationale\b",
    r"\bno explanation provided\.?\b",
    r"\bnone\b$",
    r"\btbd\b",
    r"\bput (an )?explanation here\b",
    r"\bexplain.*here\b",
]
_ph_re = re.compile("|".join(PLACEHOLDER_PATTERNS), re.I)

def _is_placeholder_expl(s: str) -> bool:
    if not s or len(s.strip()) < 15:
        return True
    return bool(_ph_re.search(s))

def _unwrap_code_fences(s: str) -> str:
    def _block_sub(m):
        return m.group(2).strip()
    s = _CODE_FENCE_BLOCK_RE.sub(_block_sub, s)
    s = _CODE_FENCE_INLINE_RE.sub(r"\1", s)
    return s

def _maybe_parse_json_string(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    txt = s.strip()
    if (txt.startswith("{") and txt.endswith("}")) or (txt.startswith("[") and txt.endswith("]")):
        try:
            return json.loads(txt)
        except Exception:
            return None
    return None

def _clean_explanation(s: str, max_words: int = 120) -> str:
    if not s:
        return "No explanation provided."
    s = _unwrap_code_fences(s)
    maybe = _maybe_parse_json_string(s)
    if isinstance(maybe, dict):
        for k in ("explanation", "rationale", "reason", "why", "text"):
            if k in maybe and isinstance(maybe[k], str) and maybe[k].strip():
                s = maybe[k]
                break
    # JSON-Fragmente raus
    s = re.sub(r"\{[^{}]*\}", " ", s)
    # Präfixe/Marker raus
    s = re.sub(r"\b(JSON:|Note:|Hinweis:|Beispiel:|Example:)\s*", "", s, flags=re.I)
    s = re.sub(r"^#{1,6}\s+", "", s, flags=re.M)
    s = re.sub(r"^\s*[-*]\s+", "", s, flags=re.M)
    # Whitespace normalisieren
    s = re.sub(r"\s+", " ", s).strip()
    if _is_placeholder_expl(s):
        return "No explanation provided."
    # Wortlimit
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words]).rstrip(",.;:") + "..."
    # Satzlimit als Sicherheitsnetz
    parts = re.split(r'(?<=[.!?])\s+', s)
    if len(parts) > 4:
        s = " ".join(parts[:4]).strip()
    return s or "No explanation provided."

def _extract_between_tags(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    if not text:
        return None
    s = text
    start_idx = s.find(start_tag)
    end_idx = s.find(end_tag, start_idx + len(start_tag)) if start_idx != -1 else -1
    if start_idx != -1 and end_idx != -1:
        return s[start_idx + len(start_tag): end_idx].strip()
    return None

def _collect_top_level_json_objects(s: str) -> List[str]:
    objs = []
    depth = 0
    in_str = False
    esc = False
    start = -1
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        objs.append(s[start:i+1])
                        start = -1
    return objs

def _safe_json_extract(text: str, input_context: str | None = None) -> Dict[str, Any]:
    """
    Robust gegen Echo:
    - nimmt bevorzugt das JSON direkt zwischen BEGIN/END_DECISION_JSON
    - ansonsten: erstes JSON NACH BEGIN-Tag
    - ansonsten: kleinstes JSON mit 'verdict' & 'explanation'
    - verwirft Kandidaten, deren explanation im input_context (experts_json) als Substring vorkommt
    - verwirft reine Experten-Objekte (name/weight)
    """
    raw = _unwrap_code_fences(text or "")

    # 1) Bereich zwischen Tags
    between = _extract_between_tags(raw, "BEGIN_DECISION_JSON", "END_DECISION_JSON")
    search_space = between if between is not None else raw

    # 2) JSONs im relevanten Bereich sammeln
    objs = _collect_top_level_json_objects(search_space)

    def is_from_input(expl: str) -> bool:
        if not input_context or not expl:
            return False
        # einfache Substring-Prüfung, robust gegen Kleinschreibung
        return expl.strip() and expl.lower() in input_context.lower()

    candidates = []
    for o in objs:
        try:
            j = json.loads(o)
            if not isinstance(j, dict):
                continue
            if "verdict" not in j:
                continue
            # nur True/False zulassen
            v = str(j.get("verdict", "")).strip()
            if v not in ("True", "False"):
                continue
            # Experten-Objekte verwerfen
            if any(k in j for k in ("name", "weight")) and set(j.keys()).issubset({"name", "weight", "verdict", "explanation"}):
                continue
            expl = j.get("explanation", "")
            # Erklärung darf nicht 1:1 aus Input stammen
            if isinstance(expl, str) and is_from_input(expl):
                continue
            candidates.append(j)
        except Exception:
            continue

    # 3) Scoring: bevorzugt JSON, das BEIDES hat und klein ist
    def score(obj):
        has_expl = 1 if ("explanation" in obj and isinstance(obj["explanation"], str) and obj["explanation"].strip()) else 0
        extra_keys = len(set(obj.keys()) - {"verdict", "explanation"})
        size = len(json.dumps(obj, ensure_ascii=False))
        return (-has_expl, extra_keys, size)

    chosen = None
    if candidates:
        candidates.sort(key=score)
        chosen = candidates[0]

    # 4) Vorsichtiger Fallback: Regex auf kleinstes Objekt mit verdict + explanation
    if not chosen:
        m = re.search(r'\{[^{}]*"verdict"\s*:\s*"(True|False)"[^{}]*"explanation"\s*:\s*"(?:[^"\\]|\\.)*"\s*[^{}]*\}', search_space)
        if m:
            try:
                j = json.loads(m.group(0))
                if not is_from_input(j.get("explanation", "")):
                    chosen = j
            except Exception:
                pass

    # 5) Letzter Fallback: neutrales Ergebnis mit gesäuberter Completion
    if not chosen:
        # Keine harte Heuristik auf irgendein "True" im Text – default: False
        return {"verdict": "False", "explanation": _clean_explanation(search_space)}

    verdict = "True" if str(chosen.get("verdict", "")).strip() == "True" else "False"
    explanation = _clean_explanation(str(chosen.get("explanation", "")).strip())

    return {"verdict": verdict, "explanation": explanation}

def _generate_json(tokenizer, model, prompt: str,
                   max_new_tokens=256, temperature=0.2, top_p=0.9,
                   input_context: str | None = None) -> Dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature and temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Modell-Completion sicher extrahieren
    prompt_decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    completion = text[len(prompt_decoded):]
    return _safe_json_extract(completion, input_context=input_context)

# ----------------- Explanation Retry / Rewrite -----------------
def _regenerate_explanation(tokenizer, model, claim: str, subject_block: str, verdict: str,
                            max_new_tokens=120, temperature=0.2, top_p=0.9) -> str:
    prompt = f"""You already decided the verdict for this claim.

Claim: {claim}
{subject_block}

Verdict: "{verdict}"

Now output ONLY JSON with:
- explanation: 40–90 words, concrete facts, 1–2 generic source mentions (e.g., (CBO 2017; DHS report)), no placeholders. Do not copy any sentence from earlier text.

JSON:"""
    out = _generate_json(tokenizer, model, prompt,
                         max_new_tokens=max_new_tokens,
                         temperature=temperature, top_p=top_p)
    expl = str(out.get("explanation", "")).strip()
    return expl

def _rewrite_decision_expl_if_copied(tokenizer, model, verdict: str, experts: List[Dict[str, Any]],
                                     claim: str, subject_block: str) -> str:
    """Falls die Decision-Explanation aus dem Input kopiert wurde, neu in eigenen Worten erzeugen."""
    # Mini-Zusammenfassung der stärksten Gründe für das Rewrite
    # (keine Zitate der Experten-Sätze, nur verdichtete Stichworte)
    reasons = []
    # nimm max 3, gewichtet
    experts_sorted = sorted(experts, key=lambda e: e.get("weight", 1.0), reverse=True)
    for e in experts_sorted[:3]:
        reasons.append(f"{e['name']} (w={e['weight']}): {e['verdict']}")
    skeleton = "; ".join(reasons)
    prompt = f"""Rewrite a decision explanation in your own words. Do NOT copy any sentence from input.

Input summary:
- Verdict: {verdict}
- Expert votes: {skeleton}
- Claim: {claim}
- {subject_block}

Output ONLY JSON: {{"explanation": "40–90 words, concrete, objective, with 1–2 generic source mentions (e.g., (GAO 2018; CRS))"}}"""
    out = _generate_json(tokenizer, model, prompt, max_new_tokens=140, temperature=0.2, top_p=0.9)
    return _clean_explanation(out.get("explanation", ""))

# ----------------- Public API -----------------
def classify_claim_multiagent(
    statement: str,
    subject: Optional[str],
    experts: List[ExpertConfig],
    decision: DecisionConfig,
    return_intermediates: bool = False,
) -> Dict[str, Any]:
    expert_outputs: List[Dict[str, Any]] = []
    subject_block = f"Subject(s): {subject}" if subject else "Subject(s): (none/provided)"
    for cfg in experts:
        tok, mdl = _get_model(cfg.model_name, cfg.device_map, cfg.torch_dtype)
        prompt = cfg.prompt_template.format(statement=statement, subject_block=subject_block)
        result = _generate_json(tok, mdl, prompt,
                                max_new_tokens=cfg.max_new_tokens,
                                temperature=cfg.temperature,
                                top_p=cfg.top_p)
        verdict = "True" if str(result.get("verdict", "False")).strip() == "True" else "False"
        explanation = _clean_explanation(str(result.get("explanation", "")).strip())
        # Retry, falls Platzhalter
        if _is_placeholder_expl(explanation):
            retry_expl = _regenerate_explanation(tok, mdl, statement, subject_block, verdict,
                                                 max_new_tokens=min(160, cfg.max_new_tokens+40),
                                                 temperature=max(0.2, cfg.temperature),
                                                 top_p=cfg.top_p)
            cleaned_retry = _clean_explanation(retry_expl)
            if not _is_placeholder_expl(cleaned_retry):
                explanation = cleaned_retry
        expert_outputs.append({
            "name": cfg.name,
            "weight": float(cfg.weight),
            "verdict": verdict,
            "explanation": explanation
        })

    # Decision Layer
    all_empty = all(_is_placeholder_expl(e["explanation"]) for e in expert_outputs)
    if all_empty:
        for e in expert_outputs:
            e["explanation"] = "Insufficient factual justification given by this expert."

    tok_d, mdl_d = _get_model(decision.model_name, decision.device_map, decision.torch_dtype)

    # Rauschen für Decision-Input reduzieren (kürzere Erklärungen)
    def _trim_expl(e: Dict[str, Any], max_words: int = 60) -> str:
        return _clean_explanation(e.get("explanation", ""), max_words=max_words)

    compact_experts = [
        {
            "name": e["name"],
            "weight": e["weight"],
            "verdict": e["verdict"],
            "explanation": _trim_expl(e, max_words=60)
        }
        for e in expert_outputs
    ]

    experts_json = json.dumps(compact_experts, ensure_ascii=False, indent=2)
    dec_prompt = decision.prompt_template.format(experts_json=experts_json)

    # WICHTIG: input_context=experts_json an Parser durchreichen, damit kopierte Sätze verworfen werden
    final_json = _generate_json(tok_d, mdl_d, dec_prompt,
                                max_new_tokens=decision.max_new_tokens,
                                temperature=decision.temperature,
                                top_p=decision.top_p,
                                input_context=experts_json)

    final_verdict = "True" if str(final_json.get("verdict", "False")).strip() == "True" else "False"
    final_expl = _clean_explanation(str(final_json.get("explanation", "")).strip())

    # Falls Explanation dennoch aus dem Input kopiert wurde → Rewriter
    if final_expl and experts_json and final_expl.lower() in experts_json.lower():
        rewritten = _rewrite_decision_expl_if_copied(tok_d, mdl_d, final_verdict, compact_experts, statement, subject_block)
        if rewritten and not _is_placeholder_expl(rewritten):
            final_expl = rewritten

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

# ----------------- CSV Runner -----------------
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def run_multiagent_on_csv(
    csv_path: str,
    experts: list,
    decision: DecisionConfig,
    statement_col: str = "statement",
    subject_col: str = "subjects",
    label_col: str = "label",
    limit: int | None = None,
    save_path: str | None = None,
    show_progress: bool = True,
    return_intermediates: bool = False,
):
    df = pd.read_csv(csv_path, sep="\t", quotechar='"', engine="python", dtype=str)
    if limit:
        df = df.head(limit).copy()

    preds, expls, expert_dumps = [], [], []
    it = tqdm(df.iterrows(), total=len(df), desc="🔎 Fact-checking") if show_progress else df.iterrows()
    for _, row in it:
        statement = row[statement_col]
        subject = row[subject_col] if subject_col in df.columns else None
        res = classify_claim_multiagent(statement=statement, subject=subject,
                                        experts=experts, decision=decision,
                                        return_intermediates=return_intermediates)
        preds.append(res["final_verdict"])
        expls.append(res["final_explanation"])
        expert_dumps.append(res.get("experts") if return_intermediates else None)

    df["prediction"] = preds
    df["explanation"] = expls
    if return_intermediates:
        df["experts"] = expert_dumps

    metrics = {}
    if label_col in df.columns:
        y_true = df[label_col]
        y_pred = df["prediction"]
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_true": precision_score(y_true, y_pred, pos_label="True"),
            "recall_true": recall_score(y_true, y_pred, pos_label="True"),
            "f1_true": f1_score(y_true, y_pred, pos_label="True"),
            "report": classification_report(y_true, y_pred, labels=["True","False"]),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["True", "False"])
        }
        print("\n📊 Evaluation")
        for k, v in metrics.items():
            if k != "report":
                print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
        print("\nClassification report:\n", metrics["report"])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"💾 Ergebnisse gespeichert unter: {save_path}")

    return df, metrics
