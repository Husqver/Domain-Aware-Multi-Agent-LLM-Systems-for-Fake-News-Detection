from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import json
import os
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging

# -> optional: nur wichtige Meldungen von Transformers zeigen
hf_logging.set_verbosity_error()

# ----------------- GPU -----------------
os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4, 5, 6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Prompts -----------------
DEFAULT_EXPERT_PROMPT = """
You are a domain expert and careful fact-checker.
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

JSON:
"""

# Strenger, anti-Echo, eindeutige Tags und Kopierverbot
DEFAULT_DECISION_PROMPT = """
You are the final decision maker. You receive multiple expert votes with weights. Aggregate their analyses and decide the final verdict.

OUTPUT REQUIREMENTS (read carefully):
- Output MUST be EXACTLY one JSON object with keys "verdict" and "explanation".
- DO NOT repeat or quote the input JSON. DO NOT add prose, markdown, code fences, or extra keys.
- DO NOT copy any sentence from INPUT. Use your own words.
- Do not include steps, numbered lists, or chain-of-thought. Output only the final justification.
- "explanation": 40–100 words, summarize the strongest concrete, checkable reasons (dates, actors, scope).

Tie-breaking & policy:
- Consider expert weights explicitly.
- Prefer well-justified analyses; break ties with higher weight and stronger reasoning quality.
- If most experts give no reasons, default to "False" because insufficient evidence → False, and explain that lack of evidence is the reason.

INPUT:
<EXPERTS_JSON>
{experts_json}
</EXPERTS_JSON>

Return your answer between these exact tags, on a single line
JSON:
BEGIN_DECISION_JSON
{{"verdict": "True" or "False", "explanation": "<your own 40–100 word synthesis>"}}
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
    temperature: float = 0.0  # deterministischer, weniger Echo
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
# Fix fehlerhafte Regexe für Codefences
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


# --- CoT-Heuristik ---
_COT_PHRASES_RE = re.compile(
    r"\b(step\s*\d+|let's|we need to|first,|second,|third,|analysis:)\b", re.I
)


def _is_chain_of_thoughty(s: str) -> bool:
    if not s:
        return False
    return bool(_COT_PHRASES_RE.search(s))


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

    s = re.sub(r"\{[^{}]*\}", " ", s)  # JSON-Fragmente raus
    s = re.sub(r"\b(JSON:|Note:|Hinweis:|Beispiel:|Example:)\s*", "", s, flags=re.I)
    s = re.sub(r"^#{1,6}\s+", "", s, flags=re.M)
    s = re.sub(r"^\s*[-*]\s+", "", s, flags=re.M)
    s = re.sub(r"\s+", " ", s).strip()

    if _is_placeholder_expl(s):
        return "No explanation provided."

    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words]).rstrip(",.;:") + "..."

    parts = re.split(r'(?<=[.!?])\s+', s)
    if len(parts) > 4:
        s = " ".join(parts[:4]).strip()

    # CoT neutralisieren
    if _is_chain_of_thoughty(s):
        s = re.sub(_COT_PHRASES_RE, "", s).strip()

    if len(s.split()) < 12:
        return "No explanation provided."

    return s or "No explanation provided."


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
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        objs.append(s[start: i + 1])
                        start = -1
    return objs


# -------- Generatoren: Parsed + Raw --------
def _generate_json_and_text(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    input_context: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    do_sample = bool(temperature and temperature > 0)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    full = tokenizer.decode(out[0], skip_special_tokens=True)
    prompt_decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    completion = full[len(prompt_decoded):]
    parsed = _safe_json_extract(completion, input_context=input_context)
    return parsed, completion


DECISION_STOP_STRINGS = ["END_DECISION_JSON"]


def _truncate_at_stop_strings(s: str, stops=DECISION_STOP_STRINGS) -> str:
    cut = len(s)
    for t in stops:
        i = s.find(t)
        if i != -1:
            cut = min(cut, i + len(t))
    return s[:cut]


def _generate_json(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
    input_context: Optional[str] = None,
) -> Dict[str, Any]:
    parsed, _ = _generate_json_and_text(
        tokenizer,
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        input_context=input_context,
    )
    return parsed


def _find_all_tag_ranges(text: str, start_tag: str, end_tag: str) -> List[Tuple[int, int, str]]:
    """Finde alle Segmente zwischen Tags. Liefert (start_index, end_index, segment)."""
    res = []
    i = 0
    while True:
        s = text.find(start_tag, i)
        if s == -1:
            break
        e = text.find(end_tag, s + len(start_tag))
        if e == -1:
            break
        seg = text[s + len(start_tag): e].strip()
        res.append((s, e + len(end_tag), seg))
        i = e + len(end_tag)
    return res


def _collect_top_level_json_objects_with_spans(s: str, base_offset: int = 0) -> List[Tuple[str, int]]:
    """Wie _collect_top_level_json_objects, aber inkl. Startposition im Ursprungstext (für Scoring)."""
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
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        objs.append((s[start: i + 1], base_offset + start))
                        start = -1
    return objs


def _safe_json_extract(text: str, input_context: Optional[str] = None) -> Dict[str, Any]:
    """
    Verbesserter Parser:
    - scannt ALLE BEGIN/END_DECISION_JSON-Segmente (nicht nur das erste)
    - zieht gültige JSONs aus den Segmenten; wenn keine, dann aus dem Gesamttext
    - bevorzugt JSONs nach dem letzten BEGIN_DECISION_JSON
    - filtert Beispiel-/Template-/CoT-/kopierte Erklärungen
    """
    raw = _unwrap_code_fences(text or "")

    # Alle Tag-Segmente + Fallback auf den gesamten Raw-Text
    ranges = _find_all_tag_ranges(raw, "BEGIN_DECISION_JSON", "END_DECISION_JSON")
    last_begin_idx = raw.rfind("BEGIN_DECISION_JSON")

    # Kandidaten sammeln: (json_dict, start_pos, raw_str)
    candidates: List[Tuple[Dict[str, Any], int, str]] = []

    def add_candidates_from_segment(segment: str, base_offset: int):
        for js, pos in _collect_top_level_json_objects_with_spans(segment, base_offset):
            try:
                j = json.loads(js)
            except Exception:
                continue
            if not isinstance(j, dict) or "verdict" not in j:
                continue
            v = str(j.get("verdict", "")).strip()
            if v not in ("True", "False"):
                continue
            # Offensichtliche Experten-Objekte aussortieren
            if any(k in j for k in ("name", "weight")) and set(j.keys()).issubset(
                {"name", "weight", "verdict", "explanation"}
            ):
                continue

            expl = j.get("explanation", "")
            if isinstance(expl, str) and expl.strip():
                # nicht 1:1 aus Input übernehmen
                if input_context and expl.lower() in input_context.lower():
                    continue
                # Platzhalter/Beispiel/CoT wegfiltern
                if "<your own" in expl.lower():
                    continue
                if _is_placeholder_expl(expl):
                    continue
                if _is_chain_of_thoughty(expl):
                    # sehr CoT-ig → eher meiden, aber nicht komplett ausschließen: wir penalizen später
                    pass
            candidates.append((j, pos, js))

    # 1) Aus allen Tag-Segmenten (in Reihenfolge)
    for s_idx, _e_idx, seg in ranges:
        add_candidates_from_segment(seg, base_offset=s_idx + len("BEGIN_DECISION_JSON"))

    # 2) Falls nichts Gültiges aus Tags: aus dem gesamten Raw
    if not candidates:
        add_candidates_from_segment(raw, base_offset=0)

    if not candidates:
        # Letzter Fallback: Erklärung aus Raw säubern
        return {"verdict": "False", "explanation": _clean_explanation(raw)}

    # Scoring: bestes Objekt wählen
    def score(item: Tuple[Dict[str, Any], int, str]) -> Tuple[int, int, int, int]:
        j, pos, js_raw = item
        expl = str(j.get("explanation", "") or "")

        # 1) nach letztem BEGIN_DECISION_JSON?
        after_last_tag = 1 if (last_begin_idx != -1 and pos >= last_begin_idx) else 0

        # 2) Wortanzahl im Zielbereich?
        wc = len(expl.split())
        in_range = 1 if 40 <= wc <= 120 else 0

        # 3) CoT/Example-Penalty
        cot_pen = 1 if _is_chain_of_thoughty(expl) or ("example" in js_raw.lower()) else 0

        # 4) spätere Position bevorzugen
        return (after_last_tag, in_range, -cot_pen, pos)

    candidates.sort(key=score, reverse=True)
    chosen = candidates[0][0]

    verdict = "True" if str(chosen.get("verdict", "")).strip() == "True" else "False"
    explanation = _clean_explanation(str(chosen.get("explanation", "")).strip())

    return {"verdict": verdict, "explanation": explanation}


# ----------------- Explanation Retry / Rewrite -----------------
def _regenerate_explanation(
    tokenizer,
    model,
    claim: str,
    subject_block: str,
    verdict: str,
    max_new_tokens: int = 120,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    prompt = f"""
You already decided the verdict for this claim.
Claim: {claim}
{subject_block}
Verdict: "{verdict}"

Now output ONLY JSON with:
- explanation: 40–90 words, concrete facts, 1–2 generic source mentions (e.g., (CBO 2017; DHS report)), no placeholders.
Do not copy any sentence from earlier text.

JSON:
"""
    out = _generate_json(
        tokenizer,
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    expl = str(out.get("explanation", "")).strip()
    return expl


def _rewrite_decision_expl_if_copied(
    tokenizer, model, verdict: str, experts: List[Dict[str, Any]], claim: str, subject_block: str
) -> str:
    reasons = []
    experts_sorted = sorted(experts, key=lambda e: e.get("weight", 1.0), reverse=True)
    for e in experts_sorted[:3]:
        reasons.append(f"{e['name']} (w={e['weight']}): {e['verdict']}")
    skeleton = "; ".join(reasons)

    prompt = f"""
Rewrite a decision explanation in your own words. Do NOT copy any sentence from input.
Input summary:
- Verdict: {verdict}
- Expert votes: {skeleton}
- Claim: {claim}
- {subject_block}

Output ONLY JSON: {{"explanation": "40–90 words, concrete, objective, with 1–2 generic source mentions (e.g., (GAO 2018; CRS))"}}
"""
    out = _generate_json(
        tokenizer,
        model,
        prompt,
        max_new_tokens=140,
        temperature=0.2,
        top_p=0.9,
    )
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
    expert_raw_texts: List[str] = []

    subject_block = (
        f"Subject(s): {subject}" if subject else "Subject(s): (none/provided)"
    )

    for cfg in experts:
        tok, mdl = _get_model(cfg.model_name, cfg.device_map, cfg.torch_dtype)
        prompt = cfg.prompt_template.format(
            statement=statement, subject_block=subject_block
        )
        parsed, raw_text = _generate_json_and_text(
            tok,
            mdl,
            prompt,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        verdict = "True" if str(parsed.get("verdict", "False")).strip() == "True" else "False"
        explanation = _clean_explanation(str(parsed.get("explanation", "")).strip())

        if _is_placeholder_expl(explanation):
            retry_expl = _regenerate_explanation(
                tok,
                mdl,
                statement,
                subject_block,
                verdict,
                max_new_tokens=min(160, cfg.max_new_tokens + 40),
                temperature=max(0.2, cfg.temperature),
                top_p=cfg.top_p,
            )
            cleaned_retry = _clean_explanation(retry_expl)
            if not _is_placeholder_expl(cleaned_retry):
                explanation = cleaned_retry

        expert_outputs.append(
            {
                "name": cfg.name,
                "weight": float(cfg.weight),
                "verdict": verdict,
                "explanation": explanation,
            }
        )
        expert_raw_texts.append(raw_text)

    all_empty = all(_is_placeholder_expl(e["explanation"]) for e in expert_outputs)
    if all_empty:
        for e in expert_outputs:
            e["explanation"] = "Insufficient factual justification given by this expert."

    tok_d, mdl_d = _get_model(decision.model_name, decision.device_map, decision.torch_dtype)

    def _trim_expl(e: Dict[str, Any], max_words: int = 60) -> str:
        return _clean_explanation(e.get("explanation", ""), max_words=max_words)

    compact_experts = [
        {
            "name": e["name"],
            "weight": e["weight"],
            "verdict": e["verdict"],
            "explanation": _trim_expl(e, max_words=60),
        }
        for e in expert_outputs
    ]
    experts_json = json.dumps(compact_experts, ensure_ascii=False, indent=2)

    dec_prompt = decision.prompt_template.format(experts_json=experts_json)
    final_json, decision_raw_text = _generate_json_and_text(
        tok_d,
        mdl_d,
        dec_prompt,
        max_new_tokens=decision.max_new_tokens,
        temperature=decision.temperature,
        top_p=decision.top_p,
        input_context=experts_json,
    )

    final_verdict = (
        "True" if str(final_json.get("verdict", "False")).strip() == "True" else "False"
    )
    final_expl = _clean_explanation(str(final_json.get("explanation", "")).strip())

    if final_expl and experts_json and final_expl.lower() in experts_json.lower():
        rewritten = _rewrite_decision_expl_if_copied(
            tok_d, mdl_d, final_verdict, compact_experts, statement, subject_block
        )
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
        result["experts_raw"] = expert_raw_texts
        result["decision_raw_text"] = decision_raw_text
        result["decision_raw_parsed"] = final_json

    return result


# ----------------- CSV Runner -----------------
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def run_multiagent_on_csv(
    csv_path: str,
    experts: list,
    decision: DecisionConfig,
    statement_col: str = "statement",
    subject_col: str = "subjects",
    label_col: str = "label",
    limit: Optional[int] = None,
    save_path: Optional[str] = None,
    show_progress: bool = True,
    return_intermediates: bool = False,
):
    df = pd.read_csv(csv_path, sep="\t", quotechar='"', engine="python", dtype=str)

    if limit:
        df = df.head(limit).copy()

    preds, expls, expert_dumps = [], [], []
    decisions_raw_text, experts_raw_text = [], []

    it = (
        tqdm(df.iterrows(), total=len(df), desc="🔎 Fact-checking")
        if show_progress
        else df.iterrows()
    )

    for _, row in it:
        statement = row[statement_col]
        subject = row[subject_col] if subject_col in df.columns else None

        res = classify_claim_multiagent(
            statement=statement,
            subject=subject,
            experts=experts,
            decision=decision,
            return_intermediates=return_intermediates,
        )

        preds.append(res["final_verdict"])
        expls.append(res["final_explanation"])
        expert_dumps.append(res.get("experts") if return_intermediates else None)
        if return_intermediates:
            decisions_raw_text.append(res.get("decision_raw_text"))
            experts_raw_text.append(json.dumps(res.get("experts_raw"), ensure_ascii=False))
        else:
            decisions_raw_text.append(None)
            experts_raw_text.append(None)

    df["prediction"] = preds
    df["explanation"] = expls

    if return_intermediates:
        df["experts"] = expert_dumps
        df["decision_raw_text"] = decisions_raw_text
        df["experts_raw_text"] = experts_raw_text

    metrics: Dict[str, Any] = {}
    if label_col in df.columns:
        y_true = df[label_col]
        y_pred = df["prediction"]
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_true": precision_score(y_true, y_pred, pos_label="True"),
            "recall_true": recall_score(y_true, y_pred, pos_label="True"),
            "f1_true": f1_score(y_true, y_pred, pos_label="True"),
            "report": classification_report(y_true, y_pred, labels=["True", "False"]),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["True", "False"]),
        }
        print("\n📊 Evaluation")
        for k, v in metrics.items():
            if k != "report":
                print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
        print("\nClassification report:\n", metrics["report"])

    if save_path:
        dirn = os.path.dirname(save_path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"💾 Ergebnisse gespeichert unter: {save_path}")

    return df, metrics


__all__ = [
    "ExpertConfig",
    "DecisionConfig",
    "classify_claim_multiagent",
    "run_multiagent_on_csv",
]
