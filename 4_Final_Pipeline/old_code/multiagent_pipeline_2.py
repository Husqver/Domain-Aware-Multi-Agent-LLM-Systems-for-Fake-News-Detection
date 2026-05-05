import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple, Literal

import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
from peft import PeftModel

from langchain_experimental.pydantic_v1 import BaseModel, Field
from langchain_experimental.llms import LMFormatEnforcer

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # eine GPU auswählen

# -----------------------------
# JSON schema
# -----------------------------
class ClaimVerdict(BaseModel):
    verdict: Literal["True", "False"] = Field(..., description="Binary verdict")
    explanation: str = Field(..., description="2–4 sentences, brief, no stepwise reasoning")


JSON_SCHEMA = ClaimVerdict.schema()


# -----------------------------
# Helpers
# -----------------------------
def is_llama_model(base_model_id: str) -> bool:
    s = (base_model_id or "").lower()
    return ("llama" in s) or ("meta-llama" in s)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pick_dtype(force_fp16: bool = True) -> torch.dtype:
    if torch.cuda.is_available():
        return torch.float16 if force_fp16 else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return torch.float32


def norm_bool_label(x: Any) -> str:
    if not isinstance(x, str):
        return "False"
    s = x.strip().strip(".,:;!?)(").lower()
    if s.startswith("true"):
        return "True"
    if s.startswith("false"):
        return "False"
    return "False"


def extract_first_json(text_or_obj: Any) -> Dict[str, Any]:
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
                                if isinstance(obj, dict):
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
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

    raise ValueError("No JSON object found in model output")


def load_tokenizer(
    base_model_id: str,
    prefer_dir: Optional[str] = None,
    trust_remote_code: bool = True,
    force_base_tokenizer: bool = False,
):
    if force_base_tokenizer:
        # FIX 5: explicit trust_remote_code=False for Llama
        tok = AutoTokenizer.from_pretrained(
            base_model_id,
            use_fast=True,
            trust_remote_code=False,
        )
    else:
        if prefer_dir:
            try:
                tok = AutoTokenizer.from_pretrained(prefer_dir, use_fast=True, trust_remote_code=trust_remote_code)
            except Exception:
                tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, trust_remote_code=trust_remote_code)
        else:
            tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, trust_remote_code=trust_remote_code)

    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    return tok


# -----------------------------
# Prompt rendering
# -----------------------------
def render_prompt_strict_system_user(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    add_generation_prompt: bool = True,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def render_prompt_robust(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    add_generation_prompt: bool = True,
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return render_prompt_strict_system_user(tokenizer, system_prompt, user_prompt, add_generation_prompt)
        except Exception:
            pass
        try:
            merged = f"{system_prompt}\n\n{user_prompt}"
            messages2 = [{"role": "user", "content": merged}]
            return tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=add_generation_prompt)
        except Exception:
            pass

    return f"{system_prompt}\n\n{user_prompt}\n\nAnswer:" if add_generation_prompt else f"{system_prompt}\n\n{user_prompt}\n\n"


def load_base_and_lora(
    base_model_id: str,
    lora_dir: str,
    device: torch.device,
    force_fp16: bool = True,
    trust_remote_code: bool = True,
):
    dtype = pick_dtype(force_fp16=force_fp16)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    ).to(device)
    base.eval()

    model = PeftModel.from_pretrained(base, lora_dir).to(device)
    model.eval()
    return base, model


# -----------------------------
# Prompts
# -----------------------------
ROUTER_SYSTEM_PROMPT_RICH = """You are a strict domain classifier.

You receive a political or public policy claim (statement) and must map it
to exactly ONE high-level domain label from this list:

- economy: money, jobs, taxes, trade, business, budget, social security, pensions, stimulus
- health_social: healthcare, medicare, medicaid, public health, drugs, families, women, children, veterans, hunger, disability
- foreign_security: foreign policy, wars, military, terrorism, homeland security, nuclear weapons, international relations
- law_rights: crime, courts, civil rights, guns, immigration, legal issues, human rights, abortion, constitutional questions
- politics_government: elections, campaigns, candidates, parties, congress, political promises, redistricting, government rules and procedures
- environment_energy: climate, environment, energy, oil, gas, water, weather, transportation, pollution, natural resources
- society_culture: education, religion, diversity, LGBT, marriage, sports, pop culture, social norms, demographics
- misc: everything that does not clearly fit any of the above

You MUST answer with EXACTLY ONE label string from this list.
Do NOT explain. Only output the label.
"""


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


# -----------------------------
# Router
# -----------------------------
@torch.no_grad()
def run_router(
    df: pd.DataFrame,
    base_model_id: str,
    router_lora_dir: str,
    super_labels: List[str],
    text_col: str = "statement",
    max_new_tokens: int = 8,
    force_fp16: bool = True,
    strict_prompting: bool = True,
) -> pd.Series:
    device = get_device()
    llama = is_llama_model(base_model_id)

    tokenizer = load_tokenizer(
        base_model_id=base_model_id,
        prefer_dir=router_lora_dir,
        trust_remote_code=(not llama),
        force_base_tokenizer=llama,
    )

    base, model = load_base_and_lora(
        base_model_id=base_model_id,
        lora_dir=router_lora_dir,
        device=device,
        force_fp16=force_fp16,
        trust_remote_code=(not llama),
    )

    preds: List[str] = []
    label_list = ", ".join(super_labels)

    for row in tqdm(df.itertuples(), total=len(df), desc="🌍 Routing-only", dynamic_ncols=True):
        statement = getattr(row, text_col)

        user_content = (
            f"Valid labels:\n{label_list}\n\n"
            f"Statement:\n{statement}\n\n"
            "Return ONLY ONE label from the list above.\n"
            "Respond ONLY with the label.\n\n"
            "Label:"
        )

        if strict_prompting:
            prompt = render_prompt_strict_system_user(tokenizer, ROUTER_SYSTEM_PROMPT_RICH, user_content, add_generation_prompt=True)
        else:
            prompt = render_prompt_robust(tokenizer, ROUTER_SYSTEM_PROMPT_RICH, user_content, add_generation_prompt=True)

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

        if not answer:
            # FIX 1: always "misc", no condition — matches original exactly
            preds.append("misc")
            continue

        # FIX 1 cont.: pred_dom always starts as "misc"
        pred_dom = "misc"
        for L in super_labels:
            pattern = r"\b" + re.escape(L.lower()) + r"\b"
            if re.search(pattern, answer):
                pred_dom = L
                break
        else:
            # substring fallback, only runs if no break above (for-else)
            for L in super_labels:
                if L.lower() in answer:
                    pred_dom = L
                    break

        preds.append(pred_dom)

    del model
    del base
    torch.cuda.empty_cache()

    return pd.Series(preds, index=df.index, name="domain_pred_router")


# -----------------------------
# Experts
# -----------------------------
def run_expert_for_domain(
    df: pd.DataFrame,
    domain_name: str,
    base_model_id: str,
    expert_root: str,
    text_col: str = "statement",
    max_new_tokens: int = 192,
    force_fp16: bool = True,
    strict_prompting: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    device = get_device()
    llama = is_llama_model(base_model_id)

    expert_dir = os.path.join(expert_root, domain_name)
    if not os.path.isdir(expert_dir):
        raise FileNotFoundError(f"Expert directory for domain '{domain_name}' not found: {expert_dir}")

    tokenizer = load_tokenizer(
        base_model_id=base_model_id,
        prefer_dir=expert_dir,
        trust_remote_code=(not llama),
        force_base_tokenizer=llama,
    )

    base, model = load_base_and_lora(
        base_model_id=base_model_id,
        lora_dir=expert_dir,
        device=device,
        force_fp16=force_fp16,
        trust_remote_code=(not llama),
    )

    gen_pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        return_full_text=False,
    )

    enforced_llm = LMFormatEnforcer(pipeline=gen_pipe, json_schema=JSON_SCHEMA)
    system_prompt = build_expert_system_prompt(domain_name)

    verdicts: List[str] = []
    explanations: List[str] = []

    for row in tqdm(df.itertuples(), total=len(df), desc=f"🧠 {domain_name}", dynamic_ncols=True):
        statement = getattr(row, text_col)
        user_content = f"Claim:\n{statement}\n\nReturn the JSON object now."

        if strict_prompting:
            prompt_text = render_prompt_strict_system_user(tokenizer, system_prompt, user_content, add_generation_prompt=True)
        else:
            prompt_text = render_prompt_robust(tokenizer, system_prompt, user_content, add_generation_prompt=True)

        try:
            raw = enforced_llm.invoke(prompt_text, max_new_tokens=max_new_tokens)
            parsed = extract_first_json(raw)
            # FIX 2+3: get verdict with NO default (None on missing), then two-step guard
            verdict = parsed.get("verdict")
            expl = (parsed.get("explanation", "") or "").strip()
        except Exception as e:
            print(f"[WARN] JSON parse failed for domain '{domain_name}': {e}")
            verdict = "False"
            expl = f"Fallback due to parsing error: {str(e)[:200]}"

        # FIX 2: TWO-STEP guard — exactly matching original code
        # Step 1: explicit membership check (catches None, wrong strings, etc.)
        if verdict not in ["True", "False"]:
            verdict = "False"
        # Step 2: norm_bool_label at METRICS time (not here) — so we store raw string
        # NOTE: verdict is already "True" or "False" at this point, norm_bool_label
        # at metrics time is a no-op, but we keep it there for safety.

        if not expl:
            expl = "No explanation was provided."

        verdicts.append(verdict)
        explanations.append(expl)

    del model
    del base
    torch.cuda.empty_cache()

    return pd.Series(verdicts, index=df.index), pd.Series(explanations, index=df.index)


# -----------------------------
# Full pipeline
# -----------------------------
def run_full_pipeline(
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
    max_router_new_tokens: int = 8,
    max_expert_new_tokens: int = 192,
    force_fp16: bool = True,
    strict_prompting: Optional[bool] = None,  # None => auto
) -> pd.DataFrame:

    # FIX 4: auto-detect strict_prompting BEFORE any function call
    # (original always used strict=True for Llama at the call site)
    if strict_prompting is None:
        strict_prompting = is_llama_model(base_model_id)

    df = pd.read_csv(test_path, sep=sep, quotechar='"', engine="python", dtype=str)
    df = df.dropna(subset=[text_col, label_col])
    df = df[df[text_col].str.strip() != ""]
    df = df[df[label_col].isin(["True", "False"])]

    print("Test set size:", len(df))
    if domain_col in df.columns:
        print("True domain distribution:")
        print(df[domain_col].value_counts())

    # --- Router ---
    df["domain_pred_router"] = run_router(
        df=df,
        base_model_id=base_model_id,
        router_lora_dir=router_lora_dir,
        super_labels=super_labels,
        text_col=text_col,
        max_new_tokens=max_router_new_tokens,
        force_fp16=force_fp16,
        strict_prompting=strict_prompting,  # FIX 4: passed explicitly
    )

    print("\nPredicted domain distribution (router):")
    print(df["domain_pred_router"].value_counts())

    df["domain_pred"] = df["domain_pred_router"]
    df["verdict_pred"] = "False"
    df["explanation_pred"] = ""

    # --- Experts sequentially ---
    for dom in super_labels:
        idxs = df.index[df["domain_pred_router"] == dom].tolist()
        if not idxs:
            print(f"\n⚠️ No samples routed to domain '{dom}'.")
            continue

        v_dom, e_dom = run_expert_for_domain(
            df=df.loc[idxs],
            domain_name=dom,
            base_model_id=base_model_id,
            expert_root=expert_root,
            text_col=text_col,
            max_new_tokens=max_expert_new_tokens,
            force_fp16=force_fp16,
            strict_prompting=strict_prompting,  # FIX 4
        )

        df.loc[idxs, "verdict_pred"] = v_dom
        df.loc[idxs, "explanation_pred"] = e_dom

    # --- Metrics: domain routing ---
    if domain_col in df.columns:
        mask_valid = df[domain_col].isin(super_labels) & df["domain_pred"].isin(super_labels)
        if mask_valid.any():
            y_true_dom = df.loc[mask_valid, domain_col]
            y_pred_dom = df.loc[mask_valid, "domain_pred"]
            dom_acc = accuracy_score(y_true_dom, y_pred_dom)
            print(f"\n🌍 Domain routing accuracy: {dom_acc:.3f}")
            print("Confusion matrix (rows=true, cols=pred):")
            print(confusion_matrix(y_true_dom, y_pred_dom, labels=super_labels))
            print("Label order:", super_labels)

    # --- Metrics: verdict ---
    # norm_bool_label at metrics time only (FIX 2)
    y_true = df[label_col].map(norm_bool_label)
    y_pred = df["verdict_pred"].map(norm_bool_label)
    verdict_acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Verdict accuracy (overall): {verdict_acc:.3f}\n")
    print("Classification report (True/False):")
    print(classification_report(y_true, y_pred, labels=["True", "False"], zero_division=0))
    print("Confusion matrix [rows=true, cols=pred] (True, False):")
    print(confusion_matrix(y_true, y_pred, labels=["True", "False"]))

    if domain_col in df.columns:
        print("\n===== Per-domain verdict accuracy (by true domain) =====")
        for dom in super_labels:
            df_dom_true = df[df[domain_col] == dom]
            if df_dom_true.empty:
                continue
            acc_dom = accuracy_score(
                df_dom_true[label_col].map(norm_bool_label),
                df_dom_true["verdict_pred"].map(norm_bool_label),
            )
            print(f"{dom:22s} n={len(df_dom_true):4d} acc={acc_dom:.3f}")

    try:
        fb = (df["explanation_pred"].astype(str).str.startswith("Fallback")).sum()
        print(f"\n[debug] Fallback count (parsing/enforcer): {fb}")
    except Exception:
        pass

    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\n💾 Saved predictions to: {out_path}")

    return df