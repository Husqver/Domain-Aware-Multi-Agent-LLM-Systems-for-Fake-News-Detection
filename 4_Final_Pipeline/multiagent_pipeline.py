import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import re
import json
from typing import Dict, Any, List, Optional, Tuple

import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from jinja2.exceptions import TemplateError


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
from peft import PeftModel


try:
    from langchain_experimental.pydantic_v1 import BaseModel, Field
    from langchain_experimental.llms import LMFormatEnforcer
except Exception as e:
    BaseModel = None
    Field = None
    LMFormatEnforcer = None
    _LMFE_IMPORT_ERROR = e
else:
    _LMFE_IMPORT_ERROR = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 0) Helpers: dtype, device, JSON
# -----------------------------
def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        # prefer bf16 if supported, else fp16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_first_json(text_or_obj: Any) -> Dict[str, Any]:
    """
    Robustly extract the first JSON object from:
      - dict
      - list containing dict/strings
      - string containing a JSON object
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


def norm_bool_label(x: Any) -> str:
    if not isinstance(x, str):
        return "False"
    s = x.strip().strip(".,:;!?)(").lower()
    if s.startswith("true"):
        return "True"
    if s.startswith("false"):
        return "False"
    return "False"

ROUTER_SYSTEM_PROMPT = """You are a strict domain classifier.

You receive a political or public policy claim (statement) and must map it
to exactly ONE high-level domain label from a given list.

You MUST answer with EXACTLY ONE label string from the list.
Do NOT explain. Only output the label.
"""



# ---------------------------------------
# 1) Robust prompt builder (chat/no-chat)
# ---------------------------------------
def build_prompt(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    add_generation_prompt: bool,
) -> str:
    """
    Build a prompt that works across:
      - chat/instruct tokenizers with templates
      - templates that don't support "system" role
      - base LMs without chat templates

    Strategy:
      1) Try apply_chat_template with system+user
      2) If that fails, merge system into user and try user-only
      3) If no chat template, use plain formatting
    """
    has_chat = hasattr(tokenizer, "apply_chat_template") and bool(getattr(tokenizer, "chat_template", None))

    if has_chat:
        # 1) Try system + user
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass

        # 2) Retry with user-only (system merged into user)
        try:
            merged = f"{system_prompt}\n\n{user_prompt}"
            messages = [{"role": "user", "content": merged}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass

    # 3) Plain fallback
    if add_generation_prompt:
        return f"{system_prompt}\n\n{user_prompt}\n\nAnswer:"
    return f"{system_prompt}\n\n{user_prompt}\n\n"


# ---------------------------------------
# 2) Default prompts (router + experts)
# ---------------------------------------
def make_router_system_prompt(super_labels: List[str]) -> str:
    labels_bullets = "\n".join([f"- {l}" for l in super_labels])
    return (
        "You are a strict domain classifier.\n\n"
        "You receive a political or public policy claim (statement) and must map it\n"
        "to exactly ONE high-level domain label from this list:\n\n"
        f"{labels_bullets}\n\n"
        "You MUST answer with EXACTLY ONE label string from the list.\n"
        "Do NOT explain. Only output the label."
    )


def make_router_user_prompt(statement: str, super_labels: List[str]) -> str:
    label_list = ", ".join(super_labels)
    return (
        f"Valid labels:\n{label_list}\n\n"
        f"Statement:\n{statement}\n\n"
        "Return ONLY ONE label from the list above.\n"
        "Respond ONLY with the label."
    )


def get_claim_verdict_schema() -> Dict[str, Any]:
    """
    JSON schema for experts: {"verdict": "True/False", "explanation": "..."}
    """
    if BaseModel is None or Field is None:
        raise RuntimeError(
            "langchain_experimental is not available, but it's required for the current LMFormatEnforcer setup. "
            f"Import error: {_LMFE_IMPORT_ERROR}"
        )

    class ClaimVerdict(BaseModel):
        verdict: str = Field(..., description="Binary verdict: True or False")
        explanation: str = Field(..., description="2–4 sentences, brief, no stepwise reasoning")

    schema = ClaimVerdict.schema()
    # tighten allowed values (helps schema enforcers)
    schema["properties"]["verdict"]["enum"] = ["True", "False"]
    return schema


def build_expert_system_prompt(domain_name: str, json_schema: Dict[str, Any]) -> str:
    return (
        f"You are a fact-checking expert specialized in the '{domain_name}' domain.\n\n"
        "You receive a political or public policy claim (statement) and must decide if it is factually correct.\n\n"
        "You must output a JSON object that follows this schema:\n\n"
        f"{json.dumps(json_schema, indent=2)}\n\n"
        'The field "verdict" MUST be either "True" or "False".\n'
        'The field "explanation" MUST be 2–4 concise sentences.\n'
        "Do NOT include step-by-step reasoning or lists.\n\n"
        "Reply with JSON only. No markdown, no extra text."
    )


def build_expert_user_prompt(statement: str) -> str:
    return f"Claim:\n{statement}\n\nReturn the JSON object now."

def render_prompt(tokenizer, system_prompt: str, user_prompt: str, add_generation_prompt: bool) -> str:
    has_chat = hasattr(tokenizer, "apply_chat_template") and bool(getattr(tokenizer, "chat_template", None))

    if has_chat:
        # Try normal system+user
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except (TemplateError, Exception):
            # Some templates (e.g., certain Gemma/others) reject "system"
            merged = f"{system_prompt}\n\n{user_prompt}"
            msgs2 = [{"role": "user", "content": merged}]
            return tokenizer.apply_chat_template(
                msgs2, tokenize=False, add_generation_prompt=add_generation_prompt
            )

    # Plain fallback (e.g., deepseek-llm-7b-base)
    if add_generation_prompt:
        return f"{system_prompt}\n\n{user_prompt}\n\nLabel:"
    return f"{system_prompt}\n\n{user_prompt}\n\n"


# ---------------------------------------
# 3) Loading utilities (tokenizer / model)
# ---------------------------------------
def load_tokenizer(model_id_or_dir: str):
    tok = AutoTokenizer.from_pretrained(model_id_or_dir, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def load_base_model(model_id: str):
    dtype = pick_dtype()
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    mdl.eval()
    return mdl


def load_lora_model(base_model_id: str, lora_dir: str):
    base = load_base_model(base_model_id)
    mdl = PeftModel.from_pretrained(base, lora_dir)
    mdl.eval()
    return mdl


# ---------------------------------------
# 4) Router inference (statement -> label)
# ---------------------------------------
@torch.no_grad()
def run_router(
    df: pd.DataFrame,
    base_model_id: str,
    router_lora_dir: str,
    super_labels: List[str],
    text_col: str = "statement",
    max_new_tokens: int = 8,
) -> pd.Series:
    print("\n=== Phase 1: Routing (Statement -> super_domain) ===")

    try:
        tokenizer = AutoTokenizer.from_pretrained(router_lora_dir, use_fast=True, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, trust_remote_code=True)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = pick_dtype()
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    base.eval()

    model = PeftModel.from_pretrained(base, router_lora_dir)
    model.eval()

    label_list = ", ".join(super_labels)
    system_prompt = domain_config.build_system_prompt() if "domain_config" in globals() else ROUTER_SYSTEM_PROMPT

    preds: List[str] = []

    for row in tqdm(df.itertuples(), total=len(df), desc="🌍 Routing-only", dynamic_ncols=True):
        statement = getattr(row, text_col)

        # EXACT TRAINING STYLE:
        user_content = (
            f"Valid labels:\n{label_list}\n\n"
            f"Claim:\n{statement}\n\n"
            "Return ONLY ONE label from the list above.\n"
            "Label:"
        )

        # EXACT TRAINING STYLE: add_generation_prompt=False
        prompt = render_prompt(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_content,
            add_generation_prompt=False,
        )

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        new_tokens = gen[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().lower()

        if not answer:
            preds.append("misc" if "misc" in super_labels else super_labels[0])
            continue

        pred_dom = "misc" if "misc" in super_labels else super_labels[0]

        # word match -> substring fallback
        for L in super_labels:
            if re.search(r"\b" + re.escape(L.lower()) + r"\b", answer):
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



# ---------------------------------------
# 5) Expert inference (domain subset -> JSON)
# ---------------------------------------
def run_expert_for_domain(
    df: pd.DataFrame,
    domain_name: str,
    base_model_id: str,
    expert_root: str,
    text_col: str = "statement",
    max_new_tokens: int = 192,
) -> Tuple[pd.Series, pd.Series]:
    """
    Loads ONE expert LoRA and produces verdict/explanation for the domain subset.
    Uses LMFormatEnforcer to enforce JSON schema.
    """
    if LMFormatEnforcer is None:
        raise RuntimeError(
            "LMFormatEnforcer is not available. Install langchain-experimental or adjust the code to another enforcer. "
            f"Import error: {_LMFE_IMPORT_ERROR}"
        )

    expert_dir = os.path.join(expert_root, domain_name)
    if not os.path.isdir(expert_dir):
        raise FileNotFoundError(f"Expert directory for domain '{domain_name}' not found: {expert_dir}")

    device = get_device()
    json_schema = get_claim_verdict_schema()

    # Tokenizer: prefer expert dir, else base
    try:
        tokenizer = load_tokenizer(expert_dir)
    except Exception:
        tokenizer = load_tokenizer(base_model_id)

    model = load_lora_model(base_model_id, expert_dir)

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
        json_schema=json_schema,
    )

    verdicts: List[str] = []
    explanations: List[str] = []

    system_prompt = build_expert_system_prompt(domain_name, json_schema)

    for row in tqdm(df.itertuples(), total=len(df), desc=f"🧠 Expert: {domain_name}", dynamic_ncols=True):
        statement = getattr(row, text_col)
        user_prompt = build_expert_user_prompt(statement)

        prompt_text = build_prompt(tokenizer, system_prompt, user_prompt, add_generation_prompt=True)

        try:
            raw = enforced_llm.invoke(prompt_text, max_new_tokens=max_new_tokens)
            parsed = extract_first_json(raw)
            verdict = parsed.get("verdict", "False")
            expl = (parsed.get("explanation", "") or "").strip()
        except Exception as e:
            verdict = "False"
            expl = f"Fallback due to parsing/enforcement error: {str(e)[:200]}"

        verdict = norm_bool_label(verdict)
        if not expl:
            expl = "No explanation was provided."

        verdicts.append(verdict)
        explanations.append(expl)

    # free VRAM
    del model
    torch.cuda.empty_cache()

    return pd.Series(verdicts, index=df.index), pd.Series(explanations, index=df.index)


# ---------------------------------------
# 6) Full pipeline runner
# ---------------------------------------
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
) -> pd.DataFrame:
    """
    Full end-to-end evaluation:
      1) Load test data
      2) Router predicts domain for each statement
      3) For each domain, load corresponding expert LoRA and predict JSON verdict/explanation
      4) Compute routing + verdict metrics
      5) Save CSV with predictions

    Returns the full DataFrame with added columns:
      - domain_pred_router
      - domain_pred
      - verdict_pred
      - explanation_pred
    """
    df = pd.read_csv(
        test_path,
        sep=sep,
        quotechar='"',
        engine="python",
        dtype=str,
    )

    # Basic filtering (keeps evaluation sane)
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
    )

    print("\nPredicted domain distribution (router):")
    print(df["domain_pred_router"].value_counts())

    # initialize outputs
    df["domain_pred"] = df["domain_pred_router"]
    df["verdict_pred"] = "False"
    df["explanation_pred"] = ""

    # --- Experts sequentially ---
    for dom in super_labels:
        idxs = df.index[df["domain_pred_router"] == dom].tolist()
        if not idxs:
            print(f"\n⚠️ No samples routed to domain '{dom}'.")
            continue

        df_dom = df.loc[idxs]
        v_dom, e_dom = run_expert_for_domain(
            df=df_dom,
            domain_name=dom,
            base_model_id=base_model_id,
            expert_root=expert_root,
            text_col=text_col,
            max_new_tokens=max_expert_new_tokens,
        )

        df.loc[idxs, "verdict_pred"] = v_dom
        df.loc[idxs, "explanation_pred"] = e_dom

    # --- Metrics ---
    print("\nSample rows with predictions:")
    show_cols = [text_col, domain_col, "domain_pred", label_col, "verdict_pred"]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].head())

    # Routing accuracy (only if true domains exist)
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
        else:
            print("\n⚠️ Domain routing accuracy skipped (no valid domain labels).")

    # Verdict accuracy
    y_true = df[label_col].map(norm_bool_label)
    y_pred = df["verdict_pred"].map(norm_bool_label)
    verdict_acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Verdict accuracy (overall): {verdict_acc:.3f}\n")
    print("Classification report (True/False):")
    print(classification_report(y_true, y_pred, labels=["True", "False"], zero_division=0))
    print("Confusion matrix [rows=true, cols=pred] (True, False):")
    print(confusion_matrix(y_true, y_pred, labels=["True", "False"]))

    # Per-domain verdict accuracy (if true domains exist)
    if domain_col in df.columns:
        print("\n===== Per-domain verdict accuracy (by true domain) =====")
        for dom in super_labels:
            df_dom_true = df[df[domain_col] == dom]
            if df_dom_true.empty:
                continue
            yt = df_dom_true[label_col].map(norm_bool_label)
            yp = df_dom_true["verdict_pred"].map(norm_bool_label)
            acc_dom = accuracy_score(yt, yp)
            print(f"{dom:22s} n={len(df_dom_true):4d} acc={acc_dom:.3f}")

    # --- Save ---
    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\n💾 Saved predictions to: {out_path}")

    return df
