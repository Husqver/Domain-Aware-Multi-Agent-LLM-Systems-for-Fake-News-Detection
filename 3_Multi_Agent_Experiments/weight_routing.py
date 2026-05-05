import os, json
from typing import Optional, Dict, Any, List

import torch
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel, PeftConfig

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # wähle deine GPU(s)


# =========================
# 1) JSON Schema (Router)
# =========================
JSON_SCHEMA_ROUTER_WEIGHTS = {
    "type": "object",
    "properties": {
        "politics": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "economy": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "health_science": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["politics", "economy", "health_science"],
    "additionalProperties": False,
}


# =========================
# 2) Prompt
# =========================
ROUTER_SYSTEM = (
    "You are a routing module for a 3-expert fact-checking panel.\n"
    "Experts:\n"
    "- politics: U.S. politics, elections, legislation, public records\n"
    "- economy: macroeconomics, labor stats, budgets, taxation\n"
    "- health_science: public health, healthcare policy, scientific evidence\n\n"
    "Task: Given a claim (and optional subjects), output weights indicating relevance of each expert.\n"
    "Weights MUST sum to 1.\n"
    "If unclear, distribute more evenly.\n"
)

ROUTER_USER_TMPL = (
    "Return ONLY a JSON object with keys politics, economy, health_science.\n"
    "Each value is a number in [0,1]. Values MUST SUM to 1.\n\n"
    "Claim: {statement}\n"
    "{subject_block}\n"
    "JSON:"
)


def _subject_block(subjects: Optional[str]) -> str:
    s = (subjects or "").strip()
    return f"Subjects: {s}" if s else "Subjects: (none)"


def _normalize_weights(obj: Dict[str, Any]) -> Dict[str, float]:
    keys = ["politics", "economy", "health_science"]
    vals = []
    for k in keys:
        try:
            v = float(obj.get(k, 0.0))
        except Exception:
            v = 0.0
        v = max(0.0, min(1.0, v))
        vals.append(v)

    s = sum(vals)
    if s <= 1e-12:
        return {"politics": 1/3, "economy": 1/3, "health_science": 1/3}
    vals = [v / s for v in vals]
    return dict(zip(keys, vals))


# =========================
# 3) Model loader (Base or LoRA folder)
# =========================
def _load_model_any(model_id_or_adapter_path: str, torch_dtype=torch.float16):
    use_cuda = torch.cuda.is_available()
    dtype = torch_dtype if use_cuda else torch.float32
    device_map = "auto" if use_cuda else None

    adapter_cfg = os.path.join(model_id_or_adapter_path, "adapter_config.json")

    # ----- LoRA adapter folder -----
    if os.path.exists(adapter_cfg):
        peft_cfg = PeftConfig.from_pretrained(model_id_or_adapter_path)
        base_id = peft_cfg.base_model_name_or_path

        # tokenizer: adapter if present, else base
        try:
            tok = AutoTokenizer.from_pretrained(model_id_or_adapter_path, use_fast=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)

        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id

        cfg = AutoConfig.from_pretrained(base_id)
        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            config=cfg,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        mdl = PeftModel.from_pretrained(base, model_id_or_adapter_path)

        try:
            if use_cuda and torch.cuda.device_count() <= 1:
                mdl = mdl.merge_and_unload()
        except Exception:
            pass

        mdl.eval()
        return tok, mdl

    # ----- Normal HF model -----
    tok = AutoTokenizer.from_pretrained(model_id_or_adapter_path, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    cfg = AutoConfig.from_pretrained(model_id_or_adapter_path)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id_or_adapter_path,
        config=cfg,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    ).eval()
    return tok, mdl


# =========================
# 4) Router class
# =========================
class RouterWeights:
    def __init__(self, router_model_id: str, torch_dtype: str = "float16"):
        dtype = torch.float16 if torch_dtype == "float16" else torch.bfloat16
        self.tokenizer, self.model = _load_model_any(router_model_id, torch_dtype=dtype)

        self.parser = JsonSchemaParser(JSON_SCHEMA_ROUTER_WEIGHTS)
        self.prefix_fn = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, self.parser)

    def _build_prompt(self, statement: str, subjects: Optional[str], use_subjects: bool) -> str:
        sb = _subject_block(subjects if use_subjects else "")

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": ROUTER_USER_TMPL.format(statement=statement, subject_block=sb)},
        ]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return (
                f"[SYSTEM]\n{ROUTER_SYSTEM}\n[/SYSTEM]\n"
                f"{ROUTER_USER_TMPL.format(statement=statement, subject_block=sb)}"
            )

    @torch.no_grad()
    def weights_for_row(
        self,
        statement: str,
        subjects: Optional[str],
        use_subjects: bool,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Dict[str, float]:
        prompt = self._build_prompt(statement, subjects, use_subjects=use_subjects)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            first_device = next(self.model.parameters()).device
            inputs = {k: v.to(first_device) for k, v in inputs.items()}

        do_sample = bool(temperature and temperature > 0)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id,
            prefix_allowed_tokens_fn=self.prefix_fn,
        )
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)

        out = self.model.generate(**inputs, **gen_kwargs)

        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        txt = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        try:
            obj = json.loads(txt)
            if not isinstance(obj, dict):
                raise ValueError("not dict")
        except Exception:
            obj = {"politics": 1/3, "economy": 1/3, "health_science": 1/3}

        return _normalize_weights(obj)


# =========================
# 5) File runner
# =========================
def generate_router_weights_for_file(
    input_path: str,
    router_model_id: str,
    output_path: Optional[str] = None,
    sep: str = "\t",
    statement_col: str = "statement",
    subject_col: str = "subjects", 
    use_subjects: bool = True,
    nrows: Optional[int] = None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    torch_dtype: str = "float16",
    show_progress: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep=sep, dtype=str, nrows=nrows)

    if statement_col not in df.columns:
        raise ValueError(f"statement_col='{statement_col}' not found. Columns: {list(df.columns)}")

    router = RouterWeights(router_model_id=router_model_id, torch_dtype=torch_dtype)

    pol_w, eco_w, hs_w = [], [], []

    it = tqdm(df.iterrows(), total=len(df), desc="🧭 Routing weights") if show_progress else df.iterrows()
    for _, row in it:
        statement = row.get(statement_col, "")
        subjects = row.get(subject_col, "") if subject_col in df.columns else ""

        w = router.weights_for_row(
            statement=statement,
            subjects=subjects,
            use_subjects=use_subjects,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        pol_w.append(w["politics"])
        eco_w.append(w["economy"])
        hs_w.append(w["health_science"])

    df["w_politics"] = pol_w
    df["w_economy"] = eco_w
    df["w_health_science"] = hs_w

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"✅ Saved: {output_path}")

    return df