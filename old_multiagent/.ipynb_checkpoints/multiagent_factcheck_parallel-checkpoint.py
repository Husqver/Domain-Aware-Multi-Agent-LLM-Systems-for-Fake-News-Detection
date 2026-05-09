#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-agent fact-checking with batched, multi-GPU inference using Hugging Face Accelerate.
- Runs 3 expert "roles" + 1 decision layer, all with the SAME base model (e.g., Llama-3.1-8B-Instruct).
- Uses batching over samples and data-parallel execution across available GPUs.
- Robust JSON extraction; early-exit option; metrics & CSV output.

Usage (example):
  accelerate launch --num_processes 4 multiagent_factcheck_distributed.py \
    --csv_path own_datasets/test_binary_labels.csv \
    --save_path results/multiagent_results.csv \
    --batch_size 8 \
    --model_name meta-llama/Llama-3.1-8B-Instruct

Notes:
- Input is expected to be TSV by default (change --sep if needed).
- Requires `accelerate`, `transformers`, `torch`, `pandas`, `scikit-learn`.
"""

from __future__ import annotations
import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# -------------------------- Config dataclasses --------------------------

DEFAULT_EXPERT_PROMPT = """You are a domain expert and careful fact-checker.
Task: Verify the claim strictly with objective, publicly verifiable facts. If evidence is insufficient, choose False.
Return ONLY a compact JSON object with fields:
- verdict: "True" or "False"
- explanation: short rationale (2-4 sentences, cite known facts/sources generically)

Input:
Claim: {statement}
{subject_block}
JSON:"""

DEFAULT_DECISION_PROMPT = """You are a senior fact-checking adjudicator.
You receive multiple expert analyses, each with a verdict, explanation, and weight.

Task:
- Evaluate the experts' reasoning quality, factual grounding, and agreement with publicly verifiable evidence.
- Consider the explicit weights assigned to each expert.
- Prioritize well-supported, domain-relevant arguments over unsupported or vague statements.
- If the majority view (weighted) is inconclusive or poorly justified, choose "False".

Return ONLY this JSON:
{{"verdict":"True|False","explanation":"2-4 explaining sentences"}}

Input JSON (array of expert objects):
{experts_json}
JSON:"""

@dataclass
class ExpertConfig:
    name: str
    prompt_template: str = DEFAULT_EXPERT_PROMPT
    weight: float = 1.0
    max_new_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.95

@dataclass
class DecisionConfig:
    prompt_template: str = DEFAULT_DECISION_PROMPT
    max_new_tokens: int = 256
    temperature: float = 0.0  # deterministic
    top_p: float = 1.0

# -------------------------- JSON utils --------------------------

_CODE_FENCE_RE = re.compile(r"```.*?```", re.S)

MARKERS = ["\n```", "```", "\nJSON:", "\nCode:", "\nExample:", "\nBeispiel:", "\nOutput:"]

def _truncate_at_markers(s: str) -> str:
    for m in MARKERS:
        i = s.find(m)
        if i != -1:
            return s[:i]
    return s

def _clean_explanation(s: str) -> str:
    if not s:
        return "No explanation provided."
    s = _CODE_FENCE_RE.sub("", s)
    s = re.sub(r"`{1,3}", "", s)
    s = re.sub(r"(?:^|\s)(JSON:|Note:|Hinweis:|Beispiel:).*", "", s, flags=re.I | re.S)
    s = re.sub(r"\s+", " ", s).strip()
    parts = re.split(r'(?<=[.!?])\s+', s)
    s = " ".join(parts[:4]).strip()
    return s or "No explanation provided."

# Robust JSON extractor: remove code fences, then try from the last '{' backwards

def safe_json_extract(text: str) -> Dict[str, Any]:
    s = _truncate_at_markers(text.strip())
    s = _CODE_FENCE_RE.sub("", s)
    last = s.rfind("{")
    while last != -1:
        try:
            obj = json.loads(s[last:])
            verdict = "True" if obj.get("verdict") == "True" else "False"
            expl = _clean_explanation(str(obj.get("explanation", "")))
            return {"verdict": verdict, "explanation": expl}
        except Exception:
            last = s.rfind("{", 0, last)
    # conservative fallback
    fallback_verdict = "True" if re.search(r'\b"verdict"\s*:\s*"True"\b', s) else "False"
    return {"verdict": fallback_verdict, "explanation": _clean_explanation(s)}

# -------------------------- Model utilities --------------------------

_MODEL_CACHE: Dict[Tuple[str, str], Tuple[AutoTokenizer, AutoModelForCausalLM]] = {}

def get_model_and_tokenizer(model_name: str, mixed_precision: Optional[str], accelerator: Accelerator):
    """Cache by (model_name, mixed_precision). Place on accelerator.device."""
    key = (model_name, mixed_precision or "none")
    if key in _MODEL_CACHE:
        tok, mdl = _MODEL_CACHE[key]
        return tok, mdl

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    torch_dtype = None
    if mixed_precision == "fp16":
        torch_dtype = torch.float16
    elif mixed_precision == "bf16":
        torch_dtype = torch.bfloat16

    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    mdl.to(accelerator.device)
    mdl.eval()

    _MODEL_CACHE[key] = (tok, mdl)
    return tok, mdl

@torch.inference_mode()
def generate_batched(model, tokenizer, prompts: List[str], max_new_tokens: int, temperature: float, top_p: float) -> List[str]:
    """Generate outputs for a batch of prompts on the current device.
    Returns a list of decoded completions (WITHOUT the prompt prefix).
    """
    # Tokenize with padding to longest in batch
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    # Generate
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Slice off the prompt tokens for each item
    input_lens = (enc["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
    outputs = []
    for i in range(gen.size(0)):
        gen_ids = gen[i, input_lens[i] :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        outputs.append(text)
    return outputs

# -------------------------- Prompts --------------------------

def build_expert_prompt(template: str, statement: str, subject: Optional[str]) -> str:
    subject_block = f"Subject(s): {subject}" if subject else "Subject(s): (none/provided)"
    return template.format(statement=statement, subject_block=subject_block)

# -------------------------- Pipeline core --------------------------

@dataclass
class RoleRuntime:
    name: str
    prompt_template: str
    weight: float
    max_new_tokens: int
    temperature: float
    top_p: float


def run_pipeline(
    accelerator: Accelerator,
    df: pd.DataFrame,
    model_name: str,
    expert_roles: List[RoleRuntime],
    decision_cfg: DecisionConfig,
    statement_col: str,
    subject_col: Optional[str],
    batch_size: int,
    early_exit_margin: Optional[float] = None,
) -> Dict[str, Any]:
    """Runs the full batched multi-agent pipeline on this process' shard of the data.
    Returns a dict with local results (indices, predictions, explanations, optional experts JSON).
    """
    # Split indices across processes
    all_indices = list(range(len(df)))
    local_indices = accelerator.split_between_processes(all_indices)

    # Load (shared) model/tokenizer once per process
    tok, mdl = get_model_and_tokenizer(model_name, accelerator.mixed_precision, accelerator)

    # Containers for this process' results
    local_results = {
        "idx": [],
        "prediction": [],
        "explanation": [],
        "experts": [],  # list of per-sample expert dicts
    }

    # Helper to create batches of local indices
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    for batch_idx in chunks(local_indices, batch_size):
        # Gather inputs for this batch
        statements = df.loc[batch_idx, statement_col].tolist()
        subjects = (
            df.loc[batch_idx, subject_col].tolist() if subject_col and subject_col in df.columns else [None] * len(batch_idx)
        )

        # 1) Experts: run per role, but batched over samples; collect JSONs
        per_role_outputs: List[List[Dict[str, Any]]] = []  # [role][sample_idx] -> {verdict, explanation}

        for role in expert_roles:
            prompts = [build_expert_prompt(role.prompt_template, s, subj) for s, subj in zip(statements, subjects)]
            completions = generate_batched(
                mdl,
                tok,
                prompts,
                max_new_tokens=role.max_new_tokens,
                temperature=role.temperature,
                top_p=role.top_p,
            )
            jsons = [safe_json_extract(c) for c in completions]
            per_role_outputs.append(jsons)

        # Build expert packets per sample
        expert_packets_per_sample: List[List[Dict[str, Any]]] = []
        for j in range(len(batch_idx)):
            packets = []
            for role, json_out in zip(expert_roles, (col[j] for col in per_role_outputs)):
                verdict = "True" if json_out.get("verdict") == "True" else "False"
                packets.append(
                    {
                        "name": role.name,
                        "weight": float(role.weight),
                        "verdict": verdict,
                        "explanation": json_out.get("explanation", ""),
                    }
                )
            expert_packets_per_sample.append(packets)

        # Optional early exit: if strong consensus, skip decision LLM
        skip_decision_mask = [False] * len(batch_idx)
        consensus_verdicts = [None] * len(batch_idx)
        if early_exit_margin is not None:
            for j, packets in enumerate(expert_packets_per_sample):
                score = sum(p["weight"] * (1 if p["verdict"] == "True" else -1) for p in packets)
                if abs(score) >= early_exit_margin:
                    skip_decision_mask[j] = True
                    consensus_verdicts[j] = "True" if score > 0 else "False"

        # 2) Decision: build batched prompts only for the ones not skipped
        decision_prompts = []
        decision_positions = []  # map back into batch order
        for j, packets in enumerate(expert_packets_per_sample):
            if skip_decision_mask[j]:
                continue
            experts_json = json.dumps(packets, ensure_ascii=False)
            dec_prompt = decision_cfg.prompt_template.format(experts_json=experts_json)
            decision_prompts.append(dec_prompt)
            decision_positions.append(j)

        decision_outputs = [None] * len(batch_idx)
        if decision_prompts:
            dec_completions = generate_batched(
                mdl,
                tok,
                decision_prompts,
                max_new_tokens=decision_cfg.max_new_tokens,
                temperature=decision_cfg.temperature,
                top_p=decision_cfg.top_p,
            )
            dec_jsons = [safe_json_extract(c) for c in dec_completions]
            for pos, js in zip(decision_positions, dec_jsons):
                decision_outputs[pos] = js

        # 3) Consolidate final outputs for this batch
        for j, global_i in enumerate(batch_idx):
            if skip_decision_mask[j]:
                final_verdict = consensus_verdicts[j]
                final_expl = "Early exit by weighted consensus."
            else:
                js = decision_outputs[j] or {"verdict": "False", "explanation": "No decision."}
                final_verdict = "True" if js.get("verdict") == "True" else "False"
                final_expl = js.get("explanation", "")

            local_results["idx"].append(global_i)
            local_results["prediction"].append(final_verdict)
            local_results["explanation"].append(final_expl)
            local_results["experts"].append(expert_packets_per_sample[j])

    return local_results

# -------------------------- CLI and main --------------------------

EXPERT_ROLES_PRESET = [
    ExpertConfig(
        name="Politics&Elections SME",
        weight=1.4,
        max_new_tokens=112,
        temperature=0.2,
        top_p=1.0,
        prompt_template='''You are a subject-matter expert in U.S. politics, elections, legislation, and public records.

Task: Verify the claim with objective, publicly verifiable facts (official roll calls, bill texts, FEC/CRP, governor/congress records, reputable outlets). If evidence is insufficient/ambiguous, output "False".

Return ONLY this JSON:
{{"verdict":"True|False","explanation":"2-4 explaining sentences"}}

Focus domains (if present): elections, campaign-finance, voting records, state/federal legislation, party platforms.

Input:
Claim: {statement}
{subject_block}
JSON:''' ,
    ),
    ExpertConfig(
        name="Economy&Taxes SME",
        weight=1.3,
        max_new_tokens=112,
        temperature=0.2,
        top_p=1.0,
        prompt_template='''You are a subject-matter expert in macroeconomics, labor stats, budgets, and taxation.

Task: Verify using BLS, BEA, CBO, JCT, Treasury/IRS, OMB, WTO/IMF/World Bank, and reputable reports. Quantify time ranges and definitions. If data is unclear or cherry-picked, output "False".

Return ONLY this JSON:
{{"verdict":"True|False","explanation":"2-4 explaining sentences"}}

Focus domains: taxes, jobs, economy, trade, social security/medicare finances, government budget.

Input:
Claim: {statement}
{subject_block}
JSON:''',
    ),
    ExpertConfig(
        name="Health&Science SME",
        weight=1.3,
        max_new_tokens=112,
        temperature=0.2,
        top_p=1.0,
        prompt_template='''You are a subject-matter expert for healthcare policy, public health, and scientific claims.

Task: Verify with CDC, FDA, NIH, CMS, WHO, Cochrane, PubMed-indexed studies, and official rulemaking. Distinguish correlation vs. causation and specify years/locations. If evidence is weak, output "False".

Return ONLY this JSON:
{{"verdict":"True|False","explanation":"2-4 explaining sentences"}}

Focus domains: health-care, ACA/Medicare/Medicaid, vaccines, drugs/overdoses, abortion/stem cells, general science.

Input:
Claim: {statement}
{subject_block}
JSON:''',
    ),
]

DECISION_PRESET = DecisionConfig()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, required=True, help="Input dataset path (TSV by default).")
    p.add_argument("--save_path", type=str, required=True, help="Where to write results (TSV).")
    p.add_argument("--sep", type=str, default="\t", help="CSV separator. Default: TAB (TSV)")
    p.add_argument("--statement_col", type=str, default="statement")
    p.add_argument("--subject_col", type=str, default="subjects")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--early_exit_margin", type=float, default=None, help="If set, skip decision when |score| >= margin.")
    return p.parse_args()


def main():
    args = parse_args()

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print("\n>>> Launch config:")
        print(vars(args))
        print(f"Accelerate device: {accelerator.device}, mixed_precision={accelerator.mixed_precision}")

    # Load data
    df = pd.read_csv(args.csv_path, sep=args.sep, quotechar='"', engine="python", dtype=str)
    if args.limit:
        df = df.head(args.limit).copy()

    # Prepare roles
    roles: List[RoleRuntime] = [
        RoleRuntime(
            name=cfg.name,
            prompt_template=cfg.prompt_template,
            weight=cfg.weight,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        for cfg in EXPERT_ROLES_PRESET
    ]

    # Run pipeline on local shard
    local = run_pipeline(
        accelerator=accelerator,
        df=df,
        model_name=args.model_name,
        expert_roles=roles,
        decision_cfg=DECISION_PRESET,
        statement_col=args.statement_col,
        subject_col=args.subject_col,
        batch_size=args.batch_size,
        early_exit_margin=args.early_exit_margin,
    )

    # Gather results to main process
    gathered = accelerator.gather_object(local)

    if accelerator.is_main_process:
        # Flatten
        idx_all, pred_all, expl_all, experts_all = [], [], [], []
        for part in gathered:
            idx_all.extend(part["idx"])  # type: ignore
            pred_all.extend(part["prediction"])  # type: ignore
            expl_all.extend(part["explanation"])  # type: ignore
            experts_all.extend(part["experts"])  # type: ignore

        # Reorder by original index
        order = sorted(range(len(idx_all)), key=lambda k: idx_all[k])
        pred_all = [pred_all[i] for i in order]
        expl_all = [expl_all[i] for i in order]
        experts_all = [experts_all[i] for i in order]

        # Attach to dataframe
        df_out = df.copy()
        df_out["prediction"] = pred_all
        df_out["explanation"] = expl_all
        df_out["experts"] = [json.dumps(x, ensure_ascii=False) for x in experts_all]

        # Metrics if labels exist
        metrics = {}
        if args.label_col in df_out.columns:
            y_true = df_out[args.label_col]
            y_pred = df_out["prediction"]
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision_true": float(precision_score(y_true, y_pred, pos_label="True", zero_division=0)),
                "recall_true": float(recall_score(y_true, y_pred, pos_label="True", zero_division=0)),
                "f1_true": float(f1_score(y_true, y_pred, pos_label="True", zero_division=0)),
                "report": classification_report(y_true, y_pred, labels=["True", "False"], zero_division=0),
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["True", "False"]).tolist(),
            }
            print("\n📊 Evaluation")
            for k, v in metrics.items():
                if k != "report":
                    print(f"{k}: {v}")
            print("\nClassification report:\n", metrics["report"])  # noqa

        # Save
        # Ensure directory exists
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        df_out.to_csv(args.save_path, sep="\t", index=False)
        print(f"\n💾 Saved: {args.save_path}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
