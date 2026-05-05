# expert_trainer.py
import os
import torch
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

from train_expert_expert_config import ExpertConfig


# === Environment-Settings (optional wie bei dir) ===
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # sichtbare GPUs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ===== Callback für schöne tqdm-Ausgabe (optional) =====
class TQDMProgressCallback(TrainerCallback):
    def __init__(self):
        self.progress_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        if state.max_steps is None or state.max_steps == 0:
            return
        self.progress_bar = tqdm(
            total=state.max_steps,
            desc="🚀 Training",
            dynamic_ncols=True
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self.progress_bar is not None:
            self.progress_bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        if self.progress_bar is not None:
            self.progress_bar.close()


def create_tokenizer(base_model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

from jinja2.exceptions import TemplateError

def render_chat_or_plain(tokenizer, system_prompt: str, user_prompt: str, assistant_text: str | None):
    """
    Returns ONE training text string (prompt + optional assistant target).
    Works across:
      - Llama-Instruct (chat template supports system)
      - Some Gemma templates (may reject system role -> fallback)
      - Base LMs without templates (DeepSeek base) -> plain fallback
    """
    has_chat = hasattr(tokenizer, "apply_chat_template") and bool(getattr(tokenizer, "chat_template", None))

    if has_chat:
        # Try standard chat roles first
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if assistant_text is not None:
            msgs.append({"role": "assistant", "content": assistant_text})

        try:
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except (TemplateError, Exception):
            # Fallback: merge system into user (some templates reject "system")
            merged_user = f"{system_prompt}\n\n{user_prompt}"
            msgs2 = [{"role": "user", "content": merged_user}]
            if assistant_text is not None:
                msgs2.append({"role": "assistant", "content": assistant_text})
            return tokenizer.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=False)

    # Plain fallback (no chat template)
    if assistant_text is None:
        return f"{system_prompt}\n\n{user_prompt}\n\n"
    return f"{system_prompt}\n\n{user_prompt}\n\n{assistant_text}\n"



def build_chat_tokenize_fn(domain_name: str, config: ExpertConfig, tokenizer):
    system_prompt = config.build_system_prompt(domain_name)

    def tokenize_example(example):
        claim = example[config.text_column]
        label = example[config.label_column]  # "True"/"False"

        user_content = (
            f"Claim:\n{claim}\n\n"
            "Is this claim factually correct? Answer strictly with 'True' or 'False'."
        )

        text = render_chat_or_plain(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_content,
            assistant_text=label,   # teacher forcing target
        )

        # Optional: EOS helps some models learn clean endings
        if tokenizer.eos_token:
            text = text + tokenizer.eos_token

        return tokenizer(
            text,
            max_length=config.max_length,
            truncation=True,
        )

    return tokenize_example



def train_expert_for_domain(
    domain_name: str,
    df_domain: pd.DataFrame,
    config: ExpertConfig,
    output_root: str,
):
    """
    Trainiert EINEN Fact-Checking-Experten (LoRA) für EINE Superdomain.
    df_domain enthält nur Zeilen für diese Domain.
    """

    if df_domain.empty:
        print(f"⚠️  Domain '{domain_name}' hat keine Trainingsdaten – überspringe.")
        return None

    print(f"\n=== Training expert for domain: {domain_name} ===")
    print(f"Samples: {len(df_domain)}")

    # 1) Tokenizer
    tokenizer = create_tokenizer(config.base_model_id)

    # 2) Dataset bauen
    ds = Dataset.from_pandas(df_domain[[config.text_column, config.label_column]])

    # 3) Tokenisierungsfunktion (Chat-Template)
    tokenize_fn = build_chat_tokenize_fn(domain_name, config, tokenizer)
    train_tok = ds.map(
        tokenize_fn,
        remove_columns=ds.column_names,
    )

    # 4) Basis-Modell + LoRA
    base = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=None,             
        low_cpu_mem_usage=False, 
    )

    base = base.to(DEVICE)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 5) TrainingArguments
    out_dir = os.path.join(output_root, domain_name)
    os.makedirs(out_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=25,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        data_collator=collator,
        callbacks=[TQDMProgressCallback()],
    )

    trainer.train()

    # 6) Speichern
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"✔️ Expert-LoRA for '{domain_name}' saved to {out_dir}")
    return out_dir


def train_all_experts(
    df: pd.DataFrame,
    config: ExpertConfig,
    output_root: str,
):
    """
    Nimmt EINEN großen Trainings-DataFrame und trainiert für jede Superdomain ein eigenes LoRA-Modell.

    Erwartet:
      - df[config.text_column]  (z.B. 'statement')
      - df[config.label_column] (z.B. 'label', mit 'True'/'False')
      - df[config.domain_column] (z.B. 'super_domain', mit Werten aus config.super_domains)
    """

    os.makedirs(output_root, exist_ok=True)
    saved_paths = {}

    for domain_name in config.super_domains:
        df_dom = df[df[config.domain_column] == domain_name]
        path = train_expert_for_domain(
            domain_name=domain_name,
            df_domain=df_dom,
            config=config,
            output_root=output_root,
        )
        saved_paths[domain_name] = path

    return saved_paths
