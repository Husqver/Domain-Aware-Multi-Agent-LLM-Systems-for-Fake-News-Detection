from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch
import os
import gc


def fine_tune_lora_model(
    model_name: str,
    dataset_path: str,
    training_args_dict: dict,
    output_dir: str = "./results",
    log_dir: str = "./logs",
    max_length: int = 512,
    max_steps: int = 1000,
    target_modules: list = ["q_proj", "v_proj"],
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    cuda_devices: str = "0",
):
    # CUDA-Gerät festlegen
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer laden
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Datensatz laden
    dataset = load_dataset("json", data_files={"train": dataset_path})["train"]

    # Tokenisierung (verwende nur das vorbereitete 'text'-Feld)
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    tokenized_dataset = dataset.map(tokenize, batched=True)

    # Modell laden
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # LoRA-Konfiguration
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, peft_config)

    # Trainingsargumente setzen
    training_args_dict["max_steps"] = max_steps
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=log_dir,
        **training_args_dict
    )

    # Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer initialisieren
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training starten
    trainer.train()

    # Modell + Tokenizer speichern
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Speicher bereinigen
    torch.cuda.empty_cache()
    gc.collect()

    print(f"✅ Training abgeschlossen. Modell gespeichert unter: {output_dir}")
    return model

import torch
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import seaborn as sns
import matplotlib.pyplot as plt
    
def evaluate_model_on_statements(
    model_name: str,
    test_data_path: str,
    prompt_template: str,
    cuda_device: str = "0",
    limit: int = 2000,
    use_subjects: bool = False,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prompt-Builder mit optionalen Subjects
    def build_input(statement, subjects=None):
            return prompt_template.format(statement=statement, subjects=subjects)

    # Daten laden
    df = pd.read_csv(test_data_path)
    df = df.iloc[:limit].copy()

    print(f"\n📦 Lade Modell: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    # Token IDs für " True" und " False"
    true_token_id = tokenizer(" True", add_special_tokens=False)["input_ids"][0]
    false_token_id = tokenizer(" False", add_special_tokens=False)["input_ids"][0]

    predictions = []
    confidences = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Verarbeitung"):
        statement = row["statement"]
        subjects = row["subjects"] if use_subjects else None
        prompt = build_input(statement, subjects)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)

        prob_true = probs[true_token_id].item()
        prob_false = probs[false_token_id].item()

        if prob_true > prob_false:
            predictions.append("True")
            confidences.append(prob_true)
        else:
            predictions.append("False")
            confidences.append(prob_false)

    # Ergebnisse speichern
    suffix = model_name.split("/")[-1]
    df[f"label_{suffix}"] = predictions
    df[f"confidence_{suffix}"] = confidences

    # Ground truth normalisieren
    true_labels = ["true", "mostly-true", "half-true"]
    false_labels = ["barely-true", "false", "pants-fire", "pants-on-fire"]

    def map_label(label):
        label = str(label).strip().lower()
        if label in true_labels:
            return "True"
        elif label in false_labels:
            return "False"
        return "False"

    y_true = df["label"].apply(map_label)
    y_pred = df[f"label_{suffix}"].apply(map_label)

    # Evaluation
    print("\n📊 Evaluation:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=["True", "False"]))
    print(f"\nAccuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred, pos_label='True'):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred, pos_label='True'):.3f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, pos_label='True'):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["True", "False"]))

    # Plot
    df["label_true"] = y_true
    df["correct"] = df["label_true"] == y_pred

    sns.histplot(
        data=df,
        x=f"confidence_{suffix}",
        hue="correct",
        bins=20,
        kde=True,
        palette=["red", "green"]
    )
    plt.title(f"Confidence Distribution ({suffix})")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.show()

    return df


import pandas as pd
import json

def generate_prompt_jsonl_from_csv(
    csv_path: str,
    prompt_template: str,
    output_path: str,
    use_subjects: bool = False,
):
    # CSV laden
    df = pd.read_csv(csv_path, header=0, sep="\t",          # Tab als Trenner
                     quotechar='"',     # Äußere Anführungszeichen entfernen
                     engine="python",   # flexiblerer Parser
                     dtype=str
                    )


    jsonl_data = []
    for _, row in df.iterrows():
        statement = row["statement"]
        label = row["label"]
        subjects = row.get("subjects", "")

        if use_subjects and subjects:
            prompt_text = prompt_template.format(
                statement=statement,
                subjects=subjects,
                label=label
            )
        else:
            prompt_text = prompt_template.format(
                statement=statement,
                label=label
            )

        jsonl_data.append({"text": prompt_text})

    # JSONL speichern
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ {len(jsonl_data)} Beispiele gespeichert in '{output_path}'")


