# expert_config.py
from dataclasses import dataclass
from typing import List


@dataclass
class ExpertConfig:
    """
    Hält alle Einstellungen für die Fact-Checking-Experten (LoRA-Modelle).
    Du gibst hier nur an, wie deine Daten aussehen und welche Domains du trainieren willst.
    """
    super_domains: List[str]              # z.B. ["socioeconomic_policy", "foreign_security", ...]
    base_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Spaltennamen im Trainings-DataFrame:
    text_column: str = "statement"        # Claim-Text
    label_column: str = "label"           # "True"/"False"
    domain_column: str = "super_domain"   # Superdomain je Zeile

    max_length: int = 256                 # Max Token-Länge für Input
    num_train_epochs: int = 3
    learning_rate: float = 5e-4
    per_device_train_batch_size: int = 4

    def build_system_prompt(self, domain_name: str) -> str:
        """
        Domain-spezifischer Systemprompt für den jeweiligen Experten.
        Kannst du bei Bedarf später anpassen/überschreiben.
        """
        return f"""You are a fact-checking expert specialized in the '{domain_name}' domain.

You receive short political or public policy claims (statements) and must decide if they are factually correct.

Answer STRICTLY with one of these two labels:
- True  (the claim is factually correct)
- False (the claim is factually incorrect)

Do NOT explain. Do NOT add any extra text. Only output 'True' or 'False'.
"""
