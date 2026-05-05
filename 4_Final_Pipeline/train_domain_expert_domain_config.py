# domain_config.py
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional


@dataclass
class DomainConfig:
    """
    Holds the subject->superlabel mapping and prompt construction.

    domain_to_super:
        mapping from fine-grained LIAR subject tags (e.g., "taxes") to a super label (e.g., "economy")
    super_labels:
        explicit ordering of super labels (also used as priority if multiple subjects map to multiple super labels)
    """
    domain_to_super: Dict[str, str]
    super_labels: Optional[List[str]] = None
    super_label_descriptions: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.super_labels is None:
            s: Set[str] = set(self.domain_to_super.values())
            self.super_labels = sorted(s)

    def map_subjects_to_super(self, subjects_str: str) -> str:
        """
        Map LIAR 'subjects' (comma-separated tags) to exactly one super label.
        If multiple are present, resolve by priority order of self.super_labels.
        Fallback: 'misc' if present, else first label.
        """
        if not isinstance(subjects_str, str) or not subjects_str.strip():
            return "misc" if "misc" in self.super_labels else self.super_labels[0]

        domains = [s.strip() for s in subjects_str.split(",") if s.strip()]
        mapped = {self.domain_to_super.get(d, "misc") for d in domains}

        for label in self.super_labels:
            if label in mapped:
                return label

        return "misc" if "misc" in self.super_labels else self.super_labels[0]

    def build_system_prompt(self) -> str:
        """
        Strict router system prompt listing valid labels.
        """
        lines = [
            "You are a strict domain classifier.\n",
            "You receive a short political or public policy claim (a 'statement') "
            "and must map it to exactly ONE high-level domain label from this list:\n",
        ]

        for label in self.super_labels:
            desc = self.super_label_descriptions.get(label, "").strip()
            if desc:
                lines.append(f"- {label}: {desc}")
            else:
                lines.append(f"- {label}")

        lines.append("\nYou MUST answer with EXACTLY ONE label string from this list.")
        lines.append("Do NOT explain. Only output the label.")
        return "\n".join(lines)