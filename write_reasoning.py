import os

REFRACTOR_DIR = "refactor"
FILE_NAME = "reasoning.py"

os.makedirs(REFRACTOR_DIR, exist_ok=True)

code = """\
\"\"\"
Symbolic Reasoning Engine for Sully
Handles multiple reasoning styles and frameworks.
\"\"\"

from typing import Any, Dict, Optional, Callable
import random


class SymbolicReasoningNode:
    \"\"\"
    A reasoning engine that simulates multiple modes of cognition.
    \"\"\"

    def __init__(self):
        self.reasoning_styles = {
            "analytical": self._analytical_mode,
            "creative": self._creative_mode,
            "critical": self._critical_mode,
            "ethereal": self._ethereal_mode,
            "emergent": self._emergent_mode,
        }

    def reason(self, input_text: str, style: str = "analytical") -> str:
        \"\"\"
        Process input using the specified reasoning style.

        Args:
            input_text: The input to reason over.
            style: The reasoning style to use.

        Returns:
            Generated reasoning response.
        \"\"\"
        strategy = self.reasoning_styles.get(style.lower())
        if strategy:
            return strategy(input_text)
        return f"[Error] Unknown reasoning style: {style}"

    def _analytical_mode(self, input_text: str) -> str:
        return f"[Analytical] Logic chain started → Processing → Result: {input_text}"

    def _creative_mode(self, input_text: str) -> str:
        return f"[Creative] Imagine if {input_text} existed in a world of dragons and lasers..."

    def _critical_mode(self, input_text: str) -> str:
        return f"[Critical] Let's deconstruct the premise of: {input_text}"

    def _ethereal_mode(self, input_text: str) -> str:
        return f"[Ethereal] The essence of {input_text} resonates with unseen dimensions."

    def _emergent_mode(self, input_text: str) -> str:
        fragments = input_text.split()
        random.shuffle(fragments)
        return f"[Emergent] {' '.join(fragments)}"

"""

file_path = os.path.join(REFRACTOR_DIR, FILE_NAME)
with open(file_path, "w", encoding="utf-8") as f:
    f.write(code)

print(f"[✓] Wrote {FILE_NAME} to '{REFRACTOR_DIR}/'")