"""Reusable inference layer for the stress text classifier."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_FILE = ROOT / "model" / "stress_pipeline.joblib"
MIN_TEXT_LENGTH = 3

# Scores between these two values are treated as "unclear".
# Justify these numbers in the viva with the output of analysis.py.
AMBIGUOUS_LOW = 0.35
AMBIGUOUS_HIGH = 0.65

QUESTION_STARTS = (
    "what ", "why ", "how ", "should ", "can i ", "could i ", "should i ", "is it ",
)


@dataclass(frozen=True)
class Prediction:
    label: int
    result: str
    stress_probability: float
    confidence: float
    level: str
    needs_more_context: bool
    input_note: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class StressDetector:
    """Load a saved classifier and turn its output into cautious user language."""

    def __init__(self, model_file: str | Path = DEFAULT_MODEL_FILE) -> None:
        self.model_file = Path(model_file)
        if not self.model_file.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_file}. Run train_model.py first."
            )
        self.pipeline = joblib.load(self.model_file)

    @staticmethod
    def validate_text(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("Text must be a string.")
        cleaned = " ".join(text.split())
        if len(cleaned) < MIN_TEXT_LENGTH:
            raise ValueError("Please enter at least three characters.")
        return cleaned

    @staticmethod
    def looks_like_general_question(text: str) -> bool:
        lower = text.lower().strip()
        return "?" in lower or lower.startswith(QUESTION_STARTS)

    def predict(self, text: str) -> Prediction:
        cleaned = self.validate_text(text)
        probabilities = self.pipeline.predict_proba([cleaned])[0]
        classes = list(self.pipeline.classes_)
        stress_probability = float(probabilities[classes.index(1)])
        raw_label = int(self.pipeline.predict([cleaned])[0])
        confidence = max(stress_probability, 1.0 - stress_probability)
        short = len(cleaned.split()) < 5

        # The classifier only knows stress vs non-stress text. Questions and
        # borderline scores are reported as "unclear" instead of a false claim.
        if self.looks_like_general_question(cleaned):
            result, level, needs_more_context = "unclear", "no clear stress signal", True
            note = ("This looks like a question. The model detects stress patterns "
                    "in statements, not answers to questions.")
        elif AMBIGUOUS_LOW <= stress_probability <= AMBIGUOUS_HIGH:
            result, level, needs_more_context = "unclear", "some indication", True
            note = ("The score is close to the middle, so the model cannot "
                    "decide confidently.")
        elif stress_probability > AMBIGUOUS_HIGH:
            result, level, needs_more_context = "stress-related patterns detected", "higher indication", short
            note = "The text contains patterns seen more often in the stress-related training examples."
        else:
            result, level, needs_more_context = "no clear stress signal", "lower indication", short
            note = "The text does not strongly match the stress-related training examples."

        return Prediction(
            label=raw_label,
            result=result,
            stress_probability=round(stress_probability, 4),
            confidence=round(confidence, 4),
            level=level,
            needs_more_context=needs_more_context,
            input_note=note,
        )
