"""Run with: python -m unittest test_backend -v"""

import unittest

from backend import DEFAULT_MODEL_FILE, StressDetector


class TextRuleTests(unittest.TestCase):
    """These need no trained model."""

    def test_empty_text_rejected(self):
        with self.assertRaises(ValueError):
            StressDetector.validate_text("   ")

    def test_too_short_rejected(self):
        with self.assertRaises(ValueError):
            StressDetector.validate_text("hi")

    def test_non_string_rejected(self):
        with self.assertRaises(TypeError):
            StressDetector.validate_text(123)

    def test_whitespace_normalised(self):
        self.assertEqual(StressDetector.validate_text("  a   b\n c "), "a b c")

    def test_question_detected(self):
        self.assertTrue(StressDetector.looks_like_general_question("How do I study?"))
        self.assertFalse(StressDetector.looks_like_general_question("I feel overwhelmed."))

    def test_missing_model_file(self):
        with self.assertRaises(FileNotFoundError):
            StressDetector("does_not_exist.joblib")


@unittest.skipUnless(DEFAULT_MODEL_FILE.exists(), "train the model first")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = StressDetector()

    def test_output_is_valid(self):
        p = self.detector.predict("I have been feeling anxious about my exams and cannot sleep at night.")
        self.assertTrue(0.0 <= p.stress_probability <= 1.0)
        self.assertIn(p.result, {"stress-related patterns detected", "no clear stress signal", "unclear"})

    def test_question_is_unclear(self):
        self.assertEqual(self.detector.predict("How do I study better?").result, "unclear")


if __name__ == "__main__":
    unittest.main()
