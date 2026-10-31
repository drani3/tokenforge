"""Run with: python -m unittest discover tests"""
import unittest

from lab.scorers import extract_answer, numeric_match


class ExtractAnswerTest(unittest.TestCase):
    def test_format_line(self):
        self.assertEqual(extract_answer("12 + 2 = 14\nAnswer: 14"), (14.0, "format"))

    def test_last_answer_wins_when_model_corrects_itself(self):
        self.assertEqual(extract_answer("Answer: 12 oranges... wait, recount. Answer: 15"), (15.0, "format"))

    def test_markdown_and_currency(self):
        self.assertEqual(extract_answer("**Answer:** $1,250.50"), (1250.5, "format"))
        self.assertEqual(extract_answer("**Answer: 57.5%**"), (57.5, "format"))

    def test_prose_answer_word_falls_back_to_last_number(self):
        text = "here's the answer: we have 12 oranges and 4 apples, we eat 2 fruits, now we have 14 fruits"
        self.assertEqual(extract_answer(text), (14.0, "fallback"))

    def test_fallback_known_weakness_trailing_number(self):
        # Documented limitation: without the format, a trailing number that isn't the answer wins.
        self.assertEqual(extract_answer("So we have 14 fruits. That took 3 steps."), (3.0, "fallback"))

    def test_negative_and_no_number(self):
        self.assertEqual(extract_answer("Answer: -7"), (-7.0, "format"))
        self.assertEqual(extract_answer("I don't know"), (None, "none"))


class NumericMatchTest(unittest.TestCase):
    def test_strict_requires_format(self):
        r = numeric_match("now we have 14 fruits", 14)
        self.assertEqual((r["score"], r["strict_score"]), (1.0, 0.0))

    def test_strict_and_flexible_agree_with_format(self):
        r = numeric_match("Answer: 14", 14)
        self.assertEqual((r["score"], r["strict_score"]), (1.0, 1.0))

    def test_wrong_answer(self):
        r = numeric_match("Answer: 13", 14)
        self.assertEqual((r["score"], r["strict_score"]), (0.0, 0.0))

    def test_float_answer(self):
        self.assertEqual(numeric_match("Answer: 32.5", 32.5)["score"], 1.0)
        self.assertEqual(numeric_match("Answer: 32.50", 32.5)["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
