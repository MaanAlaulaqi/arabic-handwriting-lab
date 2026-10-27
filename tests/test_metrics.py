import unittest
from pathlib import Path

from muharaf_repro.metrics import metrics


class MetricContractTests(unittest.TestCase):
    def test_trailing_spaces_change_literal_but_not_jiwer_default(self):
        entries = [{"image": Path("a.png"), "reference": "abc"}]
        rows = {"a.png": {"image_name": "a.png", "reference": "abc", "prediction": "abc    "}}
        result = metrics(entries, rows)
        self.assertEqual(result["jiwer_default_cer"], 0)
        self.assertEqual(result["literal_cer"], 4 / 3)
        self.assertEqual(result["trailing_spaces"], 4)


if __name__ == "__main__":
    unittest.main()
