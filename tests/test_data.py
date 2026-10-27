import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from muharaf_repro.data import disjoint, prepare_image, read_split


class DataContractTests(unittest.TestCase):
    def test_preparation_dimensions_and_left_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "line.png"
            Image.new("RGB", (20, 10), (0, 0, 0)).save(path)
            result = prepare_image(path)
            self.assertEqual(result.size, (512, 102))
            self.assertEqual(result.getpixel((0, 0)), (0, 0, 0))
            self.assertEqual(result.getpixel((500, 50)), (255, 255, 255))

    def test_manifest_uses_original_txt_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "a.png"
            image.write_bytes(b"not decoded by read_split")
            image.with_suffix(".txt").write_text("عليّ", encoding="utf-8")
            manifest = root / "train.jsonl"
            manifest.write_text(json.dumps({"image_path": "elsewhere/a.png", "text": "عليّ"}, ensure_ascii=False) + "\n")
            with patch("muharaf_repro.data.COUNTS", {"train": 1}):
                self.assertEqual(read_split(root, "train", {"a.png": image})[0]["reference"], "عليّ")
                manifest.write_text(json.dumps({"image_path": "elsewhere/a.png", "text": "على"}, ensure_ascii=False) + "\n")
                with self.assertRaises(ValueError):
                    read_split(root, "train", {"a.png": image})

    def test_line_overlap_rejected(self):
        line = {"image": Path("a.png")}
        with self.assertRaises(ValueError):
            disjoint([line], [line])


if __name__ == "__main__":
    unittest.main()
