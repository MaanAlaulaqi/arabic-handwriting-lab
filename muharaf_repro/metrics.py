import json
from pathlib import Path

from jiwer import cer, wer
from rapidfuzz.distance import Levenshtein

from . import MARKS


def read_predictions(path: Path, entries):
    expected = {x["image"].name: x["reference"] for x in entries}
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            name = row["image_name"]
            if name in rows or name not in expected or row["reference"] != expected[name]:
                raise ValueError(f"Duplicate, unknown, or reference-mismatched prediction: {name}")
            rows[name] = row
    return rows


def metrics(entries, rows):
    if cer(["x"], ["x   "]) != 0:
        raise RuntimeError("Installed JiWER no longer matches the notebook's edge-stripping CER rule")
    names = [x["image"].name for x in entries]
    if set(rows) != set(names):
        raise ValueError(f"Incomplete predictions: {len(rows)} of {len(names)}")
    refs = [x["reference"] for x in entries]
    hyps = [rows[name]["prediction"] for name in names]
    without = lambda s: "".join(ch for ch in s if ch not in MARKS)
    matched = sum(
        sum(ch in MARKS for ch in ref[a:b])
        for ref, hyp in zip(refs, hyps)
        for tag, a, b, _, _ in Levenshtein.opcodes(ref, hyp)
        if tag == "equal"
    )
    ref_marks = sum(ch in MARKS for s in refs for ch in s)
    hyp_marks = sum(ch in MARKS for s in hyps for ch in s)
    return {
        "jiwer_default_cer": cer(refs, hyps),
        "jiwer_default_wer": wer(refs, hyps),
        "jiwer_cer_without_basic_harakat": cer(list(map(without, refs)), list(map(without, hyps))),
        "literal_cer": sum(Levenshtein.distance(a, b) for a, b in zip(refs, hyps)) / sum(map(len, refs)),
        "reference_basic_harakat": ref_marks,
        "predicted_basic_harakat": hyp_marks,
        "matched_basic_harakat": matched,
        "unmatched_reference_basic_harakat": ref_marks - matched,
        "unmatched_predicted_basic_harakat": hyp_marks - matched,
        "trailing_spaces": sum(len(h) - len(h.rstrip(" ")) for h in hyps),
    }


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
