"""Score published JSONL predictions without loading model weights or images."""
import argparse
import json
from pathlib import Path

from muharaf_repro.metrics import metrics


def main():
    parser = argparse.ArgumentParser(description="Verify frozen and selected saved test predictions")
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--selected", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, help="Optional set0 test.jsonl for independent image-name checks")
    args = parser.parse_args()

    def load(path):
        rows = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            name = row["image_name"]
            if name in rows or not isinstance(row["reference"], str) or not isinstance(row["prediction"], str):
                raise ValueError(f"Duplicate or invalid prediction in {path}: {name}")
            rows[name] = row
        return rows

    frozen, selected = load(args.frozen), load(args.selected)
    if len(frozen) != 1334 or set(frozen) != set(selected):
        raise ValueError("Expected identical sets of 1,334 test predictions")
    for name in frozen:
        if frozen[name]["reference"] != selected[name]["reference"]:
            raise ValueError(f"Different references for {name}")

    if args.manifest:
        observed = set()
        for line in args.manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            path = next((v for v in row.values() if isinstance(v, str) and v.lower().endswith(".png")), None)
            if path is None:
                raise ValueError("Manifest contains a row without an image path")
            name = Path(path).name
            if name in observed:
                raise ValueError(f"Duplicate manifest image: {name}")
            observed.add(name)
            if row.get("text") != frozen.get(name, {}).get("reference"):
                raise ValueError(f"Prediction reference differs from manifest label: {name}")
        if observed != set(frozen):
            raise ValueError("Prediction image names differ from the test manifest")
    # File order is kept. Pooled edit distance is insensitive to line order.
    entries = [{"image": Path(name), "reference": row["reference"]} for name, row in frozen.items()]
    print(json.dumps({"lines": len(entries), "frozen": metrics(entries, frozen),
                      "selected": metrics(entries, selected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
