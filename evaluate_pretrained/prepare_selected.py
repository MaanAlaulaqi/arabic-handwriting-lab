"""Assemble and verify the published selected model without ML dependencies."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


MODEL_FILES = (
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
def fingerprint(directory):
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"No model files found in {directory}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True,
                        help="model.safetensors downloaded from the selected epoch-2 model folder")
    parser.add_argument("--metadata", type=Path, default=Path("published_run/selected_model_metadata"),
                        help="the five small model files included in this repository")
    parser.add_argument("--output", type=Path, default=Path("data/selected_checkpoint/model"))
    parser.add_argument("--selection-lock", type=Path,
                        default=Path("published_run/final_test_v1/selection_lock.json"))
    args = parser.parse_args()
    expected = json.loads(args.selection_lock.read_text(encoding="utf-8"))["model_sha256"]
    if not args.weights.is_file() or args.weights.name != "model.safetensors":
        parser.error("--weights must point to the downloaded model.safetensors file")
    for name in MODEL_FILES:
        if not (args.metadata / name).is_file():
            parser.error(f"Missing metadata file: {args.metadata / name}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output folder is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        shutil.copyfile(args.metadata / name, args.output / name)
    with args.weights.open("rb") as source, (args.output / "model.safetensors").open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    actual = fingerprint(args.output)
    print(f"Selected model fingerprint: {actual}")
    if actual != expected:
        raise SystemExit(f"Mismatch with selection lock ({expected}). Check the source folder and files.")
    print("Matches the published selection lock. The selected model is ready for evaluation.")


if __name__ == "__main__":
    main()
