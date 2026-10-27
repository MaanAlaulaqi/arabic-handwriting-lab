import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from . import MODEL_ID
from .data import disjoint, image_index, read_split
from .metrics import metrics, read_predictions, save_json
from .model import (device_from, fingerprint_model, load_model,
                    processor_and_limit, run_predictions)
from .train import train_epoch


def _paths(parser):
    parser.add_argument("--images-zip", type=Path, required=True, help="public_line_images.zip")
    parser.add_argument("--manifests", type=Path, required=True, help="directory with set0 train.jsonl, validation.jsonl, test.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--base-model", default=MODEL_ID,
                        help="original model ID or an offline snapshot directory")


def _release(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_pretest(args):
    digest = hashlib.sha256()
    with args.images_zip.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    args.archive_sha256 = digest.hexdigest()
    source_path = Path(args.base_model)
    args.base_sha256 = fingerprint_model(source_path) if source_path.is_dir() else None
    index = image_index(args.images_zip, args.output / "extracted_images")
    train = read_split(args.manifests, "train", index)
    valid = read_split(args.manifests, "validation", index)
    disjoint(train, valid)
    processor, limit = processor_and_limit(train, args.base_model)
    return index, train, valid, processor, limit


def _score_predictions(path, entries):
    return metrics(entries, read_predictions(path, entries))


def _locked_test(args, index, train, valid, processor, limit, selection):
    # The test manifest is first read only after the validation selection lock exists.
    test = read_split(args.manifests, "test", index)
    disjoint(train, valid, test)
    device = device_from(args.device)
    frozen_file = args.output / "frozen_test_predictions.jsonl"
    selected_file = args.output / "selected_test_predictions.jsonl"
    frozen = load_model(args.base_model, device, limit)
    run_predictions(frozen_file, test, frozen, processor, device, limit,
                    {"model_id_or_path": str(args.base_model), "commit": getattr(frozen.config, "_commit_hash", None),
                     "archive_sha256": args.archive_sha256, "base_sha256": args.base_sha256})
    _release(frozen)
    selected = load_model(args.selected_model, device, limit)
    run_predictions(selected_file, test, selected, processor, device, limit,
                    {"sha256": selection["model_sha256"], "archive_sha256": args.archive_sha256,
                     "base_sha256": args.base_sha256})
    _release(selected)
    report = {"model": MODEL_ID, "split": "public_1100/set0 test", "lines": len(test),
              "selection": selection, "frozen": _score_predictions(frozen_file, test),
              "selected": _score_predictions(selected_file, test),
              "note": "JiWER-default CER strips line edges; literal CER retains all characters. Mark matches are text alignments."}
    save_json(args.output / "test_comparison.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def quick():
    parser = argparse.ArgumentParser(description="Evaluate the frozen and selected set0 checkpoints")
    _paths(parser)
    parser.add_argument("--selected-model", type=Path, required=True, help="exported checkpoint model/ directory")
    parser.add_argument("--selection-lock", type=Path, required=True, help="selection_lock.json from the published experiment")
    args = parser.parse_args()
    if not args.selected_model.is_dir():
        parser.error("--selected-model must point to a model directory")
    lock = json.loads(args.selection_lock.read_text(encoding="utf-8"))
    actual = fingerprint_model(args.selected_model)
    if lock.get("model_sha256") != actual:
        raise ValueError(f"Selected model does not match the published selection lock: {actual}")
    index, train, valid, processor, limit = _load_pretest(args)
    if lock.get("max_new_tokens") != limit:
        raise ValueError("Tokenizer-derived generation allowance differs from selection lock")
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "selection_lock.json", lock)
    _locked_test(args, index, train, valid, processor, limit, lock)


def full():
    parser = argparse.ArgumentParser(description="Run the original bounded two-epoch set0 experiment")
    _paths(parser)
    args = parser.parse_args()
    args.device = device_from(args.device)
    if args.device != "cuda":
        parser.error("Full notebook-equivalent training requires CUDA")
    index, train, valid, processor, limit = _load_pretest(args)
    args.output.mkdir(parents=True, exist_ok=True)
    stages = {name: args.output / name for name in ("frozen_v1", "finetune_v1", "finetune_v2_second_epoch")}

    frozen_model = load_model(args.base_model, args.device, limit)
    frozen_file = stages["frozen_v1"] / "validation_predictions.jsonl"
    run_predictions(frozen_file, valid, frozen_model, processor, args.device, limit,
                    {"model_id_or_path": str(args.base_model), "commit": getattr(frozen_model.config, "_commit_hash", None),
                     "archive_sha256": args.archive_sha256, "base_sha256": args.base_sha256})
    _release(frozen_model)
    first_path, first_model = train_epoch(train, processor, limit, stages["finetune_v1"],
                                          args.base_model, args.device, 37, 1e-5,
                                          source_fingerprint=args.base_sha256)
    first_file = stages["finetune_v1"] / "validation_predictions.jsonl"
    run_predictions(first_file, valid, first_model, processor, args.device, limit,
                    {"sha256": fingerprint_model(first_path), "archive_sha256": args.archive_sha256})
    _release(first_model)
    first_state = json.loads((stages["finetune_v1"] / "checkpoint_latest" / "state.json").read_text(encoding="utf-8"))
    if first_state["step"] != 1381 or first_state["micro_done"] != 11046:
        raise ValueError("Epoch 1 is incomplete")

    second_path, second_model = train_epoch(train, processor, limit, stages["finetune_v2_second_epoch"],
                                            first_path, args.device, 38, 3e-6,
                                            source_fingerprint=fingerprint_model(first_path))
    second_file = stages["finetune_v2_second_epoch"] / "validation_predictions.jsonl"
    run_predictions(second_file, valid, second_model, processor, args.device, limit,
                    {"sha256": fingerprint_model(second_path), "archive_sha256": args.archive_sha256})
    _release(second_model)
    second_state = json.loads((stages["finetune_v2_second_epoch"] / "checkpoint_latest" / "state.json").read_text(encoding="utf-8"))
    if second_state["step"] != 1381 or second_state["micro_done"] != 11046:
        raise ValueError("Epoch 2 is incomplete")

    scores = {"frozen": _score_predictions(frozen_file, valid),
              "first_epoch": _score_predictions(first_file, valid),
              "second_epoch": _score_predictions(second_file, valid)}
    save_json(args.output / "validation_comparison.json", scores)
    selected_name = min(("first_epoch", "second_epoch"), key=lambda n: scores[n]["jiwer_default_cer"])
    args.selected_model = first_path if selected_name == "first_epoch" else second_path
    lock = {"selected": selected_name, "criterion": "lowest JiWER-default CER on set0 validation among two completed epochs",
            "validation_cer_first": scores["first_epoch"]["jiwer_default_cer"],
            "validation_cer_second": scores["second_epoch"]["jiwer_default_cer"],
            "model_sha256": fingerprint_model(args.selected_model), "max_new_tokens": limit,
            "preprocessing": "512x102 RGB; preserve height, cap width, white pad to right"}
    lock_file = args.output / "selection_lock.json"
    if lock_file.exists() and json.loads(lock_file.read_text(encoding="utf-8")) != lock:
        raise ValueError("Existing selection lock differs; stop before accessing test")
    save_json(lock_file, lock)
    _locked_test(args, index, train, valid, processor, limit, lock)
