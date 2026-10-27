import hashlib
import json
from pathlib import Path

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from . import MODEL_ID
from .data import prepare_image
from .metrics import read_predictions


def device_from(request: str):
    if request == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return request


def processor_and_limit(train, source=MODEL_ID):
    processor = TrOCRProcessor.from_pretrained(source)
    if processor.image_processor.do_resize is not False:
        raise ValueError("Processor resize setting changed; review image preparation")
    special = set(processor.tokenizer.all_special_ids)
    longest = 0
    for item in train:
        ids = processor.tokenizer(item["reference"], add_special_tokens=False)["input_ids"]
        back = processor.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if back != item["reference"] or special.intersection(ids):
            raise ValueError(f"Tokenizer does not preserve label: {item['image'].name}")
        longest = max(longest, len(ids))
    return processor, longest + 8


def load_model(source, device, token_limit):
    model = VisionEncoderDecoderModel.from_pretrained(source).to(device)
    if model.config.decoder_start_token_id is None or model.config.pad_token_id is None:
        raise ValueError("Model lacks decoder start or pad token")
    decoder_limit = getattr(model.config.decoder, "max_position_embeddings", None)
    if decoder_limit is not None and token_limit + 2 > decoder_limit:
        raise ValueError("Training label exceeds decoder position allowance")
    # The source model card calls for encoder position interpolation.
    original_forward = model.encoder.forward

    def interpolated_forward(pixel_values, *args, **kwargs):
        kwargs["interpolate_pos_encoding"] = True
        return original_forward(pixel_values, *args, **kwargs)

    model.encoder.forward = interpolated_forward
    return model


def predict(model, processor, item, device, token_limit):
    image = prepare_image(item["image"])
    pixels = processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.inference_mode():
        ids = model.generate(pixels, max_length=token_limit + 1, num_beams=1)
    return processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def run_predictions(path, entries, model, processor, device, token_limit, identity):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = path.with_suffix(path.suffix + ".source.json")
    provenance = {"source": identity, "max_new_tokens": token_limit,
                  "preprocessing": "512x102 RGB; preserve height, cap width, white pad right"}
    if meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")) != provenance:
            raise ValueError(f"Predictions belong to a different model or preprocessing: {path}")
    elif path.exists():
        raise ValueError(f"Existing predictions have no source metadata: {path}")
    else:
        meta_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    existing = read_predictions(path, entries)
    model.eval()
    with path.open("a", encoding="utf-8") as out:
        for i, item in enumerate(entries, 1):
            name = item["image"].name
            if name in existing:
                continue
            row = {"image_name": name, "reference": item["reference"],
                   "prediction": predict(model, processor, item, device, token_limit)}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            existing[name] = row
            if i % 25 == 0:
                print(f"{path.name}: {i}/{len(entries)}", flush=True)
    if len(existing) != len(entries):
        raise ValueError(f"Incomplete output: {path}")
    return existing


def fingerprint_model(directory: Path):
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No model files in {directory}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(directory)).encode("utf-8"))
        with file.open("rb") as inp:
            for chunk in iter(lambda: inp.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
