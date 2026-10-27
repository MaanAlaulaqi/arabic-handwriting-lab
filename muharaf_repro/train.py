"""Two single-epoch stages, following the original notebooks' update schedule."""
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import VisionEncoderDecoderModel

from . import MODEL_ID
from .data import prepare_image
from .metrics import save_json
from .model import load_model


class Lines(Dataset):
    def __init__(self, entries, processor, limit):
        self.entries, self.processor, self.limit = entries, processor, limit

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        item = self.entries[index]
        image = prepare_image(item["image"])
        pixels = self.processor(images=image, return_tensors="pt").pixel_values[0]
        ids = self.processor.tokenizer(item["reference"], add_special_tokens=True)["input_ids"]
        if len(ids) > self.limit + 2:
            raise ValueError(f"Training label exceeds generation allowance: {item['image'].name}")
        return pixels, ids


def collate(batch):
    pixels = torch.stack([x[0] for x in batch])
    labels = torch.full((len(batch), max(len(x[1]) for x in batch)), -100, dtype=torch.long)
    for i, (_, ids) in enumerate(batch):
        labels[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return pixels, labels


def _valid(path, config, updates, micros):
    try:
        state = json.loads((path / "state.json").read_text(encoding="utf-8"))
        return (state["config"] == config and state["step"] <= updates
                and state["micro_done"] <= micros and (path / "model").is_dir()
                and (path / "optimizer.pt").is_file() and (path / "rng.pt").is_file())
    except (OSError, ValueError, KeyError):
        return False


def train_epoch(entries, processor, limit, output: Path, source, device: str,
                seed: int, lr: float, source_fingerprint=None):
    if device != "cuda":
        raise RuntimeError("Full training follows the notebook's CUDA AMP protocol and requires a CUDA GPU")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    loader = DataLoader(Lines(entries, processor, limit), batch_size=2, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=2,
                        pin_memory=True, collate_fn=collate)
    micros = len(loader)
    updates = math.ceil(micros / 8)
    warmup = max(1, math.ceil(updates * .05))
    config = {"model": MODEL_ID, "split": "public_1100/set0", "train_lines": len(entries),
              "seed": seed, "batch_size": 2, "grad_accum": 8, "epochs": 1,
              "lr": lr, "save_every": 300, "transformers": __import__("transformers").__version__,
              "torch": torch.__version__, "max_new_tokens": limit, "image_size": [512, 102]}
    if source_fingerprint:
        config["source_model_sha256"] = source_fingerprint
    config_file = output / "run_config.json"
    if config_file.is_file() and json.loads(config_file.read_text(encoding="utf-8")) != config:
        raise ValueError(f"Existing run has different settings: {output}")
    save_json(config_file, config)

    checkpoint = output / "checkpoint_latest"
    staging = output / "checkpoint_staging"
    backup = output / "checkpoint_backup"
    candidates = [p for p in (checkpoint, staging, backup) if _valid(p, config, updates, micros)]
    resume = max(candidates, key=lambda p: json.loads((p / "state.json").read_text())["step"]) if candidates else None
    if not resume and any(p.exists() for p in (checkpoint, staging, backup)):
        raise ValueError(f"Existing incomplete or incompatible checkpoint in {output}; inspect it before continuing")
    model = load_model(resume / "model" if resume else source, device, limit)
    if not resume:
        # In the notebooks, the model was materialized before the training seed was set.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01, foreach=False)

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, updates - warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    bf16 = torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=not bf16)
    step = micro_done = 0
    if resume:
        saved = torch.load(resume / "optimizer.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        rng = torch.load(resume / "rng.pt", map_location="cpu", weights_only=False)
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        state = json.loads((resume / "state.json").read_text(encoding="utf-8"))
        step, micro_done = state["step"], state["micro_done"]
        print(f"Resuming {output.name} from update {step}", flush=True)

    def save():
        # Temporary directory and rotation retain the previous complete checkpoint.
        local = output / "checkpoint_writing"
        if local.exists():
            shutil.rmtree(local)
        local.mkdir()
        model.save_pretrained(local / "model", safe_serialization=True)
        processor.save_pretrained(local / "model")
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict()}, local / "optimizer.pt")
        torch.save({"python": random.getstate(), "numpy": np.random.get_state(),
                    "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}, local / "rng.pt")
        save_json(local / "state.json", {"config": config, "step": step, "micro_done": micro_done})
        if staging.exists():
            shutil.rmtree(staging)
        local.rename(staging)
        if not _valid(staging, config, updates, micros):
            raise ValueError("Checkpoint staging validation failed")
        if backup.exists():
            shutil.rmtree(backup)
        if checkpoint.exists():
            checkpoint.rename(backup)
        staging.rename(checkpoint)
        if backup.exists():
            shutil.rmtree(backup)
        print(f"Saved {output.name} update {step}/{updates}", flush=True)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    for index, (pixels, labels) in enumerate(loader):
        if index < micro_done:
            continue
        remaining = micros - (index // 8) * 8
        divisor = min(8, remaining)
        pixels, labels = pixels.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if bf16 else torch.float16):
            loss = model(pixel_values=pixels, labels=labels).loss / divisor
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at microbatch {index}")
        scaler.scale(loss).backward()
        micro_done = index + 1
        if micro_done % 8 == 0 or micro_done == micros:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0 or step == updates:
                print(f"{output.name}: update {step}/{updates}, {time.monotonic()-started:.0f}s", flush=True)
            if step % 300 == 0 or step == updates:
                save()
    if step != updates or micro_done != micros:
        raise RuntimeError("Training did not complete")
    return checkpoint / "model", model
