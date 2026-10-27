import json
import zipfile
from pathlib import Path

from PIL import Image

from . import COUNTS


def image_index(images_zip: Path, cache_dir: Path):
    """Safely extract the published public line archive and index unique PNG names."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(images_zip) as archive:
        members = archive.infolist()
        for entry in members:
            path = Path(entry.filename)
            if path.is_absolute() or ".." in path.parts or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe archive entry: {entry.filename}")
        if not any(m.filename.startswith("public/") and m.filename.endswith(".png") for m in members):
            raise ValueError("Expected public/*.png in the public line archive")
        archive.extractall(cache_dir)
    images = list((cache_dir / "public").rglob("*.png"))
    if len(images) != 24495:
        raise ValueError(f"Expected 24,495 PNGs, found {len(images)}")
    by_name = {}
    for path in images:
        if path.name in by_name or not path.with_suffix(".txt").is_file():
            raise ValueError(f"Duplicate image name or missing label: {path}")
        by_name[path.name] = path
    return by_name


def _pick(record, keys):
    return next((record[k] for k in keys if isinstance(record.get(k), str)), None)


def read_split(manifests: Path, name: str, by_name: dict):
    path = manifests / f"{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing set0 manifest: {path}")
    result = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, 1):
        record = json.loads(line)
        image = _pick(record, ("image_path", "image", "image_file", "png_path", "line_image_path", "line_image"))
        if not image:
            candidates = [v for v in record.values() if isinstance(v, str) and v.lower().endswith(".png")]
            if len(candidates) == 1:
                image = candidates[0]
        if not image or Path(image).name not in by_name:
            raise ValueError(f"Image missing in {path}:{line_no}: {image}")
        actual = by_name[Path(image).name]
        reference = actual.with_suffix(".txt").read_text(encoding="utf-8")
        recorded = _pick(record, ("text", "transcription", "ground_truth", "gt", "label_text", "reference"))
        if recorded is not None and recorded != reference:
            raise ValueError(f"Manifest transcription differs from TXT: {actual.name}")
        result.append({"image": actual, "reference": reference})
    if len(result) != COUNTS[name] or len({x["image"].name for x in result}) != len(result):
        raise ValueError(f"Unexpected line count or duplicates in {name}: {len(result)}")
    return result


def disjoint(*splits):
    sets = [{x["image"].name for x in split} for split in splits]
    if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
        raise ValueError("A line image occurs in more than one split")


def prepare_image(path: Path):
    # Exact 512x102 preprocessing from all four original notebooks.
    with Image.open(path) as src:
        img = src.convert("RGB")
    w, h = img.size
    new_w = max(1, round(w * 102 / h))
    img = img.resize((new_w, 102), Image.Resampling.LANCZOS)
    if new_w > 512:
        img = img.resize((512, 102), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (512, 102), (255, 255, 255))
    canvas.paste(img, (0, 0))
    return canvas
