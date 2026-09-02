#!/usr/bin/env python3
"""Create the small BDD100K regional-processing attachment used by EVOD-RoI.

The original anonymous repository selected 100 BDD images, found the half-frame
window containing the most annotated object centres, downsampled the complete
image to one quarter, and pasted that window back at full resolution.  This
command keeps that data-generation procedure and makes the selection scalable
and auditable.  It is a demonstration attachment, not a replacement for the
official BDD100K tracking annotations.

The supported source layout is the public BDD MOT image mirror used during
development::

    source/
      images/{train,val}/<sequence>/img1/*.jpg
      labels_with_ids/{train,val}/<sequence>/img1/*.txt

Each label line may be either ``class track_id cx cy w h`` or ordinary YOLO
``class cx cy w h``. The class and box coordinates are retained in the output
label while a track ID, when present, is dropped. Class values do not affect
the clear-region search; only box centres are used, matching the original
script's behaviour.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_SEED = 42
BOUNDARY_EPSILON = 1e-6


@dataclass(frozen=True)
class FrameItem:
    source_split: str
    sequence_id: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_yolo_boxes(path: Path, width: int, height: int) -> list[tuple[int, float, float, float, float, float]]:
    """Parse YOLO or YOLO-with-track-id labels.

    Returned tuples are ``(class_id, cx, cy, box_width, box_height, track_id)``
    in absolute pixels.  Invalid lines are rejected instead of silently
    changing the selection set.
    """

    if not path.is_file():
        raise ValueError(f"Missing label file: {path}")
    parsed: list[tuple[int, float, float, float, float, float]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        fields = raw_line.split()
        if len(fields) == 6:
            class_field, track_field = fields[:2]
            values = fields[2:]
        elif len(fields) == 5:
            class_field = fields[0]
            track_field = "-1"
            values = fields[1:]
        else:
            raise ValueError(f"Invalid label at {path}:{line_number}: expected 5 or 6 fields")
        try:
            class_id = int(class_field)
            track_id = int(track_field)
            cx, cy, box_width, box_height = (float(value) for value in values)
        except ValueError as error:
            raise ValueError(f"Invalid numeric label at {path}:{line_number}") from error
        if not all(-BOUNDARY_EPSILON <= value <= 1.0 + BOUNDARY_EPSILON for value in (cx, cy)):
            raise ValueError(f"Label centre outside [0,1] at {path}:{line_number}")
        if not 0.0 < box_width <= 1.0 + BOUNDARY_EPSILON or not 0.0 < box_height <= 1.0 + BOUNDARY_EPSILON:
            raise ValueError(f"Invalid box size at {path}:{line_number}")
        if cx - box_width / 2.0 < -BOUNDARY_EPSILON or cx + box_width / 2.0 > 1.0 + BOUNDARY_EPSILON:
            raise ValueError(f"Box exceeds horizontal bounds at {path}:{line_number}")
        if cy - box_height / 2.0 < -BOUNDARY_EPSILON or cy + box_height / 2.0 > 1.0 + BOUNDARY_EPSILON:
            raise ValueError(f"Box exceeds vertical bounds at {path}:{line_number}")
        cx = min(1.0, max(0.0, cx))
        cy = min(1.0, max(0.0, cy))
        box_width = min(1.0, max(0.0, box_width))
        box_height = min(1.0, max(0.0, box_height))
        parsed.append(
            (class_id, cx * width, cy * height, box_width * width, box_height * height, track_id)
        )
    return parsed


def boxes_from_labels(labels: Iterable[tuple[int, float, float, float, float, float]]) -> list[Box]:
    return [
        Box(cx - box_width / 2.0, cy - box_height / 2.0, cx + box_width / 2.0, cy + box_height / 2.0)
        for _, cx, cy, box_width, box_height, _ in labels
    ]


def find_best_region(image_shape: tuple[int, int], boxes: Sequence[Box]) -> tuple[int, int, int, int]:
    """Return the half-frame window containing the most box centres.

    This is intentionally the same centre-counting rule as the original
    ``generate_visualization.py`` and ``process_bdd100k.py`` scripts.
    """

    height, width = image_shape
    window_h, window_w = height // 2, width // 2
    if not boxes:
        return (width - window_w) // 2, (height - window_h) // 2, window_w, window_h
    grid_h, grid_w = height - window_h + 1, width - window_w + 1
    count_grid = np.zeros((grid_h, grid_w), dtype="uint16")
    for box in boxes:
        center_x = (box.x1 + box.x2) / 2.0
        center_y = (box.y1 + box.y2) / 2.0
        min_x = max(0, int(center_x - window_w))
        max_x = min(grid_w, int(center_x + 1))
        min_y = max(0, int(center_y - window_h))
        max_y = min(grid_h, int(center_y + 1))
        if min_x < max_x and min_y < max_y:
            count_grid[min_y:max_y, min_x:max_x] += 1
    max_y, max_x = np.unravel_index(count_grid.argmax(), count_grid.shape)
    return int(max_x), int(max_y), window_w, window_h


def process_image(image, region: tuple[int, int, int, int], downsample_ratio: float = 0.25):
    """Downsample the image and restore the selected region pixel-for-pixel."""

    if not 0 < downsample_ratio <= 1:
        raise ValueError("downsample_ratio must be in (0, 1]")
    height, width = image.shape[:2]
    x, y, region_width, region_height = region
    original_region = image[y : y + region_height, x : x + region_width].copy()
    reduced_width = max(1, round(width * downsample_ratio))
    reduced_height = max(1, round(height * downsample_ratio))
    low_resolution = cv2.resize(image, (reduced_width, reduced_height))
    reconstructed = cv2.resize(low_resolution, (width, height))
    reconstructed[y : y + region_height, x : x + region_width] = original_region
    return reconstructed


def draw_labeled_image(image, boxes: Sequence[Box], region: tuple[int, int, int, int]):
    labeled = image.copy()
    for box in boxes:
        cv2.rectangle(labeled, (round(box.x1), round(box.y1)), (round(box.x2), round(box.y2)), (0, 0, 255), 2)
    x, y, width, height = region
    cv2.rectangle(labeled, (x, y), (x + width, y + height), (0, 255, 0), 4)
    text_y = y - 10 if y > 20 else y + 30
    cv2.putText(labeled, "Clear Region", (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return labeled


def source_items(source_root: Path, split: str) -> list[FrameItem]:
    image_root = source_root / "images" / split
    label_root = source_root / "labels_with_ids" / split
    if not image_root.is_dir() or not label_root.is_dir():
        raise ValueError(f"Expected images/{split} and labels_with_ids/{split} below {source_root}")
    items: list[FrameItem] = []
    for sequence_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        frame_dir = sequence_dir / "img1"
        labels_dir = label_root / sequence_dir.name / "img1"
        for image_path in sorted(path for path in frame_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.is_file():
                items.append(FrameItem(split, sequence_dir.name, image_path, label_path))
    return items


def load_daytime_road_sequences(metadata_root: Path, split: str) -> set[str]:
    """Read legacy BDD metadata and keep daytime city-street/highway videos."""

    selected: set[str] = set()
    split_root = metadata_root / split
    for metadata_file in sorted(split_root.rglob("*.json")):
        try:
            payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            if not isinstance(record, dict):
                continue
            attributes = record.get("attributes")
            name = record.get("name")
            if not isinstance(attributes, dict) or not isinstance(name, str):
                continue
            if attributes.get("timeofday") == "daytime" and attributes.get("scene") in {"city street", "highway"}:
                selected.add(Path(name).stem)
    return selected


def choose_items(
    source_root: Path,
    total: int,
    train_count: int,
    seed: int,
    metadata_root: Path | None = None,
) -> list[FrameItem]:
    if total < 1 or train_count < 0 or train_count > total:
        raise ValueError("total must be positive and train_count must lie in [0,total]")
    train = source_items(source_root, "train")
    val = source_items(source_root, "val")
    if metadata_root is not None:
        train_ids = load_daytime_road_sequences(metadata_root.resolve(), "train")
        val_ids = load_daytime_road_sequences(metadata_root.resolve(), "val")
        train = [item for item in train if item.sequence_id in train_ids]
        val = [item for item in val if item.sequence_id in val_ids]
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    val_count = total - train_count
    if len(train) < train_count or len(val) < val_count:
        raise ValueError(f"Requested {train_count} train and {val_count} val frames, but only have {len(train)} and {len(val)}")
    selected = train[:train_count] + val[:val_count]
    rng.shuffle(selected)
    return selected


def write_yolo_label(
    path: Path,
    labels: Sequence[tuple[int, float, float, float, float, float]],
    width: int,
    height: int,
) -> None:
    """Write ordinary five-field YOLO labels, dropping only track identity."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{class_id} {cx / width:.8f} {cy / height:.8f} {box_width / width:.8f} {box_height / height:.8f}"
        for class_id, cx, cy, box_width, box_height, _ in labels
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_checksums(output_dir: Path) -> None:
    checksum_path = output_dir / "files.sha256"
    files = sorted(path for path in output_dir.rglob("*") if path.is_file() and path != checksum_path)
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path.relative_to(output_dir).as_posix()}\n")


def write_dataset_yaml(output_dir: Path) -> None:
    (output_dir / "dataset.yaml").write_text(
        "path: .\ntrain: processed/train\nval: processed/val\nnames:\n  0: vehicle\n",
        encoding="utf-8",
    )


def expand(
    source_root: Path,
    output_dir: Path,
    total: int = 10_000,
    train_count: int = 8_000,
    seed: int = DEFAULT_SEED,
    labeled_preview_count: int = 100,
    metadata_root: Path | None = None,
    license_file: Path | None = None,
    attribution_file: Path | None = None,
) -> dict[str, object]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    source_root = source_root.resolve()
    selected = choose_items(source_root, total, train_count, seed, metadata_root=metadata_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_root = output_dir / "processed"
    labels_root = output_dir / "labels"
    preview_root = output_dir / "labeled"
    records: list[dict[str, object]] = []
    preview_limit = max(0, min(labeled_preview_count, len(selected)))
    for index, item in enumerate(selected):
        image = cv2.imread(str(item.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {item.image_path}")
        height, width = image.shape[:2]
        labels = parse_yolo_boxes(item.label_path, width, height)
        boxes = boxes_from_labels(labels)
        region = find_best_region((height, width), boxes)
        processed = process_image(image, region)
        relative_name = Path(item.source_split) / item.sequence_id / item.image_path.name
        processed_path = processed_root / relative_name
        label_path = labels_root / relative_name.with_suffix(".txt")
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(processed_path), processed):
            raise OSError(f"Cannot write processed image: {processed_path}")
        write_yolo_label(label_path, labels, width, height)
        preview_path = None
        if index < preview_limit:
            preview_path = preview_root / relative_name
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(preview_path), draw_labeled_image(image, boxes, region)):
                raise OSError(f"Cannot write labeled preview: {preview_path}")
        records.append(
            {
                "index": index,
                "split": item.source_split,
                "sequence_id": item.sequence_id,
                "source_image": item.image_path.relative_to(source_root).as_posix(),
                "source_label": item.label_path.relative_to(source_root).as_posix(),
                "processed": processed_path.relative_to(output_dir).as_posix(),
                "label": label_path.relative_to(output_dir).as_posix(),
                "labeled_preview": preview_path.relative_to(output_dir).as_posix() if preview_path else None,
                "width": width,
                "height": height,
                "box_count": len(boxes),
                "clear_region": list(region),
                "source_sha256": sha256(item.image_path),
                "processed_sha256": sha256(processed_path),
                "source_label_sha256": sha256(item.label_path),
                "label_sha256": sha256(label_path),
            }
        )
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(records[0].keys()) if records else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "method": "original BDD100K half-frame clear-region preprocessing",
        "seed": seed,
        "total": len(records),
        "train": sum(record["split"] == "train" for record in records),
        "val": sum(record["split"] == "val" for record in records),
        "labeled_previews": sum(record["labeled_preview"] is not None for record in records),
        "downsample_ratio": 0.25,
        "label_format": "YOLO class cx cy width height; source track IDs removed",
        "metadata_filter": "daytime city street/highway" if metadata_root is not None else "none",
        "image_dimensions": sorted({(record["width"], record["height"]) for record in records}),
        "source_note": "The source mirror provides vehicle-only YOLO-with-track-id labels; this is not a complete box_track_20 release.",
    }
    write_dataset_yaml(output_dir)
    if license_file is not None:
        shutil.copyfile(license_file.resolve(), output_dir / "BDD100K_DATA_LICENSE.txt")
    if attribution_file is not None:
        shutil.copyfile(attribution_file.resolve(), output_dir / "SOURCE_ATTRIBUTION.md")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_checksums(output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total", type=int, default=10_000)
    parser.add_argument("--train-count", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--labeled-preview-count", type=int, default=100)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        help="Optional legacy BDD metadata root; when supplied, keep daytime city-street/highway sequences only.",
    )
    parser.add_argument("--license-file", type=Path, help="Optional BDD100K license to copy into the output.")
    parser.add_argument("--attribution-file", type=Path, help="Optional source-attribution file to copy into the output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = expand(
        args.source_root,
        args.output_dir,
        total=args.total,
        train_count=args.train_count,
        seed=args.seed,
        labeled_preview_count=args.labeled_preview_count,
        metadata_root=args.metadata_root,
        license_file=args.license_file,
        attribution_file=args.attribution_file,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
