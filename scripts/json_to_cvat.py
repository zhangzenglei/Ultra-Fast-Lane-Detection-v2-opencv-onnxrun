#!/usr/bin/env python3
"""Export existing UFLDv2 predictions as unreviewed CVAT 1.1 polylines.

Reads prediction JSON and original images. Never runs inference or creates
training labels. Pillow is used only to read original image metadata.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
REVIEW_STATES = ("pending", "approved_positive", "approved_negative", "excluded")


def label_definitions() -> list[dict[str, Any]]:
    def attr(name: str, kind: str, values: list[str], default: str) -> dict[str, Any]:
        return {"name": name, "mutable": False, "input_type": kind,
                "default_value": default, "values": values}

    return [
        {"name": "lane", "type": "polyline", "color": "#20C65A", "attributes": [
            attr("prediction_id", "text", [], ""),
            attr("origin", "select", ["manual", "prediction"], "manual"),
            attr("ufld_slot", "select", ["unassigned", "0", "1", "2", "3"], "unassigned"),
        ]},
        {"name": "frame_review", "type": "tag", "color": "#E6A42C", "attributes": [
            attr("status", "select", list(REVIEW_STATES), "pending"),
            attr("reviewer", "text", [], ""),
        ]},
    ]


def validate_polyline(lane: Any, width: int, height: int, context: str) -> None:
    if not isinstance(lane, list) or len(lane) < 2:
        raise ValueError(f"{context}: a polyline requires at least two points")
    unique = set()
    for point in lane:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{context}: expected [x, y], got {point!r}")
        x, y = point
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) for v in (x, y)):
            raise ValueError(f"{context}: coordinates must be finite numbers: {point!r}")
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"{context}: point {point!r} outside {width}x{height}")
        unique.add((x, y))
    if len(unique) < 2:
        raise ValueError(f"{context}: polyline has fewer than two distinct points")


@dataclass
class Frame:
    image: Path
    name: str
    prediction: Path
    prediction_sha256: str
    width: int
    height: int
    lanes: list[list[list[float]]]
    matched_by: str


def associate_image(data: dict, prediction: Path, image_root: Path,
                    by_name: dict, by_stem: dict) -> tuple[Path, str]:
    """Prefer exact source paths; permit unambiguous basename relocation."""
    source = data.get("source_image")
    if source is not None and (not isinstance(source, str) or not source):
        raise ValueError(f"{prediction}: invalid source_image")
    if source:
        path = Path(source).expanduser()
        candidates = [path] if path.is_absolute() else [image_root / path, Path.cwd() / path]
        exact = set()
        for candidate in candidates:
            if candidate.is_file():
                resolved = candidate.resolve()
                if resolved.is_relative_to(image_root):
                    exact.add(resolved)
        if len(exact) == 1:
            return exact.pop(), "source_path"
        if len(exact) > 1:
            raise ValueError(f"{prediction}: source_image resolves to multiple images")
        matches = by_name.get(path.name, [])
        if len(matches) == 1:
            return matches[0], "source_basename_relocated"
        raise ValueError(f"{prediction}: source_image missing/ambiguous under image root: {source}")

    # Existing batch output is <original-filename>_result.json. A single-image
    # output can instead be <stem>_result.json. Never pick the first duplicate.
    name = prediction.name.removesuffix(".json").removesuffix("_result")
    matches = by_name.get(name, []) if Path(name).suffix.lower() in IMAGE_EXTENSIONS else by_stem.get(name, [])
    if len(matches) == 1:
        return matches[0], "prediction_filename"
    raise ValueError(f"{prediction}: cannot uniquely associate image from filename")


def load_frames(json_dir: Path, image_root: Path, pattern: str,
                max_lanes: int, allow_missing: bool) -> tuple[list[Frame], list[str]]:
    image_root = image_root.resolve(strict=True)
    if not image_root.is_dir():
        raise ValueError(f"Not an image directory: {image_root}")
    by_name, by_stem = defaultdict(list), defaultdict(list)
    originals: set[Path] = set()
    names: dict[Path, str] = {}
    for path in sorted(image_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            resolved = path.resolve()
            if not resolved.is_relative_to(image_root):
                raise ValueError(f"Image symlink escapes image root: {path}")
            if resolved in originals:
                raise ValueError(f"Duplicate image alias: {path}")
            originals.add(resolved)
            names[resolved] = path.relative_to(image_root).as_posix()
            by_name[path.name].append(resolved)
            by_stem[path.stem].append(resolved)
    if not originals:
        raise ValueError(f"No JPG/JPEG/PNG images in {image_root}")
    predictions = sorted(json_dir.glob(pattern))
    if not predictions:
        raise ValueError(f"No prediction JSON matching {pattern!r} in {json_dir}")
    frames, seen = [], set()
    for prediction in predictions:
        raw = prediction.read_bytes()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"{prediction}: expected a JSON object")
        for key in ("image_width", "image_height", "lanes"):
            if key not in data:
                raise ValueError(f"{prediction}: missing {key}; use --pattern to select prediction JSON only")
        width, height = data["image_width"], data["image_height"]
        if any(type(v) is not int or v <= 0 for v in (width, height)):
            raise ValueError(f"{prediction}: image dimensions must be positive integers")
        image, matched_by = associate_image(data, prediction, image_root, by_name, by_stem)
        if image in seen:
            raise ValueError(f"Multiple prediction JSON files refer to {image}")
        seen.add(image)
        with Image.open(image) as actual:
            if actual.size != (width, height):
                raise ValueError(f"{prediction}: JSON {width}x{height} != actual image {actual.size}")
            if actual.getexif().get(274, 1) != 1:
                raise ValueError(f"{image}: EXIF orientation requires an explicit coordinate policy")
        lanes = data["lanes"]
        if not isinstance(lanes, list) or len(lanes) > max_lanes:
            raise ValueError(f"{prediction}: lanes must be a list with at most {max_lanes} instances")
        for i, lane in enumerate(lanes):
            validate_polyline(lane, width, height, f"{prediction}: lanes[{i}]")
        frames.append(Frame(image, names[image], prediction.resolve(),
                            hashlib.sha256(raw).hexdigest(), width, height, lanes, matched_by))
    missing = sorted(names[path] for path in originals - seen)
    if missing and not allow_missing:
        raise ValueError(f"{len(missing)} original images have no JSON (first: {missing[0]}); "
                         "use --allow-missing-predictions only for an intentional subset")
    frames.sort(key=lambda frame: frame.name)
    # CVAT/Datumaro strips extensions from item ids. Avoid ambiguous x.jpg/x.png.
    item_ids = [Path(frame.name).with_suffix("").as_posix() for frame in frames]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Images with the same relative stem but different extensions would collide in CVAT")
    return frames, missing


def child(parent: ET.Element, name: str, value: Any) -> ET.Element:
    element = ET.SubElement(parent, name)
    element.text = str(value)
    return element


def annotation_tree(frames: list[Frame], task_name: str) -> ET.ElementTree:
    root = ET.Element("annotations")
    child(root, "version", "1.1")
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    for key, value in (("id", 0), ("name", task_name), ("size", len(frames)),
                       ("mode", "annotation"), ("overlap", 0), ("bugtracker", ""),
                       ("flipped", "False"), ("start_frame", 0),
                       ("stop_frame", len(frames) - 1), ("frame_filter", "")):
        child(task, key, value)
    labels = ET.SubElement(task, "labels")
    for definition in label_definitions():
        label = ET.SubElement(labels, "label")
        for key in ("name", "color", "type"):
            child(label, key, definition[key])
        attributes = ET.SubElement(label, "attributes")
        for spec in definition["attributes"]:
            attribute = ET.SubElement(attributes, "attribute")
            for key in ("name", "mutable", "input_type", "default_value"):
                child(attribute, key, spec[key])
            child(attribute, "values", "\n".join(spec["values"]))
    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    for key, value in (("id", 0), ("start", 0), ("stop", len(frames) - 1), ("url", "")):
        child(segment, key, value)
    child(meta, "dumped", datetime.now(timezone.utc).isoformat())
    for index, frame in enumerate(frames):
        image = ET.SubElement(root, "image", {"id": str(index), "name": frame.name,
                              "width": str(frame.width), "height": str(frame.height)})
        for lane_id, lane in enumerate(frame.lanes):
            # Separate shape elements are separate instances. group_id=0 means
            # ungrouped; do not merge all lanes into one CVAT group or track.
            line = ET.SubElement(image, "polyline", {
                "label": "lane", "source": "auto", "occluded": "0", "z_order": "0",
                "group_id": "0", "points": ";".join(f"{x},{y}" for x, y in lane)})
            for key, value in (("prediction_id", f"lane_{lane_id}"),
                               ("origin", "prediction"), ("ufld_slot", "unassigned")):
                child(line, "attribute", value).set("name", key)
        tag = ET.SubElement(image, "tag", {"label": "frame_review", "source": "auto"})
        child(tag, "attribute", "pending").set("name", "status")
        child(tag, "attribute", "").set("name", "reviewer")
    ET.indent(root)
    return ET.ElementTree(root)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export(frames: list[Frame], output: Path, batch_size: int, prefix: str,
           sequence_id: str, link_images: bool, missing: list[str]) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    labels = label_definitions()
    write_json(output / "labels.json", labels)
    tasks, manifest_rows = [], []
    for start in range(0, len(frames), batch_size):
        batch = frames[start:start + batch_size]
        task_name = f"{prefix}_{start // batch_size + 1:04d}"
        directory = output / task_name
        directory.mkdir()
        annotation_tree(batch, task_name).write(directory / "annotations.xml", encoding="utf-8", xml_declaration=True)
        write_json(directory / "labels.json", labels)
        (directory / "images.txt").write_text("".join(str(f.image) + "\n" for f in batch), encoding="utf-8")
        rows = []
        for index, frame in enumerate(batch):
            if link_images:
                link = directory / "images" / frame.name
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(frame.image)
            row = {"image_name": frame.name, "frame_id": index, "source_image": str(frame.image),
                   "image_width": frame.width, "image_height": frame.height,
                   "inference_json": str(frame.prediction), "inference_sha256": frame.prediction_sha256,
                   "matched_by": frame.matched_by, "prediction_lane_count": len(frame.lanes),
                   "prediction_point_counts": [len(lane) for lane in frame.lanes],
                   "sequence_id": sequence_id, "review_status": "pending", "training_eligible": False}
            rows.append(row)
            manifest_rows.append({**row, "task_name": task_name})
        write_json(directory / "manifest.json", {"schema_version": 1, "task_name": task_name,
                                                  "annotation_status": "unreviewed_prediction", "images": rows})
        with (directory / "review_status.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_name", "status", "reviewer", "sequence_id", "split"])
            writer.writerows((f.name, "pending", "", sequence_id, "unassigned") for f in batch)
        tasks.append({"task_name": task_name, "images": len(batch),
                      "polylines": sum(len(f.lanes) for f in batch),
                      "empty_predictions": sum(not f.lanes for f in batch)})
    write_json(output / "manifest.json", {"schema_version": 1,
        "annotation_status": "unreviewed_prediction", "images": manifest_rows})
    summary = {"schema_version": 1, "images": len(frames), "tasks": tasks,
        "polylines": sum(len(f.lanes) for f in frames),
        "points": sum(len(lane) for f in frames for lane in f.lanes),
        "empty_predictions": sum(not f.lanes for f in frames),
        "lane_count_distribution": dict(sorted(Counter(len(f.lanes) for f in frames).items())),
        "unmatched_original_images": missing, "all_frames_require_review": True,
        "training_eligible_images": 0, "coordinates": "original-image pixels; no resizing/cropping",
        "image_links": link_images, "original_images_copied": False}
    write_json(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*_result.json")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--task-prefix", default="ufld_prelabels")
    parser.add_argument("--sequence-id", required=True, help="Recording/drive id; retain for future grouped splits")
    parser.add_argument("--max-lanes", type=int, default=4)
    parser.add_argument("--allow-missing-predictions", action="store_true", help="Export an intentional subset; report omitted originals")
    parser.add_argument("--no-image-links", action="store_true", help="Only write image lists, no symlinks")
    args = parser.parse_args(argv)
    try:
        if args.batch_size < 1 or args.max_lanes < 1:
            raise ValueError("--batch-size and --max-lanes must be positive")
        if not args.task_prefix or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.task_prefix):
            raise ValueError("--task-prefix must contain only ASCII letters, digits, _ or -")
        if args.output_dir.exists():
            raise ValueError(f"Output directory already exists; refusing to overwrite: {args.output_dir}")
        frames, missing = load_frames(args.json_dir, args.image_root, args.pattern, args.max_lanes,
                                      args.allow_missing_predictions)
        result = export(frames, args.output_dir, args.batch_size, args.task_prefix,
                        args.sequence_id, not args.no_image_links, missing)
        print(f"Exported {result['images']} images, {result['polylines']} separate polylines, "
              f"{result['points']} points into {len(result['tasks'])} CVAT tasks")
        print(f"Empty predictions: {result['empty_predictions']}; ALL frames pending human review; training eligible: 0")
        print(f"Output: {args.output_dir.resolve()}")
        return 0
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
