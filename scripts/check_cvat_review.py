#!/usr/bin/env python3
"""Audit corrected CVAT XML before any future CULane conversion; never train."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from json_to_cvat import REVIEW_STATES, validate_polyline, write_json


def audit(xml_path: Path, manifest_path: Path, require_slots: bool = False) -> dict:
    # CVAT exports are small XML documents, not arbitrary DTD-bearing documents.
    raw = xml_path.read_bytes()
    if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise ValueError("DTD/entities are not accepted")
    root = ET.fromstring(raw)
    if root.tag != "annotations" or root.findtext("version") != "1.1" or root.findall("track"):
        raise ValueError("Require CVAT 1.1 image annotations, not video tracks")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {item["image_name"]: item for item in manifest["images"]}
    if len(expected) != len(manifest["images"]):
        raise ValueError("Duplicate names in task manifest")
    seen, errors, statuses = set(), [], Counter()
    positive, negative, excluded = 0, 0, 0
    for image in root.findall("image"):
        name = image.get("name", "")
        local_errors = []
        if name not in expected or name in seen:
            errors.append(f"{name}: unexpected or duplicated image")
            continue
        seen.add(name)
        item = expected[name]
        try:
            width, height = int(image.attrib["width"]), int(image.attrib["height"])
            if (width, height) != (item["image_width"], item["image_height"]):
                raise ValueError("dimensions differ from original manifest; no implicit scaling allowed")
            reviews = image.findall("tag[@label='frame_review']")
            if len(reviews) != 1:
                raise ValueError("exactly one frame_review tag required")
            attrs = {a.attrib["name"]: a.text or "" for a in reviews[0].findall("attribute")}
            status = attrs.get("status", "pending")
            if status not in REVIEW_STATES:
                raise ValueError(f"invalid review status {status!r}")
            statuses[status] += 1
            if status == "pending":
                raise ValueError("still pending; an empty prediction is NOT a negative ground truth")
            if not attrs.get("reviewer", "").strip():
                raise ValueError("reviewer must be recorded for approved or excluded frames")
            shapes = [element for element in image if element.tag not in {"tag"}]
            slots, groups = set(), set()
            for shape in shapes:
                if shape.tag != "polyline" or shape.get("label") != "lane":
                    raise ValueError("all lane shapes must be separate polylines with label lane")
                pairs = [pair.split(",") for pair in shape.attrib["points"].split(";")]
                lane = [[float(v) for v in pair] for pair in pairs]
                validate_polyline(lane, width, height, name)
                group = int(shape.get("group_id", "0"))
                if group and group in groups:
                    raise ValueError("multiple lane polylines share a group; resolve instance identity")
                groups.add(group)
                attributes = {a.attrib["name"]: a.text or "" for a in shape.findall("attribute")}
                slot = attributes.get("ufld_slot", "unassigned")
                if slot != "unassigned":
                    if slot not in {"0", "1", "2", "3"} or slot in slots:
                        raise ValueError("ufld_slot must be a unique slot from 0..3")
                    slots.add(slot)
                elif require_slots and status == "approved_positive":
                    raise ValueError("training conversion requires explicit ufld_slot for every lane")
            if status == "approved_positive":
                if not 1 <= len(shapes) <= 4:
                    raise ValueError("approved_positive requires 1..4 lanes; do not silently truncate extras")
                positive += 1
            elif status == "approved_negative":
                if shapes:
                    raise ValueError("approved_negative must have zero lanes after explicit human review")
                negative += 1
            elif status == "excluded":
                excluded += 1
        except (KeyError, TypeError, ValueError) as error:
            local_errors.append(f"{name}: {error}")
        errors.extend(local_errors)
    for name in sorted(expected.keys() - seen):
        errors.append(f"{name}: missing from reviewed XML; cannot assume background")
    return {"valid": not errors, "images_expected": len(expected), "images_found": len(seen),
            "review_status_counts": dict(statuses), "approved_positive": positive,
            "approved_negative": negative, "excluded": excluded,
            "error_count": len(errors), "errors": errors,
            "slots_required": require_slots, "training_started": False,
            "culane_conversion_implemented": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--task-manifest", required=True, type=Path)
    parser.add_argument("--require-slots", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.report and args.report.exists():
            raise ValueError("Refusing to overwrite existing report")
        result = audit(args.annotations, args.task_manifest, args.require_slots)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.report, result)
        print(f"Review audit: valid={result['valid']} images={result['images_found']}/"
              f"{result['images_expected']} errors={result['error_count']}")
        for error in result["errors"][:10]:
            print(error)
        return 0 if result["valid"] else 1
    except (OSError, ValueError, KeyError, ET.ParseError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
