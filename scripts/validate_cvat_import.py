#!/usr/bin/env python3
"""Run CVAT's actual XML parsers on exports without starting a CVAT server.

Requires Datumaro in an isolated validation environment and an official CVAT
source checkout. AST selection executes the original parser definitions unchanged;
only the database/task sink is an in-memory adapter. This is NOT a web/API test.
"""
from __future__ import annotations

import argparse
import ast
from collections import OrderedDict, namedtuple
from glob import glob
import hashlib
import json
import os
import os.path as osp
from pathlib import Path
import subprocess
from types import SimpleNamespace

import datumaro as dm
from datumaro.components.annotation import (
    AnnotationType, Bbox, Label, LabelCategories, Points, Polygon, PolyLine,
)
from datumaro.components.dataset import DatasetItem
from datumaro.components.dataset_base import DEFAULT_SUBSET_NAME, DatasetBase
from datumaro.components.importer import ImportContext
from datumaro.components.media import Image
from defusedxml import ElementTree


def original_definitions(path: Path, names: set[str], scope: dict) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in names]
    if {node.name for node in selected} != names:
        raise RuntimeError(f"Expected official parser definitions not found in {path}: {names}")
    # Future annotations avoid importing Django solely for type hints. The
    # actual selected parser bodies and frame-matching function are unchanged.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0)] + selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), scope)


class TaskSink:
    """Database-free task adapter; strict filename matching, no id fallback."""
    META_FIELD = "task"
    Attribute = namedtuple("Attribute", "name value")

    def __init__(self, rows: list[dict]):
        self.meta = {"task": {"mode": "annotation"}}
        self.frame_info = {row["frame_id"]: {"path": row["image_name"]} for row in rows}
        # Deliberately permute CVAT frame order to verify matching uses the
        # actual filename rather than assuming XML frame id == task frame id.
        n = len(rows)
        self.name_to_frame = {row["image_name"]: n - 1 - i for i, row in enumerate(rows)}
        self.frame_info = {self.name_to_frame[row["image_name"]]: {"path": row["image_name"]} for row in rows}
        self.shape_by_name = {row["image_name"]: [] for row in rows}
        self.tags_by_name = {row["image_name"]: [] for row in rows}
        self.matches = []

    def match_frame(self, name, root_hint=None, *, path_has_ext=True):
        if not path_has_ext:
            candidates = [v for k, v in self.name_to_frame.items() if osp.splitext(k)[0] == name]
            return candidates[0] if len(candidates) == 1 else None
        match = self.name_to_frame.get(name)
        if match is not None:
            self.matches.append(name)
        return match

    def abs_frame_id(self, value):
        return value

    @staticmethod
    def LabeledShape(**kwargs):
        return SimpleNamespace(**kwargs)

    @staticmethod
    def Tag(**kwargs):
        return SimpleNamespace(**kwargs)

    def add_shape(self, shape):
        if shape.type != "polyline" or shape.label != "lane":
            raise ValueError("Unexpected shape in task importer")
        self.shape_by_name[self.frame_info[shape.frame]["path"]].append(shape)

    def add_tag(self, tag):
        if tag.label != "frame_review":
            raise ValueError("Unexpected tag in task importer")
        self.tags_by_name[self.frame_info[tag.frame]["path"]].append(tag)


def validate(cvat_source: Path, export_root: Path) -> dict:
    formats = cvat_source / "cvat/apps/dataset_manager/formats/cvat.py"
    bindings = cvat_source / "cvat/apps/dataset_manager/bindings.py"
    scope = dict(globals())
    task_mode = SimpleNamespace(ANNOTATION="annotation", INTERPOLATION="interpolation")
    scope.update(TaskMode=task_mode, models=SimpleNamespace(TaskMode=task_mode),
                 CvatImportError=ValueError, datumaro=dm)
    original_definitions(bindings, {"match_dm_item"}, scope)
    original_definitions(formats, {"CvatPath", "CvatExtractor", "load_anno"}, scope)
    results = []
    for directory in sorted(p for p in export_root.iterdir() if p.is_dir() and (p / "annotations.xml").exists()):
        rows = json.loads((directory / "manifest.json").read_text())["images"]
        expected = {row["image_name"]: row for row in rows}
        extractor = scope["CvatExtractor"](str(directory / "annotations.xml"))
        dataset_items = list(extractor)
        if len(dataset_items) != len(rows):
            raise AssertionError(f"Dataset importer lost frames in {directory}")
        category = extractor.categories()[AnnotationType.label]
        imported = {}
        for item in dataset_items:
            # Resolve the media path against symlinked original images to check
            # both metadata dimensions and that the names correspond to media.
            name = Path(item.media.path).relative_to(directory / "images").as_posix()
            row = expected[name]
            assert tuple(item.media.size) == (row["image_height"], row["image_width"])
            assert Path(item.media.path).resolve() == Path(row["source_image"]).resolve()
            # Read actual decoded image data for representative real frames of
            # each lane-count category, including the no-prediction case.
            imported[name] = item
        sampled_counts = set()
        for name, item in imported.items():
            count = expected[name]["prediction_lane_count"]
            if count not in sampled_counts:
                assert item.media.data.shape[:2] == tuple(item.media.size)
                sampled_counts.add(count)
        sink = TaskSink(rows)
        scope["load_anno"](str(directory / "annotations.xml"), sink)
        assert set(sink.matches) == set(expected), "Official frame matcher failed filename matching"
        lane_total, point_total, empty = 0, 0, 0
        for name, row in expected.items():
            prediction = json.loads(Path(row["inference_json"]).read_text())
            item = imported[name]
            lines = [a for a in item.annotations if a.type == AnnotationType.polyline]
            tags = [a for a in item.annotations if a.type == AnnotationType.label]
            assert len(lines) == len(prediction["lanes"]) == len(sink.shape_by_name[name])
            assert len(tags) == len(sink.tags_by_name[name]) == 1
            assert category[tags[0].label].name == "frame_review"
            assert tags[0].attributes["status"] == "pending"
            assert {a.name: a.value for a in sink.tags_by_name[name][0].attributes}["status"] == "pending"
            for index, (line, lane, shape) in enumerate(zip(lines, prediction["lanes"], sink.shape_by_name[name])):
                flat = [float(value) for point in lane for value in point]
                assert list(line.points) == flat == shape.points, (name, index)
                assert category[line.label].name == "lane"
                assert line.attributes["prediction_id"] == f"lane_{index}"
                assert line.attributes["ufld_slot"] == "unassigned"
                assert shape.group == 0
                point_total += len(lane)
            lane_total += len(lines)
            empty += not lines
            assert hashlib.sha256(Path(row["inference_json"]).read_bytes()).hexdigest() == row["inference_sha256"]
        result = {"task": directory.name, "images": len(rows), "polylines": lane_total,
                  "points": point_total, "empty_predictions": empty,
                  "original_json_hashes_unchanged": True,
                  "filename_matching_with_reversed_task_order": True,
                  "cvat_dataset_extractor": "PASS", "cvat_task_load_anno": "PASS",
                  "real_image_decode_samples": len(sampled_counts)}
        results.append(result)
        print(json.dumps(result))
    if not results:
        raise ValueError("No task exports found")
    return {"validation": "CVAT official parser execution, database sink adapted; NOT live web/API import",
        "cvat_commit": subprocess.check_output(["git", "-C", str(cvat_source), "rev-parse", "HEAD"], text=True).strip(),
        "cvat_parser_sha256": hashlib.sha256(formats.read_bytes()).hexdigest(),
        "datumaro_version": dm.__version__, "tasks": results,
        "totals": {key: sum(row[key] for row in results) for key in ("images", "polylines", "points", "empty_predictions")},
        "live_cvat_server_import": "NOT TESTED"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cvat-source", required=True, type=Path)
    parser.add_argument("--export-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Refusing to overwrite report")
    result = validate(args.cvat_source.resolve(), args.export_root.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("PASS:", result["totals"])


if __name__ == "__main__":
    main()
