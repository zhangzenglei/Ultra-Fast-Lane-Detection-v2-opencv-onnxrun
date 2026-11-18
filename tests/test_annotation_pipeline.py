"""Small deterministic fixtures; these are NOT human-reviewed training data."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from json_to_cvat import annotation_tree, export, load_frames, main, validate_polyline
from check_cvat_review import audit


class AnnotationPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.predictions = self.root / "predictions"
        self.images.mkdir()
        self.predictions.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def sample(self, name="frame.jpg", lanes=None, source=True):
        path = self.images / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), (30, 40, 50)).save(path)
        data = {"image_width": 64, "image_height": 48,
                "lanes": [[[2, 45], [10, 20]], [[30, 45], [25, 20]]] if lanes is None else lanes}
        if source:
            data["source_image"] = str(path)
        prediction = self.predictions / (name.replace("/", "_") + "_result.json")
        prediction.write_text(json.dumps(data))
        return prediction, data

    def load(self, allow_missing=False):
        return load_frames(self.predictions, self.images, "*_result.json", 4, allow_missing)

    def make_export(self, lanes=None):
        self.sample(lanes=lanes)
        frames, missing = self.load()
        out = self.root / "export"
        export(frames, out, 500, "task", "unit_fixture", True, missing)
        return out / "task_0001"

    def set_review(self, task, status, reviewer="unit_test_fixture"):
        path = task / "annotations.xml"
        tree = ET.parse(path)
        tag = tree.find("image/tag")
        tag.find("attribute[@name='status']").text = status
        tag.find("attribute[@name='reviewer']").text = reviewer
        tree.write(path, encoding="utf-8")
        return path

    def test_instance_order_empty_frames_and_source_hash_preserved(self):
        first, data = self.sample(name="road&one.jpg")
        self.sample(name="empty.jpg", lanes=[])
        before = hashlib.sha256(first.read_bytes()).hexdigest()
        frames, _ = self.load()
        tree = annotation_tree(frames, "test")
        loaded = ET.fromstring(ET.tostring(tree.getroot()))
        original = loaded.find("image[@name='road&one.jpg']")
        self.assertEqual(len(original.findall("polyline")), 2)
        self.assertEqual(original.findall("polyline")[0].attrib["points"], "2,45;10,20")
        self.assertEqual(original.findall("polyline")[1].findtext("attribute[@name='prediction_id']"), "lane_1")
        empty = loaded.find("image[@name='empty.jpg']")
        self.assertEqual(empty.findall("polyline"), [])
        self.assertEqual(empty.findtext("tag/attribute[@name='status']"), "pending")
        self.assertEqual(hashlib.sha256(first.read_bytes()).hexdigest(), before)

    def test_bad_coordinates_rejected_without_clipping(self):
        for lane in ([[[-1, 2], [3, 4]]], [[[64, 2], [3, 4]]], [[[2, 48], [3, 4]]],
                     [[[float("nan"), 1], [2, 3]]], [[[True, 1], [2, 3]]]):
            with self.subTest(lane=lane), self.assertRaises(ValueError):
                validate_polyline(lane[0], 64, 48, "fixture")

    def test_short_and_degenerate_lines_rejected(self):
        for lane in ([], [[2, 3]], [[2, 3], [2, 3]], [[2, 3, 4], [2, 5]]):
            with self.subTest(lane=lane), self.assertRaises(ValueError):
                validate_polyline(lane, 64, 48, "fixture")

    def test_real_image_dimensions_must_match_json(self):
        path, data = self.sample()
        data["image_width"] = 65
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "actual image"):
            self.load()

    def test_relocated_source_and_filename_fallback(self):
        path, data = self.sample()
        data["source_image"] = "/not_present/old_machine/frame.jpg"
        path.write_text(json.dumps(data))
        self.assertEqual(self.load()[0][0].matched_by, "source_basename_relocated")
        del data["source_image"]
        path.write_text(json.dumps(data))
        self.assertEqual(self.load()[0][0].matched_by, "prediction_filename")

    def test_ambiguous_basename_is_not_guessed(self):
        path, data = self.sample(name="one/road.jpg")
        self.sample(name="two/road.jpg")
        data["source_image"] = "/old/road.jpg"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.load()

    def test_duplicate_prediction_fails(self):
        path, _ = self.sample()
        (self.predictions / "duplicate_result.json").write_bytes(path.read_bytes())
        with self.assertRaisesRegex(ValueError, "Multiple prediction"):
            self.load()

    def test_missing_json_requires_explicit_subset(self):
        self.sample()
        Image.new("RGB", (64, 48)).save(self.images / "unpredicted.jpg")
        with self.assertRaisesRegex(ValueError, "no JSON"):
            self.load()
        self.assertEqual(self.load(allow_missing=True)[1], ["unpredicted.jpg"])

    def test_stem_collision_between_extensions_fails(self):
        self.sample(name="road.jpg")
        self.sample(name="road.png")
        with self.assertRaisesRegex(ValueError, "collide in CVAT"):
            self.load()

    def test_more_than_four_predictions_not_truncated(self):
        self.sample(lanes=[[[1, 2], [3, 4]]] * 5)
        with self.assertRaisesRegex(ValueError, "at most 4"):
            self.load()

    def test_output_links_are_original_images_and_no_overwrite(self):
        task = self.make_export()
        self.assertTrue((task / "images/frame.jpg").is_symlink())
        self.assertEqual((task / "images/frame.jpg").resolve(), (self.images / "frame.jpg").resolve())
        args = ["--json-dir", str(self.predictions), "--image-root", str(self.images),
                "--output-dir", str(task.parent), "--sequence-id", "unit"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(args)
        self.assertEqual(raised.exception.code, 1)

    def test_pending_empty_frame_is_not_a_negative(self):
        task = self.make_export(lanes=[])
        result = audit(task / "annotations.xml", task / "manifest.json")
        self.assertFalse(result["valid"])
        self.assertEqual(result["approved_negative"], 0)
        self.assertIn("NOT a negative", result["errors"][0])

    def test_explicit_reviewed_negative_requires_reviewer(self):
        task = self.make_export(lanes=[])
        self.set_review(task, "approved_negative", reviewer="")
        self.assertFalse(audit(task / "annotations.xml", task / "manifest.json")["valid"])
        self.set_review(task, "approved_negative")
        result = audit(task / "annotations.xml", task / "manifest.json")
        self.assertTrue(result["valid"])
        self.assertEqual(result["approved_negative"], 1)

    def test_positive_slots_required_only_by_explicit_training_gate(self):
        task = self.make_export()
        path = self.set_review(task, "approved_positive")
        self.assertTrue(audit(path, task / "manifest.json")["valid"])
        self.assertFalse(audit(path, task / "manifest.json", require_slots=True)["valid"])
        tree = ET.parse(path)
        for line, slot in zip(tree.findall("image/polyline"), ("1", "2")):
            line.find("attribute[@name='ufld_slot']").text = slot
        tree.write(path)
        self.assertTrue(audit(path, task / "manifest.json", require_slots=True)["valid"])

    def test_negative_with_shapes_and_positive_without_shapes_fail(self):
        task = self.make_export()
        path = self.set_review(task, "approved_negative")
        self.assertFalse(audit(path, task / "manifest.json")["valid"])
        tree = ET.parse(path)
        for shape in tree.findall("image/polyline"):
            tree.find("image").remove(shape)
        tree.write(path)
        self.set_review(task, "approved_positive")
        self.assertFalse(audit(path, task / "manifest.json")["valid"])

    def test_missing_reviewed_frame_fails(self):
        task = self.make_export()
        tree = ET.parse(task / "annotations.xml")
        tree.getroot().remove(tree.find("image"))
        tree.write(task / "annotations.xml")
        self.assertFalse(audit(task / "annotations.xml", task / "manifest.json")["valid"])

    def test_duplicate_training_slots_fail(self):
        task = self.make_export()
        path = self.set_review(task, "approved_positive")
        tree = ET.parse(path)
        for line in tree.findall("image/polyline"):
            line.find("attribute[@name='ufld_slot']").text = "1"
        tree.write(path)
        self.assertFalse(audit(path, task / "manifest.json", require_slots=True)["valid"])

    def test_more_than_four_manual_lanes_require_exclusion_not_truncation(self):
        import copy
        task = self.make_export()
        path = self.set_review(task, "approved_positive")
        tree = ET.parse(path)
        image = tree.find("image")
        for _ in range(3):
            image.append(copy.deepcopy(image.find("polyline")))
        tree.write(path)
        self.assertFalse(audit(path, task / "manifest.json")["valid"])
        self.set_review(task, "excluded")
        result = audit(path, task / "manifest.json")
        self.assertTrue(result["valid"])
        self.assertEqual(result["excluded"], 1)
        self.assertEqual(len(ET.parse(path).findall("image/polyline")), 5)

    def test_archive_preserves_filename_and_original_bytes(self):
        import zipfile
        task = self.make_export()
        target = self.root / "images.zip"
        script = Path(__file__).resolve().parents[1] / "scripts/package_cvat_images.py"
        result = subprocess.run([sys.executable, str(script), "--task-dir", str(task), "--output", str(target)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with zipfile.ZipFile(target) as archive:
            self.assertEqual(archive.namelist(), ["frame.jpg"])
            self.assertEqual(archive.read("frame.jpg"), (self.images / "frame.jpg").read_bytes())


if __name__ == "__main__":
    unittest.main()
