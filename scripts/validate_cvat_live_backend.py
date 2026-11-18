"""Run INSIDE an existing CVAT backend via manage.py shell.

Creates one NEW five-image verification task through the real REST framework
views, import/export workers and database. Uses APIClient.force_authenticate as
a local administrative integration-test context, not browser credentials or an
internet login. Never modifies existing tasks or marks predictions as reviewed.

Required environment: UFLD_VALIDATION_DIR, UFLD_VALIDATION_USERNAME.
"""
import json
import os
from pathlib import Path
import time
from urllib.parse import unquote, urlencode, urlsplit
import uuid
import xml.etree.ElementTree as ET
import zipfile

from django.contrib.auth import get_user_model
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APIClient


def run():
    directory = Path(os.environ["UFLD_VALIDATION_DIR"])
    report_path = directory / "live_report.json"
    resume = os.environ.get("UFLD_VALIDATION_RESUME") == "1"
    if report_path.exists() and not resume:
        raise RuntimeError("Refusing to overwrite live report")
    user = get_user_model().objects.get(username=os.environ["UFLD_VALIDATION_USERNAME"], is_active=True)
    client = APIClient()
    client.force_authenticate(user=user)

    def request(method, path, payload=None, form="json", expected=(200,)):
        response = getattr(client, method)(path, data=payload, format=form, HTTP_HOST="localhost")
        if response.status_code not in expected:
            raise RuntimeError(f"{method.upper()} {path}: HTTP {response.status_code}: "
                               f"{getattr(response, 'data', response.content[:1000])}")
        return response

    def wait_for_job(response, operation):
        body = response.data
        rq_id = body.get("rq_id") or body.get("id")
        if not rq_id:
            raise RuntimeError(f"Missing request id for {operation}: {body}")
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            result = request("get", f"/api/requests/{rq_id}").data
            status = result.get("status")
            if status == "finished":
                print(operation, "finished", flush=True)
                return result
            if status in {"failed", "canceled"}:
                raise RuntimeError(f"{operation}: {result}")
            time.sleep(2)
        raise TimeoutError(f"{operation}: exceeded 300 seconds")

    if resume:
        report = json.loads(report_path.read_text())
        task_id = report["task_id"]
        task = request("get", f"/api/tasks/{task_id}").data
        assert task["name"] == report["task_name"] and task["name"].startswith("UFLDv2_import_validation_")
        assert task["owner"]["id"] == user.id
        print("Resuming export verification of own validation task", task_id, flush=True)
    else:
        labels = json.loads((directory / "labels.json").read_text())
        name = f"UFLDv2_import_validation_{uuid.uuid4().hex[:8]}"
        task = request("post", "/api/tasks", {"name": name, "labels": labels}, expected=(201,)).data
        task_id = task["id"]
        report = {"task_id": task_id, "task_name": name, "task_url": f"http://localhost:8080/tasks/{task_id}",
                  "context": "local administrative DRF integration client; real database and background workers",
                  "existing_tasks_modified": False, "human_review_performed": False,
                  "server_version": request("get", "/api/server/about").data["version"]}
        report_path.write_text(json.dumps({**report, "status": "started"}, indent=2))
        print("Created new validation task", task_id, flush=True)
        response = request("post", f"/api/tasks/{task_id}/data", {
            "image_quality": 100, "sorting_method": "lexicographical",
            "client_files[0]": SimpleUploadedFile("images.zip", (directory / "images.zip").read_bytes(),
                                                    content_type="application/zip"),
        }, form="multipart", expected=(202,))
        wait_for_job(response, "image upload")

        response = request("post", f"/api/tasks/{task_id}/annotations?" + urlencode({"format": "CVAT 1.1"}), {
            "annotation_file": SimpleUploadedFile("annotations.xml", (directory / "annotations.xml").read_bytes(),
                                                   content_type="application/xml"),
        }, form="multipart", expected=(202, 201))
        if response.status_code == 202:
            wait_for_job(response, "XML annotation import")

    expected_images = {image.attrib["name"]: image for image in ET.parse(directory / "annotations.xml").getroot().findall("image")}
    meta = request("get", f"/api/tasks/{task_id}/data/meta").data
    frames = meta["frames"]
    assert len(frames) == len(expected_images) == 5
    names = {index: frame["name"] for index, frame in enumerate(frames)}
    for frame in frames:
        expected = expected_images[frame["name"]]
        assert (frame["width"], frame["height"]) == (int(expected.attrib["width"]), int(expected.attrib["height"]))
    annotations = request("get", f"/api/tasks/{task_id}/annotations").data
    report["live_api_counts"] = {key: len(annotations[key]) for key in ("shapes", "tags", "tracks")}
    assert report["live_api_counts"] == {"shapes": 10, "tags": 5, "tracks": 0}

    # Compare exact point sequences per image, independent of generated object ids.
    for frame_id, name in names.items():
        actual = sorted(tuple(shape["points"]) for shape in annotations["shapes"] if shape["frame"] == frame_id)
        expected = sorted(tuple(float(v) for pair in line.attrib["points"].split(";") for v in pair.split(","))
                          for line in expected_images[name].findall("polyline"))
        assert actual == expected, name
    report["database_coordinates_match_input_xml"] = True

    query = urlencode({"format": "CVAT for images 1.1", "save_images": "false"})
    result = wait_for_job(request("post", f"/api/tasks/{task_id}/dataset/export?{query}", expected=(202,)), "CVAT XML export")
    parsed = urlsplit(result["result_url"])
    response = request("get", parsed.path + ("?" + parsed.query if parsed.query else ""))
    data = b"".join(response.streaming_content) if response.streaming else response.content
    if not data.startswith(b"PK"):
        # The production backend uses nginx X-Accel-Redirect; an in-process
        # client receives its authorized file header, not nginx's file bytes.
        redirect = response.get("X-Accel-Redirect")
        if not redirect:
            raise RuntimeError(f"No ZIP response or nginx file header: {dict(response.headers)}")
        file_root = Path(settings.SENDFILE_ROOT).resolve()
        url_root = settings.SENDFILE_URL.rstrip("/") + "/"
        if not redirect.startswith(url_root):
            raise RuntimeError("Unexpected nginx download prefix")
        authorized_file = (file_root / unquote(redirect[len(url_root):])).resolve()
        if not authorized_file.is_relative_to(file_root):
            raise RuntimeError("nginx download path escapes configured root")
        data = authorized_file.read_bytes()
    export_path = directory / "roundtrip.zip"
    export_path.write_bytes(data)
    with zipfile.ZipFile(export_path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".xml")]
        assert len(members) == 1
        xml = archive.read(members[0])
    (directory / "roundtrip.xml").write_bytes(xml)
    roundtrip = ET.fromstring(xml)
    images = {image.attrib["name"]: image for image in roundtrip.findall("image")}
    assert set(images) == set(expected_images)
    for name, image in images.items():
        before = expected_images[name]
        assert (image.attrib["width"], image.attrib["height"]) == (before.attrib["width"], before.attrib["height"])
        actual = sorted(tuple(float(v) for pair in line.attrib["points"].split(";") for v in pair.split(","))
                        for line in image.findall("polyline"))
        expected = sorted(tuple(float(v) for pair in line.attrib["points"].split(";") for v in pair.split(","))
                          for line in before.findall("polyline"))
        assert actual == expected, name
        tags = image.findall("tag[@label='frame_review']")
        assert len(tags) == 1 and tags[0].findtext("attribute[@name='status']") == "pending"
        for line in image.findall("polyline"):
            assert line.get("label") == "lane"
            assert line.findtext("attribute[@name='origin']") == "prediction"
            assert line.findtext("attribute[@name='ufld_slot']") == "unassigned"
    report.update(status="PASS", roundtrip_original_pixel_coordinates="PASS", pending_tags_retained=True,
                  images=5, polylines=10, points=412, training_started=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


run()
