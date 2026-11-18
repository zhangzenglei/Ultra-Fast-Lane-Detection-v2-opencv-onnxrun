#!/usr/bin/env python3
"""Ablate existing C++ CUDA inference settings; no training or ground truth creation."""
import argparse
import csv
import hashlib
import html
import itertools
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--baseline-json-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fishbone-indices", required=True)
    parser.add_argument("--control-indices", required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("Refusing to overwrite experiment directory")
    files = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    fish = set(map(int, args.fishbone_indices.split(",")))
    controls = set(map(int, args.control_indices.split(",")))
    if fish & controls:
        parser.error("Fishbone candidate and control indices overlap")
    selected = sorted(fish | controls)
    if not selected or min(selected) < 0 or max(selected) >= len(files):
        parser.error("Selected frame index outside image list")
    records = []
    for index in selected:
        original = files[index].resolve()
        previous = (args.baseline_json_dir / (original.name + "_result.json")).resolve()
        records.append({"index": index, "source_image": str(original), "name": original.name,
                        "group": "fishbone_candidate" if index in fish else "ordinary_control",
                        "previous_json": str(previous), "previous_json_sha256": digest(previous)})
    output.mkdir(parents=True)
    inputs = output / "inputs"
    inputs.mkdir()
    for record in records:
        (inputs / record["name"]).symlink_to(record["source_image"])
    write_json(output / "selection.json", records)
    environment = os.environ.copy()
    environment["UFLDV2_BINARY"] = str(args.binary.resolve())
    variants, all_rows = [], []
    for color, enhance, ratio in itertools.product(("bgr", "rgb"), ("none", "clahe"), (0.5, 0.4, 0.3)):
        name = f"{color}_{enhance}_r{int(ratio * 100)}"
        folder = output / name
        folder.mkdir()
        command = [str(project / "scripts/run_cpp_gpu.sh"), "--model", str(args.model.resolve()),
                   "--image-dir", str(inputs), "--output-dir", str(folder), "--device", "cuda", "--save-points",
                   "--input-color", color, "--enhance", enhance, "--row-min-ratio", str(ratio), "--col-min-ratio", "0.25"]
        print("Running", name, flush=True)
        with (folder / "run.log").open("w") as log:
            completed = subprocess.run(command, env=environment, cwd=project, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"{name} failed with exit {completed.returncode}: see {folder / 'run.log'}")
        profiles = list((folder / "profiles").glob("*.json"))
        cuda_events = sum(event.get("args", {}).get("provider") == "CUDAExecutionProvider"
                          for profile in profiles for event in json.loads(profile.read_text()))
        if not cuda_events:
            raise RuntimeError(f"No CUDA evidence for {name}")
        rows = []
        for record in records:
            result = json.loads((folder / (record["name"] + "_result.json")).read_text())
            for lane in result["lanes"]:
                assert len(lane) >= 1
                for x, y in lane:
                    assert 0 <= x < result["image_width"] and 0 <= y < result["image_height"]
            row = {"variant": name, "index": record["index"], "group": record["group"],
                   "name": record["name"], "lanes": len(result["lanes"]),
                   "points": sum(map(len, result["lanes"])), "diagnostics": result["lane_diagnostics"]}
            rows.append(row)
        groups = {}
        for group in ("fishbone_candidate", "ordinary_control"):
            selected_rows = [row for row in rows if row["group"] == group]
            groups[group] = {"frames": len(selected_rows), "nonempty_frames": sum(row["lanes"] > 0 for row in selected_rows),
                             "total_lanes": sum(row["lanes"] for row in selected_rows),
                             "inner_slots_present": sum(d["exported_points"] > 0 for row in selected_rows
                                                        for d in row["diagnostics"] if d["branch"] == "row")}
        variant = {"name": name, "color": color, "enhance": enhance, "row_min_ratio": ratio,
                   "col_min_ratio": 0.25, "groups": groups, "cuda_kernel_events": cuda_events,
                   "command": command}
        variants.append(variant)
        all_rows.extend(rows)
        write_json(output / "progress.json", {"completed_variants": len(variants), "total_variants": 12,
                                               "latest_variant": name, "variants": variants})
        print(name, groups, flush=True)
    mismatches = []
    for record in records:
        previous = Path(record["previous_json"])
        assert digest(previous) == record["previous_json_sha256"]
        old = json.loads(previous.read_text())
        new = json.loads((output / "bgr_none_r50" / (record["name"] + "_result.json")).read_text())
        if old["lanes"] != new["lanes"]:
            mismatches.append(record["name"])
    write_json(output / "frame_results.json", all_rows)
    summary = {"note": "Output coverage/counts are NOT measured recall/precision; no reviewed ground truth used.",
               "model": str(args.model.resolve()), "binary_sha256": digest(args.binary.resolve()),
               "production_binary_sha256": digest(project / "build/ufldv2_gpu"),
               "selected_frames": len(records), "variants": variants,
               "baseline_point_mismatch_images": mismatches, "source_json_hashes_unchanged": True,
               "crop_ratio": 0.6, "training_performed": False}
    write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "group", "frames", "nonempty_frames", "total_lanes", "inner_slots_present"])
        for variant in variants:
            for group, values in variant["groups"].items():
                writer.writerow([variant["name"], group, *values.values()])
    style = "body{font-family:sans-serif;margin:24px;background:#f7f8f9;color:#182c35}table{border-collapse:collapse;background:white}td,th{border:1px solid #ccd3d6;padding:8px}a{color:#075e91}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.panel{background:white;padding:10px}.panel img{width:100%}p{line-height:1.5}"
    rows_by_key = {(row["variant"], row["index"]): row for row in all_rows}
    index_parts = [f"<!doctype html><meta charset='utf-8'><title>预处理/后处理对照</title><style>{style}</style>",
                   "<h1>预处理 / 后处理对照实验</h1><p>计数代表输出覆盖，不是召回率或准确率。鱼骨候选仅为场景分组，未提供人工真值。</p><table><tr><th>配置</th><th>鱼骨候选非空帧</th><th>鱼骨候选线数</th><th>普通对照线数</th></tr>"]
    for variant in variants:
        f, c = variant["groups"]["fishbone_candidate"], variant["groups"]["ordinary_control"]
        index_parts.append(f"<tr><td>{variant['name']}</td><td>{f['nonempty_frames']}/{f['frames']}</td><td>{f['total_lanes']}</td><td>{c['total_lanes']}</td></tr>")
    index_parts.append("</table><h2>逐图查看（全部 12 种配置）</h2><ul>")
    for record in records:
        filename = f"frame_{record['index']:04d}.html"
        index_parts.append(f"<li><a href='{filename}'>index {record['index']} · {record['group']}</a></li>")
        page = [f"<!doctype html><meta charset='utf-8'><title>Frame {record['index']}</title><style>{style}</style><a href='index.html'>返回统计</a>",
                f"<h1>Frame {record['index']} · {record['group']}</h1><p>{html.escape(record['name'])}</p>",
                f"<details><summary>查看原图</summary><img width='100%' src='inputs/{html.escape(record['name'],quote=True)}'></details><div class='grid'>"]
        for variant in variants:
            row = rows_by_key[variant["name"], record["index"]]
            diagnostics = "; ".join(f"slot {d['slot']}: {d['valid_anchors']}/{d['total_anchors']} → {d['exported_points']}点" for d in row["diagnostics"])
            picture = f"{variant['name']}/{record['name']}_result.jpg"
            page.append(f"<div class='panel'><b>{variant['name']} · {row['lanes']}条线</b><a href='{picture}'><img loading='lazy' src='{picture}'></a><p>{diagnostics}</p></div>")
        page.append("</div>")
        (output / filename).write_text("".join(page))
    index_parts.append("</ul>")
    (output / "index.html").write_text("".join(index_parts))
    print("Finished", len(variants), "variants; baseline mismatches:", len(mismatches), flush=True)


if __name__ == "__main__":
    main()
