#!/usr/bin/env python3
"""Compare existing same-frame BGR/RGB JSON; create deterministic SVG/HTML reports.

No model inference, image enhancement, fitting, training or annotation changes.
"""
import argparse
import base64
from collections import Counter
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_path(data):
    return Path(data["source_image"]).resolve(strict=True)


def verify_settings(data, color):
    # Legacy BGR JSON predates the settings field; the original fixed pipeline
    # used these defaults, as recorded in the C++ setup/validation documents.
    default = {"input_color": "bgr", "enhance": "none", "row_min_ratio": 0.5,
               "col_min_ratio": 0.25, "crop_ratio": 0.6}
    actual = data.get("inference_settings", default)
    expected = {**default, "input_color": color}
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Unexpected {color} setting {key}: {actual.get(key)!r}")


def validate_points(data):
    width, height = data["image_width"], data["image_height"]
    for lane in data["lanes"]:
        for x, y in lane:
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
                raise ValueError("Invalid original-image pixel coordinate")


def svg_start(width, height):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>text{font-family:"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#193c39}'
            '.title{font-size:28px;font-weight:bold}.label{font-size:22px}.small{font-size:18px}</style>']


def text(x, y, value, cls="label"):
    return f'<text x="{x}" y="{y}" class="{cls}">{html.escape(str(value))}</text>'


def photo_definition(frame):
    mime = "image/png" if frame["source"].suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(frame["source"].read_bytes()).decode()
    return (f'<image id="photo_{frame["index"]}" width="{frame["width"]}" height="{frame["height"]}" '
            f'xlink:href="data:{mime};base64,{data}"/>')


def panel(frame, color, x, y, width, height, zoom=False):
    original_width, original_height = frame["width"], frame["height"]
    top, span = (round(original_height * 0.42), round(original_height * 0.41)) if zoom else (0, original_height)
    result = [f'<svg x="{x}" y="{y}" width="{width}" height="{height}" '
              f'viewBox="0 {top} {original_width} {span}" preserveAspectRatio="xMidYMid meet" overflow="hidden">',
              f'<use xlink:href="#photo_{frame["index"]}"/>']
    # Same drawing style for both inputs, no semantic slot correspondence
    # invented from the original compact lanes list. Only existing points used.
    for lane in frame[color]["lanes"]:
        points = " ".join(f"{x},{y}" for x, y in lane)
        result.append(f'<polyline points="{points}" fill="none" stroke="#00ff87" '
                      'stroke-width="9" stroke-linejoin="round"/>')
        for px, py in lane:
            result.append(f'<circle cx="{px}" cy="{py}" r="6" fill="#00ff87"/>')
    result.append('</svg>')
    return "".join(result)


def detail_svg(frame):
    result = svg_start(1840, 1120)
    result.extend(['<defs>', photo_definition(frame), '</defs>',
        text(20, 38, f'同帧对照：索引 {frame["index"]} / 第 {frame["index"] + 1} 张（索引从 0 起）', 'title'),
        text(20, 73, frame["name"], 'small')])
    for color, x in (("bgr", 20), ("rgb", 950)):
        lanes = frame[color]["lanes"]
        result.append(text(x, 112, f'{color.upper()} · {len(lanes)} 条输出 · {sum(map(len, lanes))} 个点'))
        result.append(panel(frame, color, x, 130, 870, 490))
        result.append(text(x, 665, '道路区域局部放大（同一裁剪范围）'))
        result.append(panel(frame, color, x, 680, 870, 202, zoom=True))
    result.extend([text(20, 940, '控制变量：同一原图、同一 Res18；无 CLAHE；row=0.5、col=0.25；仅输入通道顺序不同。', 'small'),
        text(20, 980, '绿点是原 JSON 点坐标；细线只连接已有点方便阅读，没有平滑、拟合、外推或补点。', 'small'),
        text(20, 1020, '背景均为同一张未增强原图。输出更多不代表更准确；没有人工真值，不计算召回率。', 'small'), '</svg>'])
    return "".join(result)


def overview_svg(frames, number):
    result = svg_start(1840, 1310)
    result.extend(['<defs>', *(photo_definition(frame) for frame in frames), '</defs>',
        text(20, 34, f'鱼骨路段关键帧 · BGR / RGB 同帧对照（{number}）', 'title'),
        text(20, 74, '左：BGR，无增强，row=0.5'), text(950, 74, '右：RGB，无增强，row=0.5')])
    for row, frame in enumerate(frames):
        y = 115 + row * 215
        for color, x in (("bgr", 20), ("rgb", 950)):
            result.append(text(x, y, f'索引 {frame["index"]}（第 {frame["index"]+1} 张） · {len(frame[color]["lanes"])} 条输出'))
            result.append(panel(frame, color, x, y + 15, 870, 175, zoom=True))
    result.extend([text(20, 1230, '仅连接实际预测点供目视比较；不是人工真值，也没有通过拟合生成新轨迹。', 'small'), '</svg>'])
    return "".join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bgr-dir", required=True, type=Path)
    parser.add_argument("--rgb-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--indices", default="1950,2000,2100,2500,3500,4500,6200,6350,7150,7800")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Refusing to overwrite comparison directory")
    bgr_paths = {path.name: path for path in args.bgr_dir.glob("*_result.json")}
    rgb_paths = {path.name: path for path in args.rgb_dir.glob("*_result.json")}
    if not bgr_paths or bgr_paths.keys() != rgb_paths.keys():
        parser.error("Both directories must contain the exact same nonempty set of prediction filenames")
    counts, transitions, records, hashes = [Counter(), Counter()], Counter(), [], []
    for name in sorted(bgr_paths):
        raw = [bgr_paths[name].read_bytes(), rgb_paths[name].read_bytes()]
        bgr, rgb = map(json.loads, raw)
        verify_settings(bgr, "bgr")
        verify_settings(rgb, "rgb")
        original = source_path(bgr)
        if original != source_path(rgb):
            raise ValueError(f"Different original images for {name}")
        if (bgr["image_width"], bgr["image_height"]) != (rgb["image_width"], rgb["image_height"]):
            raise ValueError(f"Different pixel dimensions for {name}")
        if Path(bgr["model_path"]).resolve() != Path(rgb["model_path"]).resolve():
            raise ValueError(f"Different models for {name}")
        validate_points(bgr)
        validate_points(rgb)
        a, b = len(bgr["lanes"]), len(rgb["lanes"])
        counts[0][a] += 1
        counts[1][b] += 1
        transitions[a, b] += 1
        records.append({"name": original.name, "source": str(original), "json_name": name,
                        "width": bgr["image_width"], "height": bgr["image_height"],
                        "bgr_lanes": a, "rgb_lanes": b,
                        "bgr_points": sum(map(len, bgr["lanes"])), "rgb_points": sum(map(len, rgb["lanes"]))})
        hashes.append({"json_name": name, "bgr_sha256": sha(raw[0]), "rgb_sha256": sha(raw[1])})
    records.sort(key=lambda row: row["name"])
    if len({record["name"] for record in records}) != len(records):
        raise ValueError("Ambiguous duplicate source basenames")
    indices = sorted(set(int(value) for value in args.indices.split(",")))
    if min(indices) < 0 or max(indices) >= len(records):
        parser.error("Index outside source image list")
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    frames = []
    selected = []
    for index in indices:
        row = records[index]
        frame = {**row, "index": index, "source": Path(row["source"]),
                 "bgr": json.loads(bgr_paths[row["json_name"]].read_text()),
                 "rgb": json.loads(rgb_paths[row["json_name"]].read_text())}
        with Image.open(frame["source"]) as image:
            if image.size != (frame["width"], frame["height"]):
                raise ValueError(f"Original image size changed: {frame['source']}")
        folder = output / f"frame_{index:04d}"
        folder.mkdir()
        svg = folder / "comparison.svg"
        svg.write_text(detail_svg(frame), encoding="utf-8")
        ET.parse(svg)
        item = {**row, "index_zero_based": index, "ordinal_one_based": index + 1,
                "bgr_json": str(bgr_paths[row["json_name"]].resolve()),
                "rgb_json": str(rgb_paths[row["json_name"]].resolve()),
                "point_counts_bgr": [len(lane) for lane in frame["bgr"]["lanes"]],
                "point_counts_rgb": [len(lane) for lane in frame["rgb"]["lanes"]]}
        write_json(folder / "frame_info.json", item)
        selected.append(item)
        frames.append(frame)
    for start in range(0, len(frames), 5):
        path = output / f"overview_{start//5+1:02d}.svg"
        path.write_text(overview_svg(frames[start:start+5], start//5+1), encoding="utf-8")
        ET.parse(path)
    with (output / "frame_counts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index_zero_based", "ordinal_one_based", "name", "bgr_lanes", "rgb_lanes", "bgr_points", "rgb_points"])
        writer.writerows((i, i+1, row["name"], row["bgr_lanes"], row["rgb_lanes"], row["bgr_points"], row["rgb_points"]) for i, row in enumerate(records))
    summary = {"paired_images": len(records), "keyframes": selected,
               "bgr_distribution": dict(sorted(counts[0].items())), "rgb_distribution": dict(sorted(counts[1].items())),
               "bgr_nonempty": sum(value for key, value in counts[0].items() if key > 0),
               "rgb_nonempty": sum(value for key, value in counts[1].items() if key > 0),
               "bgr_empty_rgb_nonempty": sum(value for (a,b), value in transitions.items() if a == 0 and b > 0),
               "bgr_nonempty_rgb_empty": sum(value for (a,b), value in transitions.items() if a > 0 and b == 0),
               "warning": "Counts are outputs, NOT correctness/recall. Keyframes were selected for qualitative fishbone-scene comparison, not a statistical test set.",
               "inference_performed": False, "source_results_modified": False}
    write_json(output / "summary.json", summary)
    write_json(output / "input_json_sha256.json", hashes)
    style = "body{font-family:sans-serif;margin:24px;background:#f6f8f7;color:#173833}table{border-collapse:collapse;background:white}td,th{border:1px solid #ccd5d1;padding:9px}img{max-width:100%}a{color:#08639a}p{line-height:1.6}"
    parts = [f"<!doctype html><meta charset='utf-8'><title>鱼骨路段 BGR / RGB 同帧对比</title><style>{style}</style>",
             "<h1>鱼骨路段 BGR / RGB 同帧关键帧对比</h1>",
             "<p>严格匹配同一原图与模型，均无 CLAHE、row=0.5、col=0.25。只读取已有结果，不重新推理。每行左 BGR、右 RGB；绿线连接原 JSON 点，仅供阅读，不做平滑或外推。</p>",
             f"<p>全量配对 {len(records)} 张：BGR 有输出 {summary['bgr_nonempty']} 张，RGB 有输出 {summary['rgb_nonempty']} 张。BGR 空→RGB 有输出 {summary['bgr_empty_rgb_nonempty']} 张；反向 {summary['bgr_nonempty_rgb_empty']} 张。<b>这些是数量，不是准确率。</b></p>",
             "<table><tr><th>索引（0起）</th><th>第几张（1起）</th><th>原始文件名</th><th>BGR 输出线数</th><th>RGB 输出线数</th><th>详细对照</th></tr>"]
    for item in selected:
        index = item["index_zero_based"]
        parts.append(f"<tr><td>{index}</td><td>{index+1}</td><td>{html.escape(item['name'])}</td><td>{item['bgr_lanes']}</td><td>{item['rgb_lanes']}</td><td><a href='frame_{index:04d}/comparison.svg'>全图＋道路放大</a> · <a href='frame_{index:04d}/frame_info.json'>对应信息</a></td></tr>")
    parts.append("</table>")
    for index in range((len(frames)+4)//5):
        parts.append(f"<h2>关键帧总览 {index+1}</h2><a href='overview_{index+1:02d}.svg'><img src='overview_{index+1:02d}.svg'></a>")
    parts.append("<h2>原始点状结果与 JSON</h2><ul>")
    for item in selected:
        links = []
        for color, folder in (("BGR", args.bgr_dir), ("RGB", args.rgb_dir)):
            for suffix in (".jpg", ".json"):
                path = (folder / item["json_name"]).with_suffix(suffix).resolve()
                relative = html.escape(os.path.relpath(path, output), quote=True)
                links.append(f"<a href='{relative}'>{color} {suffix}</a>")
        parts.append(f"<li>索引 {item['index_zero_based']}: {' · '.join(links)}</li>")
    parts.append("</ul><p>查看 frame_counts.csv 可核对全部图片的输出数量；不存在人工真值评价。</p>")
    (output / "index.html").write_text("".join(parts), encoding="utf-8")
    for item in hashes:
        assert sha(bgr_paths[item["json_name"]].read_bytes()) == item["bgr_sha256"]
        assert sha(rgb_paths[item["json_name"]].read_bytes()) == item["rgb_sha256"]
    print(json.dumps({key: value for key, value in summary.items() if key != "keyframes"}, ensure_ascii=False, indent=2))
    print("Saved", len(selected), "same-frame comparisons:", output / "index.html")


if __name__ == "__main__":
    main()
