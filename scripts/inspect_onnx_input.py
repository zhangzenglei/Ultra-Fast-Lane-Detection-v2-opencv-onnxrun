#!/usr/bin/env python3
"""Read-only ONNX input-contract inspection. Does not infer RGB/BGR from shape."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import onnx


def describe_input(value):
    tensor = value.type.tensor_type
    shape = [dim.dim_value if dim.HasField("dim_value") else dim.dim_param or "unknown"
             for dim in tensor.shape.dim]
    return {"name": value.name, "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
            "shape": shape, "doc_string": value.doc_string}


def inspect(path):
    model = onnx.load_model(str(path), load_external_data=False)
    graph = model.graph
    initializer_names = {value.name for value in graph.initializer}
    inputs = [value for value in graph.input if value.name not in initializer_names]
    consumers = defaultdict(list)
    for node in graph.node:
        for name in node.input:
            consumers[name].append(node)
    learned = {"Conv", "ConvTranspose", "Gemm", "MatMul"}
    paths = {}
    for value in inputs:
        pending = [(value.name, 0)]
        seen, chain = set(), []
        while pending and len(chain) < 20:
            name, depth = pending.pop(0)
            for node in consumers[name]:
                identity = tuple(node.output)
                if identity in seen:
                    continue
                seen.add(identity)
                attributes = {}
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.INT:
                        attributes[attr.name] = attr.i
                    elif attr.type == onnx.AttributeProto.INTS:
                        attributes[attr.name] = list(attr.ints)
                chain.append({"name": node.name, "op": node.op_type, "depth": depth,
                              "inputs": list(node.input), "outputs": list(node.output),
                              "integer_attributes": attributes})
                if node.op_type not in learned:
                    pending.extend((output, depth + 1) for output in node.output)
        paths[value.name] = chain
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return {"model": str(path.resolve()), "file_bytes": path.stat().st_size,
            "sha256": sha.hexdigest(), "producer_name": model.producer_name,
            "producer_version": model.producer_version, "ir_version": model.ir_version,
            "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
            "model_doc_string": model.doc_string, "graph_doc_string": graph.doc_string,
            "metadata_props": {item.key: item.value for item in model.metadata_props},
            "graph_metadata_props": {item.key: item.value for item in getattr(graph, "metadata_props", [])},
            "inputs": [describe_input(value) for value in inputs],
            "outputs": [describe_input(value) for value in graph.output],
            "input_paths_until_first_learned_layer": paths,
            "node_count": len(graph.node), "initializer_count": len(graph.initializer),
            "color_semantics_conclusion": "Not automatically identifiable from shape/Conv alone; inspect metadata and matched export/training pipeline.",
            "model_modified": False, "inference_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Refusing to overwrite report")
    report = inspect(args.model)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
