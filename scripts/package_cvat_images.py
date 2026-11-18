#!/usr/bin/env python3
"""Package ONE CVAT task's original images without recompression or annotation changes."""
import argparse
import json
from pathlib import Path
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError(f"Refusing to overwrite {args.output}")
        items = json.loads((args.task_dir / "manifest.json").read_text())["images"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # ZIP_STORED preserves original compressed JPEG bytes and minimizes CPU work.
        with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_STORED) as archive:
            for item in items:
                name = item["image_name"]
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError(f"Unsafe archive member: {name}")
                archive.write(item["source_image"], arcname=name)
        print(f"Packaged {len(items)} original images: {args.output}")
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
