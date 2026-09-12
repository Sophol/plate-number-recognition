"""Turn a dataset manifest into a Label Studio task file.

    python -m ml.labeling.import_tasks --dataset dataset/v1

Writes dataset/v1/label_studio_tasks.json, which imports into a Label Studio
project created with ml/labeling/plate_config.xml.

Frames are referenced, never uploaded: Label Studio serves them from disk via
LOCAL_FILES_DOCUMENT_ROOT, so a 40 GB dataset does not get copied into its
database. Each task carries its `group` through to the annotation, which keeps
the group-aware split working on the way back out.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="dataset version dir, e.g. dataset/v1")
    parser.add_argument("--document-root", type=Path, default=None,
                        help="LOCAL_FILES_DOCUMENT_ROOT the URLs are relative to "
                             "(default: the dataset dir's parent)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    manifest = args.dataset / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"{manifest} not found - run ml.datasets.prepare first")

    document_root = (args.document_root or args.dataset.parent).resolve()
    frames_dir = (args.dataset / "frames").resolve()
    try:
        prefix = frames_dir.relative_to(document_root)
    except ValueError:
        raise SystemExit(
            f"{frames_dir} is not inside --document-root {document_root}; "
            "Label Studio can only serve files under that root"
        ) from None

    tasks = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        tasks.append({
            "data": {
                "image": f"/data/local-files/?d={prefix / record['file']}",
                "group": record["group"],
                "source": record["source"],
                "frame_index": record["frame_index"],
            }
        })

    out = args.out or args.dataset / "label_studio_tasks.json"
    out.write_text(json.dumps(tasks, indent=1))

    print(f"wrote {out} ({len(tasks)} tasks)")
    print("\nStart Label Studio with the document root these URLs assume:")
    print(f"  export LOCAL_FILES_SERVING_ENABLED=true")
    print(f"  export LOCAL_FILES_DOCUMENT_ROOT={document_root}")
    print(f"  .venv-label/bin/label-studio start")
    print(f"\nThen: create a project, paste ml/labeling/plate_config.xml as the")
    print(f"labelling config, and import {out}.")


if __name__ == "__main__":
    main()
