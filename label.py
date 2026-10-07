"""Review YOLO detections and assign operator labels by clicking or pressing a number."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


CHOICES = [
    ("1", "NCTD COASTER", "coaster", "labeled"),
    ("2", "Pacific Surfliner", "pacific_surfliner", "labeled"),
    ("3", "Metrolink", "metrolink", "labeled"),
    ("4", "MTS Trolley", "mts_trolley", "labeled"),
    ("5", "BNSF freight", "bnsf", "labeled"),
    ("6", "Other operator", "other", "labeled"),
    ("7", "Unclear / can't identify", None, "unclear"),
    ("8", "False detection", None, "false_detection"),
    ("9", "Skip", None, "skipped"),
]


def read_labels(frame_dir):
    path = frame_dir / "labels.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_samples(root):
    samples = []
    for report_path in sorted(Path(root).glob("*/frames/*/results.json")):
        try:
            report = json.loads(report_path.read_text())
        except (OSError, ValueError):
            # A recorder may still be finishing this report. Try on the next scan.
            continue
        frame_dir = report_path.parent
        labels = read_labels(frame_dir)
        for detection in report.get("detections", []):
            train_id = str(detection["train_id"])
            existing = labels.get(train_id)
            crop_path = frame_dir / f"frame_train_{train_id}.png"
            if not crop_path.exists():
                crop_path = Path(detection.get("training_crop_path", ""))
            if not crop_path.is_file():
                crop_path = frame_dir / f"frame_train_{train_id}.jpg"
            samples.append({
                "frame_dir": frame_dir,
                "frame_path": frame_dir / "frame.jpg",
                "crop_path": crop_path,
                "train_id": train_id,
                "bbox": detection["bbox"],
                "sample_id": f"{frame_dir.parent.parent.name}/{frame_dir.name}/train_{train_id}",
                "stream_seconds": report.get("approx_stream_seconds"),
                "question_id": detection.get("question_id", f"{frame_dir.parent.parent.name}/{frame_dir.name}/train_{train_id}"),
                "run_id": frame_dir.parent.parent.name,
                "sample_index": report.get("sample_index"),
                "label": existing,
                "activity": detection.get("activity", "detected"),
            })
    return samples


def pending_samples(root, include_skipped=False):
    return [sample for sample in load_samples(root) if not sample["label"] or
            (include_skipped and sample["label"]["status"] == "skipped")]


class TimelineReader:
    """Read append-only telemetry without rereading complete runs on each poll."""

    def __init__(self):
        self.offsets = {}
        self.points = {}

    def read(self, run_dir, samples=()):
        run_dir = Path(run_dir)
        path = run_dir / "timeline.jsonl"
        if not path.exists():
            # Older runs only logged positives; do not invent empty-track samples.
            by_frame = {}
            for sample in samples:
                if sample["run_id"] != run_dir.name:
                    continue
                seconds = sample["stream_seconds"]
                if seconds is None:
                    continue
                point = by_frame.setdefault(seconds, {"stream_seconds": seconds, "presence_score": 1, "activity_score": 1, "train_count": 0, "active_count": 0, "question_ids": [], "legacy": True})
                point["question_ids"].append(sample["question_id"])
                point["train_count"] += 1
                point["active_count"] += 1
            return sorted(by_frame.values(), key=lambda p: p["stream_seconds"])
        key = str(path)
        if path.stat().st_size < self.offsets.get(key, 0):
            self.offsets[key] = 0
            self.points[key] = []
        points = self.points.setdefault(key, [])
        with path.open("rb") as source:
            source.seek(self.offsets.get(key, 0))
            while True:
                position = source.tell()
                line = source.readline()
                if not line or not line.endswith(b"\n"):
                    self.offsets[key] = position
                    break
                try:
                    points.append(json.loads(line))
                except ValueError:
                    pass
        return list(points)


def save_label(sample, choice):
    _, title, operator, status = choice
    labels = read_labels(sample["frame_dir"])
    labels[sample["train_id"]] = {
        "sample_id": sample["sample_id"],
        "question_id": sample.get("question_id", sample["sample_id"]),
        "operator": operator,
        "status": status,
        "choice": title,
        "labeled_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_crop_path": str(sample["crop_path"]),
        "bbox": sample["bbox"],
        "stream_seconds": sample["stream_seconds"],
    }
    path = sample["frame_dir"] / "labels.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(labels, indent=2) + "\n")
    temporary.replace(path)
    print(f"{sample['sample_id']}: {title}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, default=Path("recordings"))
    parser.add_argument("--include-skipped", action="store_true", help="Review previously skipped detections")
    args = parser.parse_args()
    try:
        import tkinter as tk
    except ImportError as exc:
        parser.exit(1, f"Tkinter is required. On Arch Linux install it with: sudo pacman -S tk\n{exc}\n")
    try:
        from gui import LabelReviewer

        root = tk.Tk()
        LabelReviewer(root, args.recordings, args.include_skipped)
        root.mainloop()
    except tk.TclError as exc:
        parser.exit(1, f"Could not open the Tkinter window; a desktop display is required: {exc}\n")


if __name__ == "__main__":
    main()
