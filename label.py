"""Review YOLO detections and assign operator labels by clicking or pressing a number."""

import argparse
import base64
from collections import defaultdict, deque
from datetime import datetime, timezone
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shlex
import time
from urllib.request import Request, urlopen
from urllib.error import URLError
from urllib.error import HTTPError

from PIL import Image, ImageDraw


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


def load_project_env(path=None):
    """Load Gemini settings from the project's .env without shell execution."""
    path = Path(path) if path is not None else Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, raw = line.partition("=")
        key = key.strip()
        if not separator or key not in ("GEMINI_API_KEY", "GEMINI_MODEL"):
            continue
        try:
            values = shlex.split(raw, comments=True, posix=True)
        except ValueError:
            raise ValueError(f"Invalid quoting in .env at line {number}.") from None
        if len(values) > 1:
            raise ValueError(f"Quote values containing spaces in .env at line {number}.")
        value = values[0] if values else ""
        if not os.environ.get(key):
            os.environ[key] = value


def read_labels(frame_dir):
    path = frame_dir / "labels.json"
    return json.loads(path.read_text()) if path.exists() else {}


def frame_annotations(frame_dir):
    """Keep detector output intact; manual annotations override its boxes."""
    path = frame_dir / "annotations.json"
    return json.loads(path.read_text()) if path.exists() else json.loads((frame_dir / "results.json").read_text())


def read_box_presets(recordings):
    path = Path(recordings) / "box_presets.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_box_preset(recordings, name, sample):
    name = name.strip()
    if not name:
        raise ValueError("Give the preset a name.")
    with Image.open(sample["frame_path"]) as image:
        width, height = image.size
    bbox = sample["bbox"]
    presets = read_box_presets(recordings)
    presets[name] = {"bbox_normalized": [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]}
    path = Path(recordings) / "box_presets.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(presets, indent=2) + "\n")
    temporary.replace(path)
    return presets


def apply_box_preset(sample, preset, add=False):
    """Apply a saved rectangle to one question, leaving other boxes intact."""
    frame_dir = sample["frame_dir"]
    report = frame_annotations(frame_dir)
    with Image.open(sample["frame_path"]) as image:
        width, height = image.size
    values = preset["bbox_normalized"]
    bbox = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    boxes = [{"train_id": d["train_id"], "bbox": d["bbox"]} for d in report["detections"]]
    if add:
        train_id = max(report.get("next_train_id", 1), max((int(b["train_id"]) for b in boxes), default=0) + 1)
        boxes.append({"train_id": train_id, "bbox": bbox})
    else:
        train_id = int(sample["train_id"])
        target = next((b for b in boxes if int(b["train_id"]) == train_id), None)
        if target is None:
            raise ValueError("This question no longer has a box.")
        if all(abs(a-b) < 1e-6 for a, b in zip(target["bbox"], bbox)):
            return str(train_id), report
        target["bbox"] = bbox
    return str(train_id), save_frame_boxes(frame_dir, boxes, reviewed=False)


def save_frame_boxes(frame_dir, boxes, reviewed=False):
    """Persist all visible train boxes and invalidate labels for changed crops."""
    original = json.loads((frame_dir / "results.json").read_text())
    previous = frame_annotations(frame_dir)
    prior = {str(d["train_id"]): d for d in previous["detections"]}
    labels = read_labels(frame_dir)
    history = list(previous.get("label_history", []))
    stamp = time.time_ns()
    detections = []
    ids = set()
    with Image.open(frame_dir / "frame.jpg") as source:
        image = source.convert("RGB")
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for box in boxes:
        train_id = str(box["train_id"])
        if not train_id.isdigit() or int(train_id) < 1 or train_id in ids:
            raise ValueError("Every box needs a unique positive ID.")
        ids.add(train_id)
        bbox = list(map(float, box["bbox"]))
        if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
            raise ValueError("Box coordinates must be four finite numbers.")
        bbox = [max(0, min(image.width, bbox[0])), max(0, min(image.height, bbox[1])),
                max(0, min(image.width, bbox[2])), max(0, min(image.height, bbox[3]))]
        if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
            raise ValueError("Each box must be at least 2 pixels wide and high.")
        detection = dict(prior.get(train_id, {}))
        changed = detection.get("bbox") != bbox
        if changed and train_id in labels:
            history.append({"reason": "box_changed", "label": labels.pop(train_id)})
        crop_bbox = detection.get("crop_bbox", [math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2]), math.ceil(bbox[3])])
        crop_path = frame_dir / Path(detection.get("training_crop_path", f"frame_train_{train_id}.png")).name
        if not crop_path.exists() and not changed:
            crop_path = crop_path.with_suffix(".jpg")
        if changed or not crop_path.exists():
            crop_bbox = [math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2]), math.ceil(bbox[3])]
            crop_path = frame_dir / f"frame_train_{train_id}_corrected_{stamp}.png"
            image.crop(tuple(crop_bbox)).save(crop_path)
        if changed and "detection_confidence" in detection:
            detection["original_detection_confidence"] = detection.pop("detection_confidence")
        detection.update({"train_id": int(train_id), "bbox": bbox, "crop_bbox": crop_bbox,
            "crop_path": str(crop_path), "training_crop_path": str(crop_path),
            "dimensions": [crop_bbox[2] - crop_bbox[0], crop_bbox[3] - crop_bbox[1]],
            "question_id": f"{frame_dir.parent.parent.name}/{frame_dir.name}/train_{train_id}",
            "annotation_source": "human", "activity": detection.get("activity", "manual")})
        if train_id in labels:
            labels[train_id]["training_crop_path"] = str(crop_path)
        detections.append(detection)
        draw.rectangle(tuple(bbox), outline="lime", width=3)
        draw.text((bbox[0], bbox[1]), f"Train {train_id}", fill="lime")
    for train_id in set(labels) - ids:
        history.append({"reason": "box_deleted", "label": labels.pop(train_id)})
    annotation = {**original, "detections": detections, "label_history": history,
        "boxes_reviewed": reviewed, "annotation_source": "human",
        "next_train_id": max(previous.get("next_train_id", 1),
            max((int(d["train_id"]) for d in original["detections"] + detections), default=0) + 1),
        "edited_at_utc": datetime.now(timezone.utc).isoformat()}
    preview.save(frame_dir / "frame_corrected.jpg", quality=95)
    for filename, payload in (("labels.json", labels), ("annotations.json", annotation)):
        path = frame_dir / filename
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(path)
    return annotation


def load_samples(root):
    samples = []
    for report_path in sorted(Path(root).glob("*/frames/*/results.json")):
        try:
            report = frame_annotations(report_path.parent)
        except (OSError, ValueError):
            # A recorder may still be finishing this report. Try on the next scan.
            continue
        frame_dir = report_path.parent
        labels = read_labels(frame_dir)
        for detection in report.get("detections", []):
            train_id = str(detection["train_id"])
            existing = labels.get(train_id)
            if existing and detection.get("annotation_source") == "human" and existing.get("bbox") != detection["bbox"]:
                existing = None
            crop_path = frame_dir / f"frame_train_{train_id}.png"
            if detection.get("annotation_source") == "human":
                crop_path = frame_dir / Path(detection["training_crop_path"]).name
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
        # The timeline keeps raw detection scores, but question links follow edits.
        overrides = {}
        for annotation_path in run_dir.glob("frames/*/annotations.json"):
            try:
                annotation = json.loads(annotation_path.read_text())
            except (OSError, ValueError):
                continue
            if annotation.get("sample_index") is not None:
                overrides[annotation["sample_index"]] = [d["question_id"] for d in annotation["detections"]]
        return [{**point, "question_ids": overrides[point["sample_index"]]}
                if point.get("sample_index") in overrides else dict(point) for point in points]


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


def winnow_request(sample):
    with Image.open(sample["crop_path"]) as source:
        crop = source.convert("RGB")
        crop.thumbnail((960, 960))
        buffer = BytesIO()
        crop.save(buffer, format="JPEG", quality=90)
    return {
        "model": "Winnow-E2B",
        "state": "Inspect the attached cropped rail-camera image. Use visible image evidence only.",
        "questions": {
            "presence": {"type": "choice", "instructions": "Is any actual train visible, including partially visible railcars? Lights, platforms, wires and buildings alone are not trains.",
                         "criteria": {"train": "A train or part of a train is visible", "no_train": "No train is visible"}},
            "operator": {"type": "choice", "instructions": "Identify the operator of the visible train. Choose unclear if the image is ambiguous, unidentifiable, or contains multiple operators.",
                         "criteria": {**{choice[2]: choice[1] for choice in CHOICES if choice[2]},
                                      "no_train": "No train is visible", "unclear": "Cannot identify one operator reliably"}},
        },
        "winnow": {"images": ["data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()], "diagnostics": True},
    }


def winnow_cases(samples, limit):
    """Balance classes and thin neighboring frames for a small pilot."""
    groups = defaultdict(deque)
    seen = set()
    for sample in samples:
        label = sample.get("label")
        if not label or label["status"] not in ("labeled", "false_detection"):
            continue
        expected = label["operator"] if label["status"] == "labeled" else "no_train"
        if not expected:
            continue
        seconds = sample.get("stream_seconds")
        bucket = int(seconds // 30) if seconds is not None else sample["sample_id"]
        key = (sample["run_id"], bucket, expected)
        if key in seen:
            continue
        seen.add(key)
        groups[expected].append((sample, expected))
    cases = []
    while len(cases) < limit and any(groups.values()):
        for group in groups.values():
            if group and len(cases) < limit:
                cases.append(group.popleft())
    return cases


def test_winnow(recordings, output_dir, limit=60, server=None, timeout=60):
    cases = winnow_cases(load_samples(recordings), limit)
    if not cases:
        raise ValueError("No confirmed operator labels or false detections are available.")
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "requests.jsonl"
    results = []
    with request_path.open("w") as requests, (output_dir / "responses.jsonl").open("w") as responses:
        for index, (sample, expected) in enumerate(cases, 1):
            payload = winnow_request(sample)
            case = {"question_id": sample["question_id"], "expected": expected, "request": payload}
            requests.write(json.dumps(case) + "\n")
            if server is None:
                continue
            request = Request(server.rstrip("/") + "/v1/systemone", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            if "e2b" not in result.get("model", "").lower():
                raise ValueError(f"Expected Winnow-E2B but server reports {result.get('model')!r}.")
            answers = result["answers"]
            probability = float(answers["presence"]["probabilities"]["train"])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Invalid train probability from server.")
            entry = {"question_id": sample["question_id"], "expected": expected,
                     "train_probability": probability, "predicted_operator": answers["operator"]["choice"], "response": result}
            responses.write(json.dumps(entry) + "\n")
            responses.flush()
            results.append(entry)
            print(f"{index}/{len(cases)}: expected={expected}, operator={entry['predicted_operator']}, train={probability:.3f}", flush=True)
    summary = {"model": "Winnow-E2B", "mode": "native direct scoring; no adaptive reasoning",
               "prepared_cases": len(cases), "evaluated_cases": len(results), "server": server,
               "sampling": "balanced classes, at most one example per class/run/30-second interval; pilot, not an independent held-out evaluation",
               "classes": {key: sum(expected == key for _, expected in cases) for key in sorted({expected for _, expected in cases})}}
    if results:
        positives = [r for r in results if r["expected"] != "no_train"]
        negatives = [r for r in results if r["expected"] == "no_train"]
        summary["presence_thresholds"] = {str(t): {"real_trains_kept": sum(r["train_probability"] >= t for r in positives),
            "real_trains_total": len(positives), "false_detections_rejected": sum(r["train_probability"] < t for r in negatives),
            "false_detections_total": len(negatives)} for t in (.5, .6, .7, .8, .9, .95)}
        summary["operator_correct"] = sum(r["predicted_operator"] == r["expected"] for r in positives)
        summary["operator_total"] = len(positives)
        confusion = defaultdict(lambda: defaultdict(int))
        for result in positives:
            confusion[result["expected"]][result["predicted_operator"]] += 1
        summary["operator_confusion"] = dict(confusion)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Prepared {len(cases)} cases; evaluated {len(results)}. Saved in {output_dir}", flush=True)
    return summary


def gemini_image_part(path, max_side=1920):
    """Encode a clean target or reference image and retain only size metadata."""
    with Image.open(path) as source:
        original_size = list(source.size)
        image = source.convert("RGB")
        image.thumbnail((max_side, max_side))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
    encoded = buffer.getvalue()
    if len(encoded) > 3 * 1024 * 1024:
        raise ValueError("Compressed image is too large; use a smaller image.")
    return ({"type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(encoded).decode()},
            {"original_size": original_size, "submitted_size": list(image.size), "jpeg_bytes": len(encoded)})


def gemini_frame_request(frame_path, context="", prompt_dir=None):
    prompt_dir = Path(prompt_dir) if prompt_dir is not None else Path(__file__).resolve().parent / "prompts/gemini"
    target, image_info = gemini_image_part(frame_path)
    operators = {c[2]: c[1] for c in CHOICES if c[2]}
    descriptions = {
        "any_train": "Is any actual train or partially visible railcar present anywhere in the image?",
        **{key: f"Is at least one {name} train visibly present anywhere in the image?" for key, name in operators.items() if key != "other"},
        "other": "Is a train from an identifiable operator other than COASTER, Pacific Surfliner, Metrolink, MTS Trolley, or BNSF visibly present?",
        "multiple_trains": "Are two or more separate trains visible? Several cars belonging to one train count as one train.",
        "partial_at_bottom": "Is a partially visible train or railcar present at the bottom edge? Do not infer that it is stationary from this single image.",
        "unclear_operator": "Is at least one actual train visible whose operator cannot be reliably identified?",
    }
    prompt = (prompt_dir / "prompt.txt").read_text().strip() + "\n\n" + "\n".join(
        f"{key}: {question}" for key, question in descriptions.items())
    if context:
        prompt += "\nAdditional scene context: " + context
    parts = [{"type": "text", "text": prompt}, {"type": "text", "text": "TARGET FRAME: classify this image only."}, target]
    references = json.loads((prompt_dir / "references.json").read_text())
    if not isinstance(references, list) or len(references) > 12:
        raise ValueError("references.json must contain a list of at most 12 reference examples.")
    image_info["references"] = []
    for number, reference in enumerate(references, 1):
        operator = reference["operator"]
        if operator not in operators:
            raise ValueError(f"Unknown reference operator: {operator!r}.")
        path = (prompt_dir / reference["file"]).resolve()
        if not path.is_relative_to(prompt_dir.resolve()):
            raise ValueError("Reference images must be inside the Gemini prompt folder.")
        part, info = gemini_image_part(path, max_side=1280)
        text = f"REFERENCE EXAMPLE {number}: {operators[operator]}. {reference.get('description', '')} Not part of the target scene."
        parts.extend([{"type": "text", "text": text}, part])
        image_info["references"].append({**reference, **info})
    schema = {"type": "object", "properties": {
        "percentages": {"type": "object", "properties": {
            key: {"type": "number", "minimum": 0, "maximum": 100, "description": question}
            for key, question in descriptions.items()}, "required": list(descriptions)},
        "explanation": {"type": "string"}}, "required": ["percentages", "explanation"]}
    payload = {"model": os.environ.get("GEMINI_MODEL", "").strip() or "gemini-3.5-flash-lite",
        "store": False,
        "input": parts,
        "response_format": {"type": "text", "mime_type": "application/json", "schema": schema}}
    if len(json.dumps(payload).encode()) > 18 * 1024 * 1024:
        raise ValueError("Gemini request is too large; reduce the number or size of reference images.")
    return payload, image_info


def test_gemini_frame(frame_path, output_dir, context="", timeout=60):
    """Request a structured full-frame opinion, without changing human labels."""
    load_project_env()
    token = os.environ.get("GEMINI_API_KEY", "").strip()
    if not token:
        raise ValueError("Fill GEMINI_API_KEY in the project's .env file; it loads automatically.")
    payload, image_info = gemini_frame_request(frame_path, context)
    request = Request("https://generativelanguage.googleapis.com/v1beta/interactions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": token})
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        exc.close()
        hints = {400: "Check the model and request settings.", 401: "Check GEMINI_API_KEY.",
                 403: "Check the API key permissions and Gemini access.", 404: "Check GEMINI_MODEL availability.",
                 429: "Gemini quota reached; wait and try again or check AI Studio limits."}
        raise ValueError(f"Gemini HTTP {exc.code}: " + hints.get(exc.code, "Try again later.")) from None
    if result.get("status") != "completed":
        raise ValueError("Gemini did not complete the request; no predictions were saved.")
    text = "".join(part.get("text", "") for step in result.get("steps", [])
                   if step.get("type") == "model_output" for part in step.get("content", [])
                   if part.get("type") == "text")
    try:
        prediction = json.loads(text)
        percentages = prediction["percentages"]
        required = payload["response_format"]["schema"]["properties"]["percentages"]["required"]
        for key in required:
            value = percentages[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f"Invalid presence estimate for {key}.")
        explanation = prediction["explanation"]
        if not isinstance(explanation, str):
            raise ValueError("Invalid explanation from Gemini.")
    except (KeyError, TypeError, json.JSONDecodeError):
        raise ValueError("Gemini returned missing or malformed structured predictions.") from None
    report = {"model": result.get("model", payload["model"]), "requested_model": payload["model"],
        "frame_path": str(Path(frame_path).resolve()), "tested_at_utc": datetime.now(timezone.utc).isoformat(),
        "image": image_info, "percentages": {key: round(percentages[key], 2) for key in required},
        "explanation": explanation, "prompt": payload["input"][0]["text"], "result": result,
        "note": "Model-generated independent presence estimates; not calibrated probabilities and do not sum to 100%."}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "result.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report, output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, default=Path("recordings"))
    parser.add_argument("--include-skipped", action="store_true", help="Review previously skipped detections")
    parser.add_argument("--winnow-test", action="store_true", help="Prepare or run a Winnow-E2B crop pilot instead of opening the GUI")
    parser.add_argument("--winnow-url", help="Winnow server base URL; omit to export requests without inference")
    parser.add_argument("--winnow-limit", type=int, default=60)
    parser.add_argument("--winnow-output", type=Path)
    parser.add_argument("--winnow-timeout", type=float, default=60)
    parser.add_argument("--gemini-test", action="store_true", help="Test the latest full frame with Gemini")
    parser.add_argument("--gemini-frame", type=Path, help="Full frame to test; implies --gemini-test")
    parser.add_argument("--gemini-output", type=Path)
    parser.add_argument("--gemini-context", type=Path, help="Optional text file describing the camera scene")
    args = parser.parse_args()
    try:
        load_project_env()
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Could not load .env: {exc}\n")
    if (args.gemini_test or args.gemini_frame) and args.winnow_test:
        parser.error("Choose either the Gemini test or the Winnow test.")
    if args.gemini_test or args.gemini_frame:
        try:
            frame = args.gemini_frame or max(args.recordings.glob("*/frames/*/frame.jpg"), key=lambda p: p.stat().st_mtime, default=None)
            if frame is None:
                raise ValueError("No saved full frames found; pass --gemini-frame IMAGE.")
            context = args.gemini_context.read_text() if args.gemini_context else ""
            report, output_path = test_gemini_frame(frame, args.gemini_output or Path("outputs") / f"gemini_{time.time_ns()}", context)
            print(f"Full frame: {frame}")
            for key, percentage in report["percentages"].items():
                print(f"{key}: {percentage:.1f}%")
            print(report["explanation"])
            print(f"Saved: {output_path}")
        except (OSError, ValueError, KeyError, URLError) as exc:
            parser.exit(1, f"Gemini test failed: {exc}\n")
        return
    if args.winnow_test:
        if args.winnow_limit <= 0 or not math.isfinite(args.winnow_timeout) or args.winnow_timeout <= 0:
            parser.error("Winnow limit and timeout must be positive.")
        if args.winnow_url and not args.winnow_url.startswith(("http://", "https://")):
            parser.error("Winnow URL must start with http:// or https://.")
        output = args.winnow_output or Path("outputs") / f"winnow_{time.time_ns()}"
        try:
            test_winnow(args.recordings, output, args.winnow_limit, args.winnow_url, args.winnow_timeout)
        except (OSError, ValueError, KeyError, URLError) as exc:
            parser.exit(1, f"Winnow test failed: {exc}\n")
        return
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
