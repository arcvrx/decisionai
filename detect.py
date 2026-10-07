"""Find trains and save lossless crops plus an annotated preview."""

import math

from PIL import ImageChops, ImageDraw, ImageStat


def detect_trains(image, model, confidence):
    names = model.names
    names = names.items() if isinstance(names, dict) else enumerate(names)
    train_ids = [index for index, name in names if name.lower() == "train"]
    if not train_ids:
        raise ValueError("The YOLO model must include a 'train' detection class.")
    result = model.predict(
        source=image, classes=train_ids, conf=confidence, imgsz=960, verbose=False
    )[0]
    detections = []
    boxes = [] if result.boxes is None else result.boxes
    for box in boxes:
        left, top, right, bottom = box.xyxy[0].tolist()
        score = float(box.conf[0])
        left, top = max(0, left), max(0, top)
        right, bottom = min(image.width, right), min(image.height, bottom)
        if right <= left or bottom <= top:
            continue
        pad_x, pad_y = (right - left) * 0.1, (bottom - top) * 0.1
        crop_box = (
            max(0, math.floor(left - pad_x)),
            max(0, math.floor(top - pad_y)),
            min(image.width, math.ceil(right + pad_x)),
            min(image.height, math.ceil(bottom + pad_y)),
        )
        train_id = len(detections) + 1
        detections.append({
            "train_id": train_id,
            "detection_confidence": score,
            "bbox": [left, top, right, bottom],
            "crop_bbox": list(crop_box),
        })
    return detections


def export_detections(image, detections, output_dir, export_empty=False):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for detection in detections:
        output_dir.mkdir(parents=True, exist_ok=True)
        crop = image.crop(tuple(detection["crop_bbox"]))
        crop_path = output_dir / f"frame_train_{detection['train_id']}.png"
        crop.save(crop_path)
        detection.update({"crop_path": str(crop_path), "training_crop_path": str(crop_path), "dimensions": list(crop.size)})
        bbox = detection["bbox"]
        draw.rectangle(tuple(bbox), outline="lime", width=4)
        label = f"Train {detection['train_id']}: {detection['detection_confidence']:.2f}"
        position = (bbox[0], max(0, bbox[1] - 18))
        draw.rectangle(draw.textbbox(position, label), fill="black")
        draw.text(position, label, fill="lime")
    if not detections:
        if not export_empty:
            return None, []
        draw.text((10, 10), "No train detected", fill="red")
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / "frame_detected.jpg"
    preview.save(preview_path, quality=95)
    return preview_path, detections


def detect_and_export(image, model, output_dir, confidence, export_empty=False):
    return export_detections(image, detect_trains(image, model, confidence), output_dir, export_empty)


def box_overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if intersection else 0


class CaptureSampler:
    """Keep moving views, while sampling repeated stationary detections less often.

    Matching is only for capture cadence. It never groups labels or assigns
    train identities across screenshots.
    """

    def __init__(self, stationary_seconds=60, motion_threshold=6):
        self.stationary_seconds = stationary_seconds
        self.motion_threshold = motion_threshold
        self.previous_image = None
        self.previous = []

    def update(self, image, detections, seconds):
        gray = image.convert("L")
        previous = self.previous if self.previous_image is not None and self.previous_image.size == gray.size else []
        used = set()
        states = []
        for detection in detections:
            bbox = detection["bbox"]
            candidates = sorted(
                [(box_overlap(bbox, state["bbox"]), i) for i, state in enumerate(previous) if i not in used],
                reverse=True,
            )
            matched = candidates and candidates[0][0] >= 0.3
            if matched and len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.1:
                matched = False
            if matched:
                _, index = candidates[0]
                used.add(index)
                state = previous[index]
                old_box = state["bbox"]
                shift = math.hypot((bbox[0] + bbox[2] - old_box[0] - old_box[2]) / 2,
                                   (bbox[1] + bbox[3] - old_box[1] - old_box[3]) / 2)
                difference = ImageChops.difference(gray.crop(tuple(bbox)).resize((48, 48)),
                                                  self.previous_image.crop(tuple(bbox)).resize((48, 48)))
                motion = ImageStat.Stat(difference).mean[0]
                moving = shift > math.hypot(*image.size) * 0.008 or motion >= self.motion_threshold
                last_saved = state["last_saved"]
                activity = "moving" if moving else "stationary"
            else:
                motion = None
                activity = "new"
                last_saved = None
            save = activity != "stationary" or last_saved is None or seconds - last_saved >= self.stationary_seconds
            detection.update({"activity": activity, "motion_score": motion, "save_sample": save})
            states.append({"bbox": bbox, "last_saved": seconds if save else last_saved})
        self.previous_image = gray
        self.previous = states
        return detections
