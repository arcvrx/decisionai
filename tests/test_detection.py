from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

from detect import CaptureSampler, detect_and_export, detect_trains
from record import save_detected_frame


class Coordinates:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class Detector:
    names = {6: "train", 2: "car"}

    def __init__(self, boxes):
        self.boxes = [SimpleNamespace(xyxy=[Coordinates(box)], conf=[0.9]) for box in boxes]

    def predict(self, **kwargs):
        if kwargs["classes"] != [6]:
            raise AssertionError("Only trains should be requested")
        return [SimpleNamespace(boxes=self.boxes)]


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "frame"
        self.image = Image.new("RGB", (200, 100), "blue")

    def test_positive_detections_export_separate_lossless_training_crops(self):
        preview, detections = detect_and_export(self.image, Detector([[20, 10, 80, 90], [110, 20, 180, 80]]), self.root, 0.25)
        self.assertTrue(preview.exists())
        self.assertEqual(len(detections), 2)
        self.assertNotEqual(detections[0]["crop_path"], detections[1]["crop_path"])
        for detection in detections:
            with Image.open(detection["crop_path"]) as saved:
                expected = self.image.crop(tuple(detection["crop_bbox"]))
                self.assertEqual(saved.format, "PNG")
                self.assertEqual(saved.size, expected.size)
                self.assertEqual(saved.tobytes(), expected.tobytes())
            self.assertNotIn("passage_id", detection)

    def test_negative_frames_do_not_create_screenshot_files(self):
        preview, detections = detect_and_export(self.image, Detector([]), self.root, 0.25)
        self.assertIsNone(preview)
        self.assertFalse(detections)
        self.assertFalse(self.root.exists())

    def test_edge_detections_are_padded_and_clipped_to_original_frame(self):
        _, detections = detect_and_export(self.image, Detector([[-5, -5, 205, 110]]), self.root, 0.25)
        self.assertEqual(detections[0]["crop_bbox"], [0, 0, 200, 100])
        self.assertEqual(detections[0]["dimensions"], [200, 100])

    def test_parked_train_is_recaptured_periodically(self):
        sampler = CaptureSampler(stationary_seconds=60)
        model = Detector([[20, 10, 80, 90]])
        states = []
        for seconds in (0, 1, 59, 60):
            detection = sampler.update(self.image, detect_trains(self.image, model, .25), seconds)[0]
            states.append((detection["activity"], detection["save_sample"]))
        self.assertEqual(states, [("new", True), ("stationary", False), ("stationary", False), ("stationary", True)])

    def test_moving_train_is_saved_alongside_unchanged_parked_train(self):
        sampler = CaptureSampler()
        sampler.update(self.image, detect_trains(self.image, Detector([[10, 10, 60, 90], [100, 10, 150, 90]]), .25), 0)
        detections = sampler.update(self.image, detect_trains(self.image, Detector([[10, 10, 60, 90], [110, 10, 160, 90]]), .25), 1)
        self.assertFalse(detections[0]["save_sample"])
        self.assertEqual(detections[1]["activity"], "moving")
        self.assertTrue(detections[1]["save_sample"])
        # Moving texture inside an unchanged box also counts as activity.
        changed = self.image.copy()
        changed.paste("white", (110, 10, 160, 90))
        detections = sampler.update(changed, detect_trains(changed, Detector([[10, 10, 60, 90], [110, 10, 160, 90]]), .25), 2)
        self.assertTrue(detections[1]["save_sample"])

    def test_history_logs_negatives_and_stationary_ticks_and_unique_question_ids(self):
        run = self.root / "run"
        sampler = CaptureSampler()
        boxes = [[20, 10, 80, 90]]
        save_detected_frame(self.image, Detector([]), run, 0, 1, .25, sampler)
        first = save_detected_frame(self.image, Detector(boxes), run, 1, 1, .25, sampler)
        self.assertIsNone(save_detected_frame(self.image, Detector(boxes), run, 2, 1, .25, sampler))
        moving = save_detected_frame(self.image, Detector([[30, 10, 90, 90]]), run, 3, 1, .25, sampler)
        ticks = [json.loads(line) for line in (run / "timeline.jsonl").read_text().splitlines()]
        self.assertEqual(len(ticks), 4)
        self.assertEqual(ticks[0]["presence_score"], 0)
        self.assertEqual(ticks[2]["train_count"], 1)
        self.assertEqual(ticks[2]["activity_score"], 0)
        self.assertFalse(ticks[2]["question_ids"])
        self.assertFalse((run / "frames" / "frame_00000002").exists())
        self.assertNotEqual(first["detections"][0]["question_id"], moving["detections"][0]["question_id"])


if __name__ == "__main__":
    unittest.main()
