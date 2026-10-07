import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from label import CHOICES, TimelineReader, load_samples, pending_samples, read_labels, save_label
from gui import ALL_RECORDINGS, LabelReviewer, review_counts


class LabelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.frame_dir = self.root / "run_001" / "frames" / "frame_00000001"
        self.frame_dir.mkdir(parents=True)
        Image.new("RGB", (100, 100), "blue").save(self.frame_dir / "frame.jpg")
        detections = []
        for train_id in (1, 2):
            Image.new("RGB", (20, 40), "blue").save(self.frame_dir / f"frame_train_{train_id}.png")
            detections.append({"train_id": train_id, "bbox": [10 * train_id, 0, 10 * train_id + 20, 40]})
        (self.frame_dir / "results.json").write_text(json.dumps({"approx_stream_seconds": 1, "detections": detections}))

    def test_two_trains_in_same_frame_get_independent_labels(self):
        first, second = pending_samples(self.root)
        save_label(first, CHOICES[0])
        remaining = pending_samples(self.root)
        self.assertEqual([s["train_id"] for s in remaining], ["2"])
        save_label(second, CHOICES[1])
        labels = read_labels(self.frame_dir)
        self.assertEqual(labels["1"]["operator"], "coaster")
        self.assertEqual(labels["2"]["operator"], "pacific_surfliner")
        self.assertFalse(pending_samples(self.root))

    def test_skipped_samples_can_be_reviewed_without_repeating_completed_labels(self):
        first, second = pending_samples(self.root)
        save_label(first, CHOICES[8])
        save_label(second, CHOICES[3])
        self.assertFalse(pending_samples(self.root))
        retry = pending_samples(self.root, include_skipped=True)
        self.assertEqual([s["train_id"] for s in retry], ["1"])

    def test_unclear_and_false_detection_do_not_create_operator_training_labels(self):
        first, second = pending_samples(self.root)
        save_label(first, CHOICES[6])
        save_label(second, CHOICES[7])
        labels = read_labels(self.frame_dir)
        self.assertIsNone(labels["1"]["operator"])
        self.assertIsNone(labels["2"]["operator"])
        self.assertEqual(labels["1"]["status"], "unclear")
        self.assertEqual(labels["2"]["status"], "false_detection")

    def test_incomplete_report_is_retried_after_recorder_finishes(self):
        report = self.frame_dir / "results.json"
        original = report.read_text()
        report.write_text('{"detections":')
        self.assertFalse(pending_samples(self.root))
        report.write_text(original)
        self.assertEqual(len(pending_samples(self.root)), 2)

    def reviewer(self):
        reviewer = LabelReviewer.__new__(LabelReviewer)
        reviewer.root = Mock()
        reviewer.messagebox = Mock()
        reviewer.closed = False
        reviewer.sample = pending_samples(self.root)[0]
        reviewer.label_overrides = {}
        reviewer.labeled_count = 0
        reviewer.poll_id = "pending_timer"
        reviewer.next_unlabeled = Mock()
        reviewer.draw_timeline = Mock()
        reviewer.unanswered_only = Mock()
        reviewer.unanswered_only.get.return_value = False
        return reviewer

    def test_button_callback_saves_label_and_advances(self):
        reviewer = self.reviewer()
        sample_id = reviewer.sample["sample_id"]
        reviewer.choose(CHOICES[2])
        self.assertEqual(read_labels(self.frame_dir)["1"]["operator"], "metrolink")
        self.assertEqual(reviewer.label_overrides[sample_id]["operator"], "metrolink")
        self.assertEqual(reviewer.sample["label"]["operator"], "metrolink")
        reviewer.next_unlabeled.assert_called_once()

    def test_failed_save_keeps_current_detection_for_retry(self):
        reviewer = self.reviewer()
        original = reviewer.sample
        with patch("gui.save_label", side_effect=OSError("Disk full")):
            reviewer.choose(CHOICES[0])
        self.assertIs(reviewer.sample, original)
        self.assertEqual(reviewer.labeled_count, 0)
        reviewer.messagebox.showerror.assert_called_once()
        reviewer.next_unlabeled.assert_not_called()

    def test_reviewed_samples_remain_browsable_and_can_be_relabelled(self):
        first = pending_samples(self.root)[0]
        save_label(first, CHOICES[0])
        samples = load_samples(self.root)
        self.assertEqual(len(samples), 2)
        save_label(samples[0], CHOICES[1])
        self.assertEqual(read_labels(self.frame_dir)["1"]["operator"], "pacific_surfliner")
        reviewer = self.reviewer()
        reviewer.visible = samples
        reviewer.sample = samples[0]
        reviewer.select = Mock()
        reviewer.navigate(1)
        reviewer.select.assert_called_once_with(samples[1])

    def test_timeline_reader_retries_incomplete_tail_without_duplicates(self):
        path = self.root / "run_001" / "timeline.jsonl"
        path.write_text('{"stream_seconds":0}\n{"stream_seconds":')
        reader = TimelineReader()
        self.assertEqual(reader.read(path.parent), [{"stream_seconds": 0}])
        with path.open("a") as output:
            output.write('1}\n')
        expected = [{"stream_seconds": 0}, {"stream_seconds": 1}]
        self.assertEqual(reader.read(path.parent), expected)
        self.assertEqual(reader.read(path.parent), expected)

    def test_legacy_timeline_does_not_invent_empty_samples(self):
        points = TimelineReader().read(self.root / "run_001", load_samples(self.root))
        self.assertEqual(len(points), 1)
        self.assertTrue(points[0]["legacy"])
        self.assertEqual(points[0]["train_count"], 2)

    def test_click_peak_selects_its_question_but_empty_interval_does_not(self):
        reviewer = self.reviewer()
        reviewer.visible = load_samples(self.root)
        reviewer.pixels_per_second = 3
        reviewer.capture_points = [{"stream_seconds": 1, "question_ids": [reviewer.visible[1]["question_id"]]}]
        reviewer.timeline = Mock()
        reviewer.timeline.canvasx.side_effect = lambda x: x
        reviewer.select = Mock()
        reviewer.status = Mock()
        event = Mock(x=33)
        reviewer.click_timeline(event)
        reviewer.select.assert_called_once_with(reviewer.visible[1])
        reviewer.select.reset_mock()
        reviewer.click_timeline(Mock(x=120))
        reviewer.select.assert_not_called()

    def test_existing_jpeg_crops_remain_reviewable(self):
        for train_id in (1, 2):
            path = self.frame_dir / f"frame_train_{train_id}.png"
            with Image.open(path) as image:
                image.save(path.with_suffix(".jpg"))
            path.unlink()
        samples = pending_samples(self.root)
        self.assertEqual(len(samples), 2)
        self.assertTrue(all(sample["crop_path"].suffix == ".jpg" for sample in samples))

    def test_review_counts_keep_skips_separate_from_answers(self):
        samples = [{"label": None}, {"label": {"status": "skipped"}},
                   {"label": {"status": "labeled"}}, {"label": {"status": "false_detection"}},
                   {"label": {"status": "unclear"}}]
        self.assertEqual(review_counts(samples), (3, 1, 1))

    def test_unanswered_filter_includes_skips_and_excludes_other_runs(self):
        reviewer = self.reviewer()
        reviewer.unanswered_only.get.return_value = True
        reviewer.include_skipped = False
        reviewer.run_id = Mock()
        reviewer.run_id.get.return_value = "run_001"
        reviewer.samples = [
            {"run_id": "run_001", "label": None},
            {"run_id": "run_001", "label": {"status": "skipped"}},
            {"run_id": "run_001", "label": {"status": "labeled"}},
            {"run_id": "run_002", "label": None},
        ]
        self.assertEqual(reviewer.filtered_samples(), reviewer.samples[:2])
        reviewer.unanswered_only.get.return_value = False
        self.assertEqual(reviewer.filtered_samples(), reviewer.samples[:3])

    def global_reviewer(self):
        reviewer = self.reviewer()
        reviewer.samples = [
            {"sample_id": "a", "run_id": "run_001", "label": None},
            {"sample_id": "b", "run_id": "run_001", "label": {"status": "labeled"}},
            {"sample_id": "c", "run_id": "run_002", "label": None},
        ]
        reviewer.sample = reviewer.samples[1]
        reviewer.include_skipped = False
        reviewer.run_id = Mock()
        reviewer.run_id.get.return_value = ALL_RECORDINGS
        reviewer.follow_live = Mock()
        reviewer.select = Mock()
        return reviewer

    def test_next_unlabeled_crosses_recordings_and_wraps_to_missed_question(self):
        reviewer = self.global_reviewer()
        reviewer.next_unlabeled = LabelReviewer.next_unlabeled.__get__(reviewer)
        reviewer.next_unlabeled()
        reviewer.select.assert_called_once_with(reviewer.samples[2])
        reviewer.run_id.set.assert_called_once_with(ALL_RECORDINGS)
        reviewer.sample = reviewer.samples[2]
        reviewer.sample["label"] = {"status": "labeled"}
        reviewer.select.reset_mock()
        reviewer.next_unlabeled()
        reviewer.select.assert_called_once_with(reviewer.samples[0])

    def test_global_pending_filter_removes_answered_questions(self):
        reviewer = self.global_reviewer()
        reviewer.unanswered_only.get.return_value = True
        self.assertEqual(reviewer.filtered_samples(), [reviewer.samples[0], reviewer.samples[2]])
        reviewer.samples[0]["label"] = {"status": "labeled"}
        self.assertEqual(reviewer.filtered_samples(), [reviewer.samples[2]])

    def server_reviewer(self):
        reviewer = self.reviewer()
        reviewer.recordings = self.root
        reviewer.image_server = None
        reviewer.server_thread = None
        reviewer.server_button = Mock()
        reviewer.status = Mock()
        return reviewer

    def test_image_server_starts_for_recordings_and_stops_cleanly(self):
        reviewer = self.server_reviewer()
        with patch("gui.ThreadingHTTPServer") as server_class, patch("gui.threading.Thread") as thread_class:
            reviewer.toggle_image_server()
            address, handler = server_class.call_args.args
            self.assertEqual(address, ("0.0.0.0", 8000))
            self.assertEqual(handler.keywords["directory"], str(self.root.resolve()))
            thread_class.return_value.start.assert_called_once()
            reviewer.messagebox.showinfo.assert_called_once()
            reviewer.toggle_image_server()
            server_class.return_value.shutdown.assert_called_once()
            server_class.return_value.server_close.assert_called_once()
            thread_class.return_value.join.assert_called_once()
            self.assertIsNone(reviewer.image_server)

    def test_image_server_port_conflict_leaves_labeler_usable(self):
        reviewer = self.server_reviewer()
        original = reviewer.sample
        with patch("gui.ThreadingHTTPServer", side_effect=OSError("Address already in use")):
            reviewer.toggle_image_server()
        self.assertIsNone(reviewer.image_server)
        self.assertIs(reviewer.sample, original)
        reviewer.messagebox.showerror.assert_called_once()


if __name__ == "__main__":
    unittest.main()
