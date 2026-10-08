import json
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from label import (CHOICES, TimelineReader, apply_box_preset, frame_annotations, load_samples,
                   pending_samples, read_box_presets, read_labels, save_box_preset, save_frame_boxes, save_label,
                   test_winnow, winnow_cases, winnow_request)
from label import gemini_frame_request, load_project_env, test_gemini_frame
from gui import ALL_RECORDINGS, BoxEditor, LabelReviewer, review_counts


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
        reviewer.auto_preset = Mock()
        reviewer.auto_preset.get.return_value = False
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

    def test_box_correction_regenerates_crop_and_invalidates_only_changed_label(self):
        first, second = load_samples(self.root)
        save_label(first, CHOICES[0])
        save_label(second, CHOICES[1])
        original = (self.frame_dir / "results.json").read_text()
        boxes = [{"train_id": 1, "bbox": [5, 5, 25, 35]},
                 {"train_id": 2, "bbox": second["bbox"]},
                 {"train_id": 3, "bbox": [50, 10, 80, 70]}]
        report = save_frame_boxes(self.frame_dir, boxes, reviewed=True)
        samples = load_samples(self.root)
        self.assertEqual(len(samples), 3)
        self.assertIsNone(samples[0]["label"])
        self.assertEqual(samples[1]["label"]["operator"], "pacific_surfliner")
        self.assertEqual(samples[1]["crop_path"], second["crop_path"])
        self.assertIsNone(samples[2]["label"])
        with Image.open(samples[0]["crop_path"]) as crop:
            self.assertEqual(crop.size, (20, 30))
        self.assertEqual(report["next_train_id"], 4)
        self.assertTrue(report["boxes_reviewed"])
        self.assertEqual(report["label_history"][0]["label"]["operator"], "coaster")
        self.assertEqual((self.frame_dir / "results.json").read_text(), original)

    def test_delete_all_boxes_saves_explicit_reviewed_negative_without_reusing_ids(self):
        first = load_samples(self.root)[0]
        save_label(first, CHOICES[0])
        save_frame_boxes(self.frame_dir, [{"train_id": 3, "bbox": [0, 0, 30, 30]}])
        negative = save_frame_boxes(self.frame_dir, [], reviewed=True)
        self.assertFalse(load_samples(self.root))
        self.assertEqual(negative["detections"], [])
        self.assertTrue(negative["boxes_reviewed"])
        self.assertEqual(negative["next_train_id"], 4)
        self.assertFalse(read_labels(self.frame_dir))
        self.assertTrue((self.frame_dir / "frame.jpg").exists())

    def test_timeline_links_follow_corrected_question_ids(self):
        report_path = self.frame_dir / "results.json"
        report = json.loads(report_path.read_text())
        report["sample_index"] = 1
        report_path.write_text(json.dumps(report))
        run = self.root / "run_001"
        (run / "timeline.jsonl").write_text(json.dumps({"sample_index": 1, "stream_seconds": 1, "question_ids": ["old"]}) + "\n")
        reader = TimelineReader()
        reader.read(run)
        annotation = save_frame_boxes(self.frame_dir, [{"train_id": 3, "bbox": [0, 0, 40, 40]}])
        points = reader.read(run)
        self.assertEqual(points[0]["question_ids"], [annotation["detections"][0]["question_id"]])
        save_frame_boxes(self.frame_dir, [], reviewed=True)
        self.assertEqual(reader.read(run)[0]["question_ids"], [])

    def test_scaled_editor_resize_and_new_box_preserve_image_coordinates(self):
        editor = BoxEditor.__new__(BoxEditor)
        editor.image = Image.new("RGB", (1000, 500))
        editor.sx = editor.sy = .5
        editor.ox, editor.oy = 100, 50
        editor.boxes = [{"train_id": 1, "bbox": [100, 100, 300, 300]}]
        editor.active = editor.boxes[0]
        editor.adding = False
        editor.drag = None
        editor.next_id = 2
        editor.draw = Mock()
        editor.canvas = Mock()
        editor.reviewed = Mock()
        editor.press(Mock(x=250, y=200))  # bottom-right handle
        editor.motion(Mock(x=300, y=225))
        self.assertEqual(editor.active["bbox"], [100, 100, 400, 350])
        editor.release()
        editor.new_box()
        editor.press(Mock(x=450, y=250))
        editor.motion(Mock(x=350, y=150))  # reverse drag normalized
        self.assertEqual(editor.active["bbox"], [500, 200, 700, 400])
        self.assertEqual(editor.active["train_id"], 2)
        editor.delete_box()
        self.assertEqual(len(editor.boxes), 1)

    def test_presets_persist_and_scale_to_full_frame_dimensions(self):
        source = load_samples(self.root)[0]
        presets = save_box_preset(self.root, "left train", source)
        self.assertEqual(read_box_presets(self.root), presets)
        another = self.root / "run_002" / "frames" / "frame_00000010"
        another.mkdir(parents=True)
        Image.new("RGB", (200, 300)).save(another / "frame.jpg")
        (another / "results.json").write_text(json.dumps({"detections": [{"train_id": 1, "bbox": [0, 0, 10, 10]}]}))
        target = {"frame_dir": another, "frame_path": another / "frame.jpg", "train_id": "1"}
        train_id, report = apply_box_preset(target, presets["left train"])
        self.assertEqual(train_id, "1")
        self.assertEqual(report["detections"][0]["bbox"], [20, 0, 60, 120])
        self.assertFalse(report["boxes_reviewed"])

    def test_preset_replaces_one_box_or_adds_a_new_question(self):
        first, second = load_samples(self.root)
        save_label(first, CHOICES[0])
        save_label(second, CHOICES[1])
        preset = {"bbox_normalized": [.5, .1, .8, .7]}
        _, report = apply_box_preset(second, preset)
        self.assertEqual(report["detections"][0]["bbox"], first["bbox"])
        self.assertEqual(read_labels(self.frame_dir)["1"]["operator"], "coaster")
        self.assertNotIn("2", read_labels(self.frame_dir))
        new_id, report = apply_box_preset(second, preset, add=True)
        self.assertEqual(new_id, "3")
        self.assertEqual(len(report["detections"]), 3)
        self.assertEqual(load_samples(self.root)[2]["train_id"], "3")

    def test_auto_preset_failure_does_not_save_operator_label(self):
        reviewer = self.reviewer()
        reviewer.auto_preset.get.return_value = True
        reviewer.apply_preset = Mock(return_value=False)
        reviewer.choose(CHOICES[0])
        self.assertFalse(read_labels(self.frame_dir))
        reviewer.next_unlabeled.assert_not_called()

    def test_auto_preset_does_not_rewrite_false_detections_or_skips(self):
        reviewer = self.reviewer()
        reviewer.auto_preset.get.return_value = True
        reviewer.apply_preset = Mock()
        reviewer.choose(CHOICES[7])
        self.assertEqual(read_labels(self.frame_dir)["1"]["status"], "false_detection")
        reviewer.apply_preset.assert_not_called()

    def test_winnow_request_contains_image_and_no_human_answer(self):
        sample = load_samples(self.root)[0]
        request = winnow_request(sample)
        self.assertEqual(request["model"], "Winnow-E2B")
        self.assertTrue(request["winnow"]["images"][0].startswith("data:image/jpeg;base64,"))
        self.assertEqual(set(request["questions"]), {"presence", "operator"})
        self.assertNotIn("expected", request)
        self.assertNotIn(sample["question_id"], json.dumps(request))

    def test_winnow_pilot_balances_classes_and_thins_neighboring_images(self):
        def sample(identifier, operator, seconds):
            return {"sample_id": identifier, "run_id": "run", "stream_seconds": seconds,
                    "label": {"status": "labeled", "operator": operator}}
        samples = [sample("a", "coaster", 0), sample("b", "coaster", 1),
                   sample("c", "coaster", 31), sample("d", "bnsf", 0)]
        cases = winnow_cases(samples, 3)
        self.assertEqual([s["sample_id"] for s, _ in cases], ["a", "d", "c"])

    def test_winnow_pilot_summarizes_mocked_predictions_without_changing_labels(self):
        first, second = load_samples(self.root)
        save_label(first, CHOICES[7])
        save_label(second, CHOICES[0])
        original_labels = (self.frame_dir / "labels.json").read_text()
        def response(probability, operator):
            return BytesIO(json.dumps({"model": "Winnow-E2B", "answers": {
                "presence": {"probabilities": {"train": probability, "no_train": 1-probability}},
                "operator": {"choice": operator}}}).encode())
        with patch("label.urlopen", side_effect=[response(.1, "no_train"), response(.8, "coaster")]):
            summary = test_winnow(self.root, self.root / "pilot", server="http://localhost:8091")
        self.assertEqual(summary["evaluated_cases"], 2)
        self.assertEqual(summary["operator_correct"], 1)
        self.assertEqual(summary["presence_thresholds"]["0.95"]["real_trains_kept"], 0)
        self.assertEqual(summary["presence_thresholds"]["0.5"]["real_trains_kept"], 1)
        self.assertEqual((self.frame_dir / "labels.json").read_text(), original_labels)

    def gemini_response(self, value=90, status="completed"):
        payload, _ = gemini_frame_request(self.frame_dir / "frame.jpg")
        keys = payload["response_format"]["schema"]["properties"]["percentages"]["required"]
        prediction = {"percentages": {key: value for key in keys}, "explanation": "Two distinct trains are visible."}
        return BytesIO(json.dumps({"model": "gemini-3.5-flash-lite", "status": status,
            "steps": [{"type": "thought", "content": [{"type": "text", "text": "ignore"}]},
                      {"type": "model_output", "content": [{"type": "text", "text": json.dumps(prediction)}]}]}).encode())

    def test_gemini_request_uses_full_frame_and_independent_operator_estimates(self):
        Image.new("RGB", (2400, 1200), "red").save(self.frame_dir / "frame.jpg")
        with patch.dict("os.environ", {}, clear=True):
            payload, metadata = gemini_frame_request(self.frame_dir / "frame.jpg", "Rail camera")
        self.assertEqual(payload["model"], "gemini-3.5-flash-lite")
        self.assertFalse(payload["store"])
        self.assertEqual(metadata["original_size"], [2400, 1200])
        self.assertEqual(metadata["submitted_size"], [1920, 960])
        import base64
        image = payload["input"][2]
        self.assertEqual(image["mime_type"], "image/jpeg")
        self.assertEqual(Image.open(BytesIO(base64.b64decode(image["data"]))).size, (1920, 960))
        self.assertIn("Rail camera", payload["input"][0]["text"])
        schema = payload["response_format"]["schema"]["properties"]["percentages"]
        for key in ("coaster", "pacific_surfliner", "bnsf", "mts_trolley"):
            self.assertIn(key, schema["required"])

    def test_gemini_references_are_labeled_and_separate_from_target(self):
        payload, metadata = gemini_frame_request(self.frame_dir / "frame.jpg")
        parts = payload["input"]
        self.assertIn("TARGET FRAME", parts[1]["text"])
        self.assertEqual(len([part for part in parts if part["type"] == "image"]), 4)
        self.assertEqual([r["operator"] for r in metadata["references"]],
                         ["coaster", "coaster", "pacific_surfliner"])
        self.assertIn("NCTD COASTER", parts[3]["text"])
        self.assertIn("Pacific Surfliner", parts[7]["text"])
        self.assertIn("never count their trains as present", parts[0]["text"])

    def test_gemini_references_cannot_escape_prompt_folder(self):
        prompt_dir = self.root / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "prompt.txt").write_text("Classify the target.")
        (prompt_dir / "references.json").write_text(json.dumps([
            {"operator": "coaster", "file": "../outside.png"}]))
        with self.assertRaisesRegex(ValueError, "inside the Gemini prompt folder"):
            gemini_frame_request(self.frame_dir / "frame.jpg", prompt_dir=prompt_dir)

    def test_gemini_percentages_can_include_two_operators_and_do_not_change_labels(self):
        sample = load_samples(self.root)[0]
        save_label(sample, CHOICES[0])
        previous = (self.frame_dir / "labels.json").read_text()
        token = "fake_test_token_not_real"
        with patch("label.load_project_env"), patch.dict("os.environ", {"GEMINI_API_KEY": token}), patch("label.urlopen", return_value=self.gemini_response()) as send:
            report, output = test_gemini_frame(sample["frame_path"], self.root / "gemini_output")
        self.assertEqual(report["percentages"]["bnsf"], 90)
        self.assertEqual(report["percentages"]["pacific_surfliner"], 90)
        self.assertEqual(report["explanation"], "Two distinct trains are visible.")
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "https://generativelanguage.googleapis.com/v1beta/interactions")
        self.assertEqual(request.get_header("X-goog-api-key"), token)
        self.assertNotIn(token, output.read_text())
        self.assertNotIn('"data"', output.read_text())
        self.assertEqual((self.frame_dir / "labels.json").read_text(), previous)

    def test_gemini_missing_credentials_does_not_make_network_request(self):
        with patch.dict("os.environ", {}, clear=True), patch("label.load_project_env"), patch("label.urlopen") as send:
            with self.assertRaisesRegex(ValueError, "Fill GEMINI_API_KEY"):
                test_gemini_frame(self.frame_dir / "frame.jpg", self.root / "out")
            send.assert_not_called()

    def test_gemini_model_can_be_overridden(self):
        with patch.dict("os.environ", {"GEMINI_MODEL": "gemini-3.7-flash"}):
            payload, _ = gemini_frame_request(self.frame_dir / "frame.jpg")
        self.assertEqual(payload["model"], "gemini-3.7-flash")

    def test_gemini_rejects_invalid_estimates_and_incomplete_responses(self):
        for value, status in [(101, "completed"), (True, "completed"), (float("nan"), "completed"), (90, "failed")]:
            with self.subTest(value=value, status=status), patch("label.load_project_env"), patch.dict("os.environ", {"GEMINI_API_KEY": "test"}), patch("label.urlopen", return_value=self.gemini_response(value, status)):
                with self.assertRaises(ValueError):
                    test_gemini_frame(self.frame_dir / "frame.jpg", self.root / "out")
                self.assertFalse((self.root / "out").exists())

    def test_gemini_quota_error_is_friendly_and_not_retried(self):
        from urllib.error import HTTPError
        error = HTTPError("https://example.com", 429, "limited", {}, None)
        with patch("label.load_project_env"), patch.dict("os.environ", {"GEMINI_API_KEY": "test"}), patch("label.urlopen", side_effect=error) as send:
            with self.assertRaisesRegex(ValueError, "quota reached"):
                test_gemini_frame(self.frame_dir / "frame.jpg", self.root / "out")
            send.assert_called_once()

    def test_gemini_rejects_missing_predictions(self):
        response = BytesIO(json.dumps({"status": "completed", "steps": []}).encode())
        with patch("label.load_project_env"), patch.dict("os.environ", {"GEMINI_API_KEY": "test"}), patch("label.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "malformed"):
                test_gemini_frame(self.frame_dir / "frame.jpg", self.root / "out")

    def test_project_env_loads_quoted_credentials_and_preserves_existing_values(self):
        path = self.root / ".env"
        path.write_text('# credentials\nGEMINI_MODEL="abc" # comment\nexport GEMINI_API_KEY=\'test-token\'\n')
        with patch.dict("os.environ", {}, clear=True):
            load_project_env(path)
            import os
            self.assertEqual(os.environ["GEMINI_MODEL"], "abc")
            self.assertEqual(os.environ["GEMINI_API_KEY"], "test-token")
            os.environ["GEMINI_API_KEY"] = "existing-token"
            load_project_env(path)
            self.assertEqual(os.environ["GEMINI_API_KEY"], "existing-token")


if __name__ == "__main__":
    unittest.main()
