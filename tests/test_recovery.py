import queue
import threading
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from record import stream_frames


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.info = {"url": "old-playback-url", "width": 10, "height": 10}
        self.image = Image.new("RGB", (10, 10))
        self.processes = []

    def sleep(self, seconds):
        self.now += seconds

    def process(self, *args, **kwargs):
        process = Mock()
        process.poll.return_value = None
        self.processes.append(process)
        return process

    def run_threads(self, **kwargs):
        worker = Mock()
        worker.start.side_effect = lambda: self.reader(*kwargs["args"])
        return worker

    def mocks(self, resolver):
        return (
            patch("record.time.monotonic", side_effect=lambda: self.now),
            patch("record.time.sleep", side_effect=self.sleep),
            patch("record.subprocess.Popen", side_effect=self.process),
            patch("record.threading.Thread", side_effect=self.run_threads),
            patch("record.resolve_stream", side_effect=resolver),
        )

    def test_eof_and_failed_url_refresh_recover_with_unique_indices(self):
        connections = 0

        def reader(stream, dimensions, samples, finished, errors, progress):
            nonlocal connections
            connections += 1
            samples.put((0, self.image))
            finished.set()  # First connection ends after its initial frame.

        self.reader = reader
        refreshed = {**self.info, "url": "fresh-playback-url"}
        patches = self.mocks([RuntimeError("Temporary YouTube error"), refreshed])
        recovery = {}
        with patches[0], patches[1] as sleep, patches[2] as popen, patches[3], patches[4] as resolver:
            frames = stream_frames("youtube-url", self.info, 1, recovery=recovery)
            first, _ = next(frames)
            second, _ = next(frames)
            frames.close()
            self.assertEqual((first, second), (0, 15))
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 10])
            self.assertEqual(resolver.call_count, 2)
            self.assertIn("fresh-playback-url", popen.call_args.args[0])
            self.assertEqual(recovery["reconnect_count"], 2)
            for process in self.processes:
                process.terminate.assert_called_once()
                process.stdout.close.assert_called_once()

    def test_stalled_live_process_is_terminated_then_reconnected(self):
        connections = 0

        def reader(stream, dimensions, samples, finished, errors, progress):
            nonlocal connections
            connections += 1
            if connections == 1:
                progress["last_frame"] = -46
            else:
                samples.put((0, self.image))

        self.reader = reader
        patches = self.mocks([self.info])
        with patches[0], patches[1], patches[2], patches[3], patches[4] as resolver:
            frames = stream_frames("youtube-url", self.info, 1)
            index, _ = next(frames)
            frames.close()
            self.assertEqual(index, 5)
            self.processes[0].terminate.assert_called_once()
            resolver.assert_called_once()

    def test_healthy_frame_delivery_does_not_refresh_url(self):
        def reader(stream, dimensions, samples, finished, errors, progress):
            samples.put((0, self.image))

        self.reader = reader
        patches = self.mocks([])
        with patches[0], patches[1] as sleep, patches[2], patches[3], patches[4] as resolver:
            frames = stream_frames("youtube-url", self.info, 1)
            next(frames)
            frames.close()
            sleep.assert_not_called()
            resolver.assert_not_called()

    def test_duration_expiring_during_retry_stops_without_new_connection(self):
        self.reader = lambda stream, dimensions, samples, finished, errors, progress: finished.set()
        patches = self.mocks([])
        with patches[0], patches[1] as sleep, patches[2], patches[3], patches[4] as resolver:
            self.assertEqual(list(stream_frames("youtube-url", self.info, 1, seconds=2)), [])
            sleep.assert_called_once_with(2)
            resolver.assert_not_called()
