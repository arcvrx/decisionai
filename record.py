"""Watch the rail livestream and collect YOLO train screenshots."""

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import math
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from PIL import Image

from detect import CaptureSampler, detect_trains, export_detections


DEFAULT_URL = "https://www.youtube.com/watch?v=h5qA5PvCjJk"


def resolve_stream(url):
    from yt_dlp import YoutubeDL

    options = {
        "format": (
            "bestvideo[height<=1080][vcodec^=avc1]/"
            "bestvideo[height<=1080]/best[height<=1080]"
        ),
        "noplaylist": True,
        "socket_timeout": 15,
        "retries": 2,
        "extractor_retries": 2,
        "quiet": True,
        "live_from_start": False,
    }
    if shutil.which("node"):
        options["js_runtimes"] = {"node": {}}
    with YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)
    if not info or not info.get("url"):
        raise ValueError("YouTube did not provide a playable stream URL.")
    if not info.get("is_live"):
        raise ValueError("This URL is not currently a live stream.")
    return info


def capture_live_frame(url, output_dir):
    """Grab a clean video frame without browser controls or screen scaling."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required to capture a live frame.")
    info = resolve_stream(url)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    headers = info.get("http_headers", {})
    if headers:
        command += ["-headers", "".join(f"{name}: {value}\r\n" for name, value in headers.items())]
    command += [
        "-rw_timeout", "15000000", "-i", info["url"],
        "-map", "0:v:0", "-an", "-frames:v", "1",
        "-c:v", "png", "-threads", "1", "-f", "image2pipe", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError(f"FFmpeg capture failed: {result.stderr.decode(errors='replace')[:2000]}")
    # Validate the decoded output before persisting a screenshot.
    with Image.open(BytesIO(result.stdout)) as frame:
        frame.load()
        if frame.format != "PNG":
            raise ValueError("Capture did not return a PNG image.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with tempfile.NamedTemporaryFile(prefix=f"live_{timestamp}_", suffix=".png", dir=output_dir, delete=False) as output:
        output.write(result.stdout)
        path = Path(output.name)
    print(f"Saved live screenshot: {path}", flush=True)
    return path


def sampling_command(source, headers, seconds, sample_fps, dimensions):
    """Decode sampled frames to memory without writing video files."""
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin"]
    if headers:
        header_text = "".join(f"{name}: {value}\r\n" for name, value in headers.items())
        command += ["-headers", header_text]
    width, height = dimensions
    command += [
        "-rw_timeout", "15000000", "-i", source,
        "-map", "0:v:0", "-an", "-vf", f"fps={sample_fps},scale={width}:{height}",
        "-pix_fmt", "rgb24", "-c:v", "rawvideo", "-threads", "1",
    ]
    if seconds is not None:
        command += ["-t", str(seconds)]
    command += ["-f", "rawvideo", "pipe:1"]
    return command


def read_samples(stream, dimensions, samples, finished, errors, progress=None):
    """Drain FFmpeg promptly; keep the newest sample if YOLO falls behind."""
    frame_bytes = dimensions[0] * dimensions[1] * 3
    index = 0
    try:
        while True:
            data = bytearray()
            while len(data) < frame_bytes:
                chunk = stream.read(frame_bytes - len(data))
                if not chunk:
                    break
                data.extend(chunk)
            if not data:
                break
            if len(data) != frame_bytes:
                raise ValueError("FFmpeg returned an incomplete sampled frame.")
            if progress is not None:
                progress["last_frame"] = time.monotonic()
            image = Image.frombytes("RGB", dimensions, bytes(data))
            try:
                samples.put_nowait((index, image))
            except queue.Full:
                try:
                    samples.get_nowait()
                except queue.Empty:
                    pass
                samples.put_nowait((index, image))
            index += 1
    except Exception as exc:
        errors.append(str(exc))
    finally:
        finished.set()


def stream_frames(url, info, sample_fps, seconds=None, stall_seconds=45, recovery=None):
    """Yield frames across reconnects; stderr warnings alone never trigger one."""
    started = time.monotonic()
    deadline = started + seconds if seconds is not None else None
    last_index = -1
    failures = 0
    first_connection = True
    while deadline is None or time.monotonic() < deadline:
        process = None
        reader = None
        reason = None
        try:
            if not first_connection:
                info = resolve_stream(url)  # Signed playback URLs can expire.
            first_connection = False
            dimensions = (info.get("width"), info.get("height"))
            if not all(isinstance(value, int) and value > 0 for value in dimensions):
                raise ValueError("Stream resolution is missing.")
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return
            command = sampling_command(info["url"], info.get("http_headers", {}), remaining, sample_fps, dimensions)
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
            samples = queue.Queue(maxsize=1)
            finished = threading.Event()
            errors = []
            progress = {"last_frame": time.monotonic()}
            reader = threading.Thread(target=read_samples,
                args=(process.stdout, dimensions, samples, finished, errors, progress), daemon=True)
            reader.start()
            resumed = False
            while deadline is None or time.monotonic() < deadline:
                try:
                    _, image = samples.get(timeout=0.5)
                except queue.Empty:
                    if finished.is_set():
                        reason = "; ".join(errors) or "FFmpeg stream ended"
                        break
                    # Watch the decoder thread, not the detection/label results.
                    timeout = max(stall_seconds, 3 / sample_fps)
                    if time.monotonic() - progress["last_frame"] >= timeout:
                        reason = f"No frames received for {timeout:g} seconds"
                        break
                    continue
                if not resumed:
                    if failures:
                        print("Stream recovered; scanning resumed.", flush=True)
                    failures = 0
                    resumed = True
                # Never overwrite earlier screenshots; outages remain gaps.
                index = max(last_index + 1, int((time.monotonic() - started) * sample_fps))
                last_index = index
                yield index, image
        except (OSError, ValueError, RuntimeError) as exc:
            reason = str(exc)
        except Exception as exc:
            # yt-dlp raises its own exception types for resolution failures.
            reason = str(exc)
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                if reader is not None:
                    reader.join(timeout=5)
                process.stdout.close()
        if deadline is not None and time.monotonic() >= deadline:
            return
        failures += 1
        delay = min(60, 5 * 2 ** min(failures - 1, 4))
        if recovery is not None:
            recovery["reconnect_count"] = recovery.get("reconnect_count", 0) + 1
            recovery["last_recovery_reason"] = reason
        if deadline is not None:
            delay = min(delay, max(0, deadline - time.monotonic()))
        print(f"Stream interrupted: {reason}. Retrying in {delay:g}s (Ctrl+C to stop).", flush=True)
        time.sleep(delay)


def save_detected_frame(image, model, run_dir, index, sample_fps, confidence, sampler=None):
    frame_dir = run_dir / "frames" / f"frame_{index:08d}"
    seconds = index / sample_fps
    all_detections = detect_trains(image, model, confidence)
    if sampler:
        sampler.update(image, all_detections, seconds)
    selected = [d for d in all_detections if d.get("save_sample", True)]
    for detection in selected:
        detection["question_id"] = f"{run_dir.name}/{frame_dir.name}/train_{detection['train_id']}"
    tick = {
        "sample_index": index, "stream_seconds": seconds,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_count": len(all_detections),
        "active_count": sum(d.get("activity") != "stationary" for d in all_detections),
        "presence_score": max((d["detection_confidence"] for d in all_detections), default=0),
        "activity_score": max((d["detection_confidence"] for d in all_detections if d.get("activity") != "stationary"), default=0),
        "question_ids": [d["question_id"] for d in selected],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    # Publish screenshot references after their files are complete.
    if not selected:
        with (run_dir / "timeline.jsonl").open("a") as timeline:
            timeline.write(json.dumps(tick) + "\n")
        return None
    preview, detections = export_detections(image, selected, frame_dir)
    frame_path = frame_dir / "frame.jpg"
    image.save(frame_path, quality=95)
    report = {
        "sample_index": index,
        "approx_stream_seconds": index / sample_fps,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "frame_path": str(frame_path),
        "preview_path": str(preview),
        "detections": detections,
    }
    (frame_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    with (run_dir / "detections.jsonl").open("a") as manifest:
        manifest.write(json.dumps(report) + "\n")
    with (run_dir / "timeline.jsonl").open("a") as timeline:
        timeline.write(json.dumps(tick) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", default=DEFAULT_URL)
    parser.add_argument("--minutes", type=float, default=0, help="Collection duration; 0 runs until Ctrl+C (default)")
    parser.add_argument("--output-dir", type=Path, default=Path("recordings"))
    parser.add_argument("--sample-fps", type=float, default=1, help="Frames per second sampled for YOLO")
    parser.add_argument("--stall-seconds", type=float, default=45, help="Reconnect if no frames arrive for this long")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO detection weights")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--stationary-seconds", type=float, default=60, help="Capture unchanged parked trains this often")
    parser.add_argument("--motion-threshold", type=float, default=6, help="Pixel-change threshold for train motion")
    parser.add_argument("--snapshot", action="store_true", help="Take one livestream screenshot and exit")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("snapshots"))
    args = parser.parse_args()
    if args.snapshot:
        try:
            capture_live_frame(args.url, args.snapshot_dir)
        except Exception as exc:
            parser.exit(1, f"Could not capture the livestream: {exc}\n")
        return
    if not math.isfinite(args.minutes) or args.minutes < 0:
        parser.error("Minutes must be finite and nonnegative.")
    if not math.isfinite(args.sample_fps) or not 0 < args.sample_fps <= 30 or not 0 < args.conf <= 1:
        parser.error("Sample FPS must be in (0, 30] and confidence in (0, 1].")
    if not math.isfinite(args.stationary_seconds) or args.stationary_seconds <= 0 or not math.isfinite(args.motion_threshold) or args.motion_threshold <= 0:
        parser.error("Stationary interval and motion threshold must be positive and finite.")
    if not math.isfinite(args.stall_seconds) or args.stall_seconds <= 0:
        parser.error("Stall timeout must be positive and finite.")
    try:
        from ultralytics import YOLO
        model = YOLO(args.model)
    except (ImportError, OSError, RuntimeError) as exc:
        parser.error(f"Could not load YOLO; run uv sync and check model weights: {exc}")
    if not shutil.which("ffmpeg"):
        parser.error("FFmpeg is required. Install it with your system package manager.")
    try:
        info = resolve_stream(args.url)
    except ImportError:
        parser.error("yt-dlp is not installed. Run uv sync.")
    except Exception as exc:
        parser.exit(1, f"Could not resolve livestream: {exc}\n")

    dimensions = (info.get("width"), info.get("height"))
    if not all(isinstance(value, int) and value > 0 for value in dimensions):
        parser.error("Stream resolution is missing; cannot sample frames safely.")
    seconds = args.minutes * 60 if args.minutes else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(tempfile.mkdtemp(prefix=f"{timestamp}_", dir=args.output_dir))
    sampler = CaptureSampler(args.stationary_seconds, args.motion_threshold)
    metadata = {
        "source_url": args.url,
        "video_id": info.get("id"),
        "title": info.get("title"),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_seconds": seconds,
        "sample_fps": args.sample_fps,
        "yolo_model": args.model,
        "yolo_confidence": args.conf,
        "stationary_seconds": args.stationary_seconds,
        "motion_threshold": args.motion_threshold,
        "save_video": False,
        "stall_seconds": args.stall_seconds,
        "format_id": info.get("format_id"),
        "width": info.get("width"),
        "height": info.get("height"),
        "status": "recording",
    }
    metadata_path = run_dir / "recording.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Collecting screenshots from {info.get('title', args.url)} into {run_dir}", flush=True)
    print("Press Ctrl+C to stop. Screenshots are saved; video is not stored.", flush=True)
    recovery = {"reconnect_count": 0}
    frames = stream_frames(args.url, info, args.sample_fps, seconds, args.stall_seconds, recovery)
    interrupted = False
    detection_error = None
    checked_frames = 0
    detected_frames = 0
    waiting = True
    print("Waiting for new or moving trains; scanning continues.", flush=True)
    try:
        for index, image in frames:
            report = save_detected_frame(image, model, run_dir, index, args.sample_fps, args.conf, sampler)
            checked_frames += 1
            if report:
                detected_frames += 1
                waiting = False
                print(f"Train detected: saved {len(report['detections'])} crop(s) — {report['frame_path']}", flush=True)
            elif not waiting:
                waiting = True
                print("Waiting for new or moving trains; scanning continues.", flush=True)
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:
        detection_error = str(exc)
    finally:
        frames.close()
    metadata.update({
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed" if detection_error else "stopped" if interrupted else "complete" if checked_frames > 0 else "failed",
        "checked_frames": checked_frames,
        "detected_frames": detected_frames,
        "detection_error": detection_error,
        **recovery,
    })
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {detected_frames} detected frame(s) in {run_dir}")
    if metadata["status"] == "failed":
        parser.exit(1, f"Collection failed: {detection_error or 'inspect FFmpeg errors above'}.\n")


if __name__ == "__main__":
    main()
