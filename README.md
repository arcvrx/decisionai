# DecisionAI — train screenshot collection

Watch the YouTube rail livestream, save YOLO train detections, and manually
label the operator for each crop.

This is a local data collection app. It detects trains with YOLO, saves full
frames and clean crops, and provides a Tkinter review center with a clickable
timeline and a labeling queue across all runs. Operator classifier training
is planned; operator predictions are not implemented yet.

## Setup

Requires Python 3.11 or newer, uv, FFmpeg, and a Python installation with
Tkinter support. Run commands from the project root. No Cloudflare API key or
other API credentials are needed.

Use uv to manage the environment. Install FFmpeg and Tk/Tcl through your system
package manager; on Arch Linux:

```bash
sudo pacman -S ffmpeg tk
uv sync --locked
```

Start the recorder:

```bash
uv run record.py
```

In another terminal, open the Tkinter labeler:

```bash
uv run label.py
```

The default source is the San Diego rail livestream. To use another live URL:

```bash
uv run record.py 'https://www.youtube.com/watch?v=VIDEO_ID'
```

The first detection run downloads YOLO weights if they are missing. If YouTube
extraction reports JavaScript runtime errors, install Node.js; the collector
uses it when available. The GUI needs a desktop display, but frame collection
can run in a terminal without the GUI.

## Review workflow

The collector runs until Ctrl+C without saving video. YOLO checks one sampled
frame per second by default. New and moving trains get a screenshot on each
examined frame, giving you several views during a train's movement. Unchanged
parked trains are captured initially and then once every 60 seconds.
The labeler displays each detection in a green box alongside its clean crop.
Click an operator button or press its number:

1. NCTD COASTER
2. Pacific Surfliner
3. Metrolink
4. MTS Trolley
5. BNSF freight
6. Other operator
7. Unclear / cannot identify
8. False detection
9. Skip

Each train in a screenshot is labeled separately. Your choice saves immediately
in `labels.json`; it is never applied automatically to neighboring frames.
Each crop has a stable question ID, including the recording, frame, and detection
number. Left/right buttons or arrow keys browse saved questions, including ones
you already labeled; you can change a label. Selecting a choice advances to the
next unlabeled question. Use `--include-skipped` to revisit
skipped items. Unclear and false-detection choices are kept separate from
confirmed operator labels.

The bottom timeline scrolls horizontally with its scrollbar or mouse wheel.
Click a peak or screenshot dot to open that frame. Green dots mean its saved
questions have been reviewed. Select a recording from the top menu to browse
older history; **Live / latest** follows incoming screenshots. Q or Escape quits.
The operator panel counts answered, unanswered, and skipped questions for the
selected view, plus unanswered/skipped totals across all recordings.
Enable **Show unanswered / skipped only** to find missed questions; skipped
items remain pending and their timeline dots do not turn green.
The labeler starts in **All recordings** with the pending filter enabled.
Answering a question or pressing **Next unlabeled** searches across every
recording, wrapping back to earlier missed questions. The timeline displays
the selected question's recording; its heading names that recording. Select an
individual recording from the View menu for focused browsing.

Click **Start image server** to share the `recordings/` folder using Python's
built-in HTTP server on port 8000. On this computer, open
`http://localhost:8000/`; on another computer on the same network, use
`http://YOUR-LAN-IP:8000/` (find your IP with `hostname -I`). Browse to
`RUN/frames/frame_NUMBER/frame.jpg` for a full image, or download PNG crops and
JSON results for classifier experiments. All files under `recordings/` are
served, including labels and any old videos. The server has no login. Click
**Stop image server** or close the GUI to stop sharing. A port conflict shows
an error without interrupting labeling.

**Activity** shows tall peaks for new/moving detections, a low gray baseline for
stationary trains, and a flat line for sampled frames with no train. This keeps
the parked Surfliner from filling the chart with tall peaks. **Presence** shows
all train detections at their YOLO confidence. These are detection/activity
scores, not operator predictions. Motion uses box movement and pixel changes;
lighting changes or detection jitter may still produce extra captures. Small
cars peeking into the frame may be missed by YOLO. No train is automatically
identified as Surfliner, and neighboring crops never share labels.

## Commands

```bash
uv run record.py --minutes 2
uv run record.py --sample-fps 2 --conf 0.4
uv run record.py --stationary-seconds 120 --motion-threshold 8
uv run record.py --stall-seconds 45
uv run label.py --recordings recordings --include-skipped
uv run record.py --snapshot
```

`record.py --snapshot` saves a clean livestream frame under `snapshots/`.
Both scripts support `--help`; the recorder accepts an optional YouTube URL.

## Saved data

Each run gets a directory under `recordings/`, containing:

- `recording.json`: collection settings/status (including `save_video: false`).
- `detections.jsonl`: positive frames and approximate stream offsets.
- `timeline.jsonl`: every examined sample, including empty and stationary frames,
  activity/presence scores, and saved question IDs.
- `frames/frame_NUMBER/frame.jpg`: clean full-frame screenshot.
- `frames/frame_NUMBER/frame_detected.jpg`: boxes and detection scores.
- `frames/frame_NUMBER/frame_train_N.png`: original-resolution lossless crop.
- `frames/frame_NUMBER/results.json`: detection coordinates and paths.
- `frames/frame_NUMBER/labels.json`: your manual operator labels.

Existing recordings and labels remain readable, including older JPEG crops.
Older recordings show saved positive detections only; their empty intervals
were not logged. Timeline times are relative to the recording start.
Frames without detections are not saved separately. If YOLO falls behind,
pending samples are skipped so scanning stays current. Only screenshots, crops,
labels, and metadata are saved; there is no video archive in new runs.
Video files from older runs remain on disk. If FFmpeg ends or decoded frames
stop arriving for 45 seconds, the collector refreshes the YouTube playback URL
and reconnects automatically. Failed retries wait 5, 10, 20, 40, then 60 seconds
until frames resume or you press Ctrl+C. TLS warnings alone do not restart it.
For low sample rates the stall timeout is at least three sample intervals.
Recovery keeps the same run and unique question IDs; downtime appears as gaps
in the timeline. `--minutes` includes reconnect downtime. Initial startup URL
resolution failures still exit with an error. Local detection or disk errors
also stop collection rather than retrying the stream.
The first detection run may download model weights.

Operator classifier training and dataset splitting are not implemented yet.
Keep neighboring frames of the same train together when building training/test
sets to avoid inflated evaluation accuracy.

## Code organization

Application code is kept in four root files:

| File | Responsibility |
| --- | --- |
| `record.py` | Stream access, FFmpeg frame sampling, and screenshots |
| `detect.py` | YOLO detection, motion/capture cadence, crops, and box previews |
| `label.py` | Saved questions, timeline reading, labels, and labeler command |
| `gui.py` | Tkinter history chart, screenshot navigation, and label buttons |

There is no installable package or wrapper layer. `pyproject.toml` and `uv.lock`
manage dependencies for local execution.

## Development

Screenshots, recordings, model weights, local credentials, and environment
files are excluded by `.gitignore`. Share the code and `uv.lock`; collect or
copy your dataset separately. Existing local data is not deleted.

Run tests with:

```bash
uv run python -m unittest discover -s tests
```

The tests use synthetic images and mocked detectors/reconnects, so they do not
require a livestream, downloaded model weights, or a desktop display.
