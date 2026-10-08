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

### Correct train boxes

Select a question, then click **Edit / add / delete train boxes**. The editor
shows every saved box in the full frame. Click a box to select it, drag a green
corner to resize, or drag inside to move it. **New box** lets you drag a separate
rectangle around a missing train; draw one tight box per visible train, including
partially visible trains. **Delete selected box** removes false detections.
Click **Save corrections**, or **Cancel** to discard the edits.

New boxes get new question IDs. Changed crops become unanswered, so label them
again; unchanged boxes keep their labels and original crops. Corrections do not
carry over to neighboring frames. Corrected crops are made from the saved full
JPEG frame and have no extra padding. Original detector results and crop files
remain intact, and invalidated labels are retained in the annotation history.
After saving, the pending filter is turned off so you can see all boxes in that
frame; answering resumes the global review queue.

Only check **I checked every visible train in this frame** after adding every
missing train and removing all false boxes. This marks the frame's box annotation
complete for future detector training. Deleting every box and checking this
option records a reviewed negative frame. Saving edits without the check leaves
the frame incomplete. This annotation collection does not yet retrain YOLO.

### Reuse a box across frames

Correct a box once, then click **Save box…** under **Reusable train boxes** and
give it a name, such as `left freight` or `parked Surfliner`. Select it from the
preset dropdown and click **Apply** to replace the current question's box, or
**Add train** to create a separate question for a missing train in that frame.
Other boxes in the frame are kept. Presets are saved in `recordings/box_presets.json`
and use relative image coordinates so they also work at another resolution.

For repeated labeling, enable **Apply preset when labeling (1–6)**: clicking
an operator applies the selected rectangle to the current question and labels
the resulting crop in one action, then advances through the queue. Unclear,
false-detection, and skip answers do not apply the preset. Switch off the checkbox
or select another preset when the train moves outside that fixed area. Presets
do not track motion or automatically mark whole frames reviewed for detector
training. Each operator choice still labels only the selected question.

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
- `frames/frame_NUMBER/annotations.json`: manual boxes, review status, and
  invalidated label history; overrides `results.json` in the review queue.
- `frames/frame_NUMBER/frame_corrected.jpg`: preview of all corrected boxes.
- `frames/frame_NUMBER/frame_train_N_corrected_TIMESTAMP.png`: corrected crop.

For edited frames, use `annotations.json` as the current boxes. `results.json`
and `detections.jsonl` preserve the original detector output. Timeline question
links follow corrected boxes, while its scores remain the original detection
scores. Deleted questions are removed from the queue, not silently relabeled.

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

## Winnow-E2B pilot

To prepare a small test of labeled crops without launching the GUI:

```bash
uv run label.py --winnow-test --winnow-limit 60
```

To evaluate them against an already running Winnow-E2B server with vision:

```bash
uv run label.py --winnow-test --winnow-url http://localhost:8091 --winnow-limit 60
```

Use the GPU host's address instead of `localhost` for a remote server. Winnow is
a separate runtime; this project does not install its model or server. Follow
the [Winnow E2B setup](https://github.com/EldanRing/winnow-inference/blob/main/docs/E2B.md)
and matching vision/projector instructions. This pilot calls `/v1/systemone`
with native direct scoring and checks that the server reports E2B. It does not
reproduce adaptive reasoning settings your friend may have used.

Output is saved under `outputs/winnow_TIMESTAMP/`: `requests.jsonl` contains
case IDs, reference answers, and image requests; only the nested `request` is
sent to the model. `responses.jsonl` stores predictions and `summary.json`
reports train-presence results at several thresholds and operator confusion.
The human labels are never changed. The pilot balances available classes and
keeps at most one sample per class/run/30-second interval. This reduces repeated
frames but is not a held-out accuracy benchmark. Skipped and unclear labels
are excluded. Model scores are not guaranteed probabilities of correctness.

## Gemini full-frame test

The labeler automatically loads the project’s Git-ignored `.env` file.
Get a key from [Google AI Studio](https://aistudio.google.com/apikey) and fill in:

```dotenv
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-3.5-flash-lite
```

No environment-file argument or new Python dependency is needed. `GEMINI_MODEL`
is optional; existing shell environment settings take precedence over `.env`.
The supported Flash free tier has quotas; model availability and limits depend
on your project. Check [Google’s pricing](https://ai.google.dev/gemini-api/docs/pricing).

Test the latest saved full frame or choose an image:

```bash
uv run label.py --gemini-test
uv run label.py --gemini-frame recordings/RUN/frames/frame_NUMBER/frame.jpg
uv run label.py --gemini-frame frame.png --gemini-context camera_description.txt
```

In `uv run label.py`, click **Test full frame: Gemini**. The request runs in the
background and displays independent operator presence estimates and a short
explanation. It analyzes the clean full frame, including trains outside the
selected box, and saves `outputs/gemini_TIMESTAMP/result.json`. It never changes
human labels or images. Cloudflare settings and Clef commands are no longer used.

Multiple operators can score high simultaneously. Additional estimates cover
any train, multiple trains, partial railcars at the bottom, and unclear operators.
These percentages are model-generated estimates, not calibrated probabilities;
they do not sum to 100%. One frame cannot establish whether a train is stationary.
The editable prompt lives in `prompts/gemini/prompt.txt`. Labeled examples are
listed in `prompts/gemini/references.json`, with their images in
`prompts/gemini/references/`. Both GUI and CLI tests load these automatically,
independent of the terminal's working directory. The initial examples are a
daytime COASTER frame, a nighttime COASTER crop, and a nighttime Surfliner crop,
confirmed by the user. Reference trains are explicitly excluded from target
counts. References are sent with each request, not permanent model training.
Results record the reference filenames, labels, descriptions and image sizes.
Keep reference examples separate from images used to evaluate accuracy; testing
the same image used as a reference is not an independent evaluation. Add or
replace examples by editing the manifest; keep paths inside the prompt folder.
The app allows at most 12 examples and caps the encoded request at 18 MiB.
Images are compressed as JPEG with a maximum side of 1920 pixels. Requests use
Google’s [Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview)
and [structured JSON output](https://ai.google.dev/gemini-api/docs/structured-output),
with `store=false`. Credentials and encoded images are not saved in reports.
Free-tier submissions may still be used to improve Google’s products under its
API terms. The default prompt includes the user's San Diego Depot observations:
long BNSF freight on the leftmost track (also sometimes used by Surfliner), parked
Surfliner on the second track from the left, and usually blue COASTER trains
farther right. These are contextual clues, not guaranteed identifications; a
single image cannot establish whether a train is stationary. Optional scene
context adds to those defaults. This is a manual experiment, not automatic
classification during recording.

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
