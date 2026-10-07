"""Tkinter screenshot labeling with clickable, scrollable detection history."""

from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time

from label import CHOICES, TimelineReader, load_samples, read_labels, save_label

ALL_RECORDINGS = "All recordings"

def review_counts(samples):
    unanswered = sum(not s.get("label") for s in samples)
    skipped = sum(bool(s.get("label")) and s["label"]["status"] == "skipped" for s in samples)
    return len(samples) - unanswered - skipped, unanswered, skipped


class LabelReviewer:
    def __init__(self, root, recordings, include_skipped=False):
        import tkinter as tk
        from tkinter import messagebox, ttk
        from PIL import Image, ImageDraw, ImageTk

        self.root = root
        self.recordings = Path(recordings)
        self.include_skipped = include_skipped
        self.messagebox = messagebox
        self.Image, self.ImageDraw, self.ImageTk = Image, ImageDraw, ImageTk
        self.samples = []
        self.visible = []
        self.points = []
        self.point_times = []
        self.capture_points = []
        self.sample = None
        self.images, self.photos = {}, {}
        self.label_overrides = {}
        self.poll_id = None
        self.closed = False
        self.labeled_count = 0
        self.reader = TimelineReader()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.next_scan = 0
        self.pixels_per_second = 3
        self.run_id = tk.StringVar(value=ALL_RECORDINGS)
        self.timeline_run = ""
        self.chart_mode = tk.StringVar(value="Activity")
        self.follow_live = tk.BooleanVar(value=False)
        self.unanswered_only = tk.BooleanVar(value=True)
        self.review_summary = tk.StringVar()
        self.image_server = None
        self.server_thread = None

        root.title("Train review center")
        root.geometry("1380x950")
        root.minsize(900, 700)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        ttk.Style(root).configure("Choice.TButton", padding=(10, 9), font=("TkDefaultFont", 11), anchor="w")
        body = ttk.Frame(root, padding=12)
        body.grid(sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(body)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(toolbar, text="View:").pack(side="left")
        self.run_box = ttk.Combobox(toolbar, textvariable=self.run_id, state="readonly", width=44)
        self.run_box.pack(side="left", padx=8)
        self.run_box.bind("<<ComboboxSelected>>", self.change_run)
        ttk.Label(toolbar, text="Timeline:").pack(side="left", padx=(16, 0))
        mode = ttk.Combobox(toolbar, textvariable=self.chart_mode, values=["Activity", "Presence"], state="readonly", width=10)
        mode.pack(side="left", padx=8)
        mode.bind("<<ComboboxSelected>>", lambda event: self.draw_timeline())
        ttk.Button(toolbar, text="Live / latest", command=self.go_live).pack(side="right")
        self.server_button = ttk.Button(toolbar, text="Start image server", command=self.toggle_image_server)
        self.server_button.pack(side="right", padx=8)

        panes = ttk.Panedwindow(body, orient="horizontal")
        panes.grid(row=1, column=0, sticky="nsew")
        image_panel = ttk.Frame(panes)
        image_panel.columnconfigure(0, weight=1)
        image_panel.rowconfigure(2, weight=3)
        image_panel.rowconfigure(4, weight=2)
        choices = ttk.Frame(panes, padding=(16, 0, 0, 0))
        panes.add(image_panel, weight=4)
        panes.add(choices, weight=1)

        navigation = ttk.Frame(image_panel)
        navigation.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.previous_button = ttk.Button(navigation, text="\u2190 Previous", command=lambda: self.navigate(-1))
        self.previous_button.pack(side="left")
        self.next_button = ttk.Button(navigation, text="Next \u2192", command=lambda: self.navigate(1))
        self.next_button.pack(side="left", padx=8)
        ttk.Button(navigation, text="Next unlabeled", command=self.next_unlabeled).pack(side="left")
        self.position = tk.StringVar(value="No screenshots yet")
        ttk.Label(navigation, textvariable=self.position).pack(side="right")
        self.instruction = tk.StringVar(value="Waiting for train detections...")
        ttk.Label(image_panel, textvariable=self.instruction, padding=(0, 0, 0, 8)).grid(row=1, column=0, sticky="w")
        self.frame_canvas = tk.Canvas(image_panel, background="#191919", highlightthickness=0)
        self.frame_canvas.grid(row=2, column=0, sticky="nsew")
        ttk.Label(image_panel, text="Clean train crop", padding=(0, 6)).grid(row=3, column=0, sticky="w")
        self.crop_canvas = tk.Canvas(image_panel, background="#191919", highlightthickness=0)
        self.crop_canvas.grid(row=4, column=0, sticky="nsew")
        for canvas in (self.frame_canvas, self.crop_canvas):
            canvas.bind("<Configure>", lambda event, canvas=canvas: self.draw_image(canvas))

        ttk.Label(choices, text="Choose the operator", font=("TkDefaultFont", 14, "bold")).pack(anchor="w", pady=(0, 8))
        self.current_label = tk.StringVar(value="Unlabeled")
        ttk.Label(choices, textvariable=self.current_label, wraplength=280).pack(anchor="w", pady=(0, 8))
        ttk.Label(choices, textvariable=self.review_summary, wraplength=280).pack(anchor="w", pady=(0, 8))
        ttk.Checkbutton(choices, text="Show unanswered / skipped only", variable=self.unanswered_only,
                        command=self.filter_questions).pack(anchor="w", pady=(0, 10))
        self.buttons = []
        for choice in CHOICES:
            button = ttk.Button(choices, text=f"{choice[0]}. {choice[1]}", style="Choice.TButton", command=lambda choice=choice: self.choose(choice))
            button.pack(fill="x", pady=2)
            self.buttons.append(button)
            root.bind(choice[0], lambda event, choice=choice: self.choose(choice))
        ttk.Label(choices, text="1–9: label this train only\nLeft / Right: browse screenshots\nLabels can be changed later\nQ or Escape: quit", padding=(0, 12), justify="left").pack(anchor="w")

        history = ttk.LabelFrame(body, text="Detection history — click a peak or screenshot marker", padding=8)
        self.history = history
        history.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        history.columnconfigure(0, weight=1)
        self.timeline = tk.Canvas(history, height=145, background="#191919", highlightthickness=0)
        self.timeline.grid(row=0, column=0, sticky="ew")
        scrollbar = ttk.Scrollbar(history, orient="horizontal", command=self.scroll_history)
        scrollbar.grid(row=1, column=0, sticky="ew")
        self.timeline.configure(xscrollcommand=scrollbar.set)
        self.timeline.bind("<Button-1>", self.click_timeline)
        self.timeline.bind("<Configure>", lambda event: self.draw_timeline())
        self.timeline.bind("<MouseWheel>", self.wheel_history)
        self.timeline.bind("<Button-4>", lambda event: self.scroll_history("scroll", -3, "units"))
        self.timeline.bind("<Button-5>", lambda event: self.scroll_history("scroll", 3, "units"))
        ttk.Label(history, text="Orange: new/moving  •  Gray: stationary  •  Green dots: reviewed  •  Gaps: not sampled").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.status = tk.StringVar(value="Run uv run record.py in another terminal.")
        ttk.Label(body, textvariable=self.status, padding=(0, 8, 0, 0), wraplength=1250).grid(row=3, column=0, sticky="w")
        root.bind("<Left>", lambda event: self.navigate(-1))
        root.bind("<Right>", lambda event: self.navigate(1))
        root.bind("q", lambda event: self.close())
        root.bind("Q", lambda event: self.close())
        root.bind("<Escape>", lambda event: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh_details()
        self.poll()

    def draw_image(self, canvas):
        canvas.delete("all")
        image = self.images.get(canvas)
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        if image is None:
            canvas.create_text(width / 2, height / 2, text="Waiting for a detection", fill="white")
            return
        resized = image.copy()
        resized.thumbnail((width, height), self.Image.Resampling.LANCZOS)
        self.photos[canvas] = self.ImageTk.PhotoImage(resized, master=self.root)
        canvas.create_image(width / 2, height / 2, image=self.photos[canvas], anchor="center")

    def select(self, sample, manual=True):
        try:
            with self.Image.open(sample["frame_path"]) as source:
                frame = source.convert("RGB")
            with self.Image.open(sample["crop_path"]) as source:
                crop = source.convert("RGB")
        except (OSError, ValueError) as exc:
            self.messagebox.showerror("Could not open screenshot", str(exc), parent=self.root)
            return
        self.sample = sample
        if self.run_id.get() == ALL_RECORDINGS and self.timeline_run != sample["run_id"]:
            self.timeline_run = sample["run_id"]
            self.points, self.point_times, self.capture_points = [], [], []
            self.next_scan = 0
        self.history.configure(text=f"Detection history: {sample['run_id']} — click a peak or screenshot marker")
        if manual:
            self.follow_live.set(False)
        self.ImageDraw.Draw(frame).rectangle(tuple(sample["bbox"]), outline="lime", width=4)
        self.images = {self.frame_canvas: frame, self.crop_canvas: crop}
        self.instruction.set(f"Question {sample['question_id']} — label the green box")
        self.refresh_details()
        self.draw_image(self.frame_canvas)
        self.draw_image(self.crop_canvas)
        self.draw_timeline()

    def refresh_details(self):
        run_samples = [s for s in self.samples if self.run_id.get() == ALL_RECORDINGS or s["run_id"] == self.run_id.get()]
        answered, unanswered, skipped = review_counts(run_samples)
        _, total_unanswered, total_skipped = review_counts(self.samples)
        self.review_summary.set(
            f"Current view: {answered} answered · {unanswered} unanswered · {skipped} skipped\n"
            f"All recordings: {total_unanswered} unanswered · {total_skipped} skipped")
        index = next((i for i, s in enumerate(self.visible) if self.sample and s["sample_id"] == self.sample["sample_id"]), -1)
        self.position.set(f"{index + 1} / {len(self.visible)} questions" if index >= 0 else "No matching questions")
        label = self.sample.get("label") if self.sample else None
        self.current_label.set(f"Saved: {label['choice']}" if label else "Unlabeled")
        if self.sample:
            seconds = self.sample["stream_seconds"]
            timing = f"{seconds:.1f}s" if seconds is not None else "time unavailable"
            self.status.set(f"{self.sample['sample_id']} | {timing} | {self.sample['activity']} | {self.labeled_count} reviewed this session")
        for button in self.buttons:
            button.state(["!disabled"] if self.sample else ["disabled"])
        self.previous_button.state(["!disabled"] if index > 0 else ["disabled"])
        self.next_button.state(["!disabled"] if 0 <= index < len(self.visible) - 1 else ["disabled"])

    def navigate(self, step):
        index = next((i for i, s in enumerate(self.visible) if self.sample and s["sample_id"] == self.sample["sample_id"]), -1)
        target = index + step
        if 0 <= target < len(self.visible):
            self.select(self.visible[target])

    def next_unlabeled(self):
        current_id = self.sample["sample_id"] if self.sample else None
        index = next((i for i, s in enumerate(self.samples) if s["sample_id"] == current_id), -1)
        ordered = self.samples[index + 1:] + self.samples[:index + 1]
        candidate = next((s for s in ordered if s["sample_id"] != current_id and self.needs_review(s)), None)
        self.run_id.set(ALL_RECORDINGS)
        self.follow_live.set(False)
        self.visible = self.filtered_samples()
        if candidate:
            self.select(candidate)
        else:
            if self.unanswered_only.get() and not self.visible:
                self.sample = None
                self.images.clear()
                self.draw_image(self.frame_canvas)
                self.draw_image(self.crop_canvas)
            self.refresh_details()
            self.status.set("No other unanswered questions across recordings. Skipped questions remain available in the filter.")
            self.draw_timeline()

    def needs_review(self, sample):
        return not sample.get("label") or ((self.include_skipped or self.unanswered_only.get())
                                          and sample["label"]["status"] == "skipped")

    def filtered_samples(self):
        return [s for s in self.samples if (self.run_id.get() == ALL_RECORDINGS or s["run_id"] == self.run_id.get())
                and (not self.unanswered_only.get() or self.needs_review(s))]

    def filter_questions(self):
        self.follow_live.set(False)
        self.visible = self.filtered_samples()
        selected = next((s for s in self.visible if self.sample and s["sample_id"] == self.sample["sample_id"]), None)
        if selected or self.visible:
            self.select(selected or self.visible[0])
        else:
            self.sample = None
            self.images.clear()
            self.draw_image(self.frame_canvas)
            self.draw_image(self.crop_canvas)
            self.refresh_details()
            self.status.set("No unanswered or skipped questions in this view.")
            self.draw_timeline()

    def load_history(self, selected_run, timeline_run=""):
        samples = load_samples(self.recordings)
        runs = sorted(p.name for p in self.recordings.iterdir() if p.is_dir() and ((p / "recording.json").exists() or (p / "frames").exists())) if self.recordings.exists() else []
        selected_run = selected_run if selected_run == ALL_RECORDINGS or selected_run in runs else ALL_RECORDINGS
        if selected_run != ALL_RECORDINGS:
            timeline_run = selected_run
        elif timeline_run not in runs:
            first_pending = next((s for s in samples if not s["label"]), None)
            timeline_run = first_pending["run_id"] if first_pending else runs[-1] if runs else ""
        points = self.reader.read(self.recordings / timeline_run, samples) if timeline_run else []
        return samples, [ALL_RECORDINGS] + runs, selected_run, timeline_run, points

    def poll(self):
        self.poll_id = None
        if self.closed:
            return
        if self.future is not None and self.future.done():
            try:
                samples, runs, selected_run, timeline_run, points = self.future.result()
                # A selection may have changed while the filesystem scan ran.
                if not self.run_id.get() or self.run_id.get() == selected_run:
                    selected_id = self.sample["sample_id"] if self.sample else None
                    for sample in samples:
                        if sample["sample_id"] in self.label_overrides:
                            sample["label"] = self.label_overrides[sample["sample_id"]]
                    self.samples = samples
                    self.run_box.configure(values=runs)
                    self.run_id.set(selected_run)
                    self.visible = self.filtered_samples()
                    if selected_run != ALL_RECORDINGS or not self.timeline_run or self.timeline_run == timeline_run:
                        self.timeline_run = timeline_run
                        self.points = points
                        self.point_times = [p["stream_seconds"] for p in points]
                        self.capture_points = [p for p in points if p["question_ids"]]
                    selected = next((s for s in self.visible if s["sample_id"] == selected_id), None)
                    if self.follow_live.get() and self.visible:
                        latest = self.visible[-1]
                        if selected_id != latest["sample_id"]:
                            self.select(latest, manual=False)
                        else:
                            self.sample = latest
                            self.refresh_details()
                    elif selected:
                        self.sample = selected
                        self.refresh_details()
                    elif self.visible:
                        selected = next((s for s in self.visible if not s["label"]), self.visible[0])
                        self.select(selected, manual=False)
                    else:
                        self.sample = None
                        self.images.clear()
                        self.refresh_details()
                        self.draw_image(self.frame_canvas)
                        self.draw_image(self.crop_canvas)
                    self.draw_timeline()
            except (OSError, ValueError, KeyError) as exc:
                self.status.set(f"Could not refresh history: {exc}")
            self.future = None
        if self.future is None and time.monotonic() >= self.next_scan:
            self.future = self.executor.submit(self.load_history, self.run_id.get(), self.timeline_run)
            self.next_scan = time.monotonic() + 1
        self.poll_id = self.root.after(200, self.poll)

    def change_run(self, event=None):
        self.sample = None
        self.visible = []
        self.points = []
        self.point_times = []
        self.capture_points = []
        self.images.clear()
        self.timeline_run = ""
        self.next_scan = 0
        self.follow_live.set(False)
        self.refresh_details()
        self.draw_image(self.frame_canvas)
        self.draw_image(self.crop_canvas)
        self.draw_timeline()

    def go_live(self):
        self.follow_live.set(True)
        if self.visible:
            self.select(self.visible[-1], manual=False)
        self.draw_timeline()

    def scroll_history(self, *args):
        self.follow_live.set(False)
        self.timeline.xview(*args)
        self.draw_timeline()

    def wheel_history(self, event):
        self.scroll_history("scroll", -3 if event.delta > 0 else 3, "units")
        return "break"

    def draw_timeline(self):
        canvas = self.timeline
        canvas.delete("all")
        end = self.points[-1]["stream_seconds"] if self.points else 0
        extent = max(canvas.winfo_width(), 60 + end * self.pixels_per_second)
        canvas.configure(scrollregion=(0, 0, extent, 145))
        if self.follow_live.get():
            canvas.xview_moveto(1)
        left, right = canvas.canvasx(0), canvas.canvasx(canvas.winfo_width())
        baseline = 113
        canvas.create_line(left, baseline, right, baseline, fill="#555555")
        if not self.points:
            canvas.create_text(left + 20, 45, anchor="w", text="Waiting for detection history...", fill="#dddddd")
            return
        start_seconds = max(0, (left - 30) / self.pixels_per_second)
        end_seconds = (right - 30) / self.pixels_per_second
        start = max(0, bisect_left(self.point_times, start_seconds) - 1)
        finish = bisect_left(self.point_times, end_seconds) + 1
        labels = {s["question_id"]: s["label"] for s in self.samples if s["run_id"] == self.timeline_run}
        for point in self.points[start:finish]:
            x = 30 + point["stream_seconds"] * self.pixels_per_second
            score = point["presence_score"] if self.chart_mode.get() == "Presence" else point["activity_score"]
            stationary = point["train_count"] > 0 and point["active_count"] == 0
            if stationary and self.chart_mode.get() == "Activity":
                score = 0.08
            y = baseline - min(1, score) * 88
            color = "#8b929c" if stationary else "#f3a34c" if score else "#7fb7c9"
            if score:
                canvas.create_line(x, baseline, x, y, fill=color, width=2)
            else:
                canvas.create_line(x, baseline, x + 2, baseline, fill=color, width=2)
            if point["question_ids"]:
                reviewed = all(labels.get(q) is not None and labels[q]["status"] != "skipped" for q in point["question_ids"])
                canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#66c58d" if reviewed else "#70b7ff", outline="")
        first_tick = int(start_seconds // 30) * 30
        for seconds in range(first_tick, int(end_seconds) + 31, 30):
            x = 30 + seconds * self.pixels_per_second
            canvas.create_text(x, 132, text=f"{seconds // 60}:{seconds % 60:02d}", fill="#bbbbbb")
        if self.sample and self.sample["stream_seconds"] is not None:
            x = 30 + self.sample["stream_seconds"] * self.pixels_per_second
            canvas.create_line(x, 8, x, baseline + 5, fill="#ffffff", dash=(3, 3))
        if any(p.get("legacy") for p in self.points[start:finish]):
            canvas.create_text(left + 12, 12, anchor="w", text="Older recording: positive detections only; empty intervals were not logged", fill="#bbbbbb")

    def click_timeline(self, event):
        seconds = (self.timeline.canvasx(event.x) - 30) / self.pixels_per_second
        if not self.capture_points:
            return
        point = min(self.capture_points, key=lambda p: abs(p["stream_seconds"] - seconds))
        if abs(point["stream_seconds"] - seconds) * self.pixels_per_second > 10:
            self.status.set("No saved screenshot at this time. Click a screenshot marker or use the arrows.")
            return
        sample = next((s for s in self.visible if s["question_id"] in point["question_ids"]), None)
        if sample:
            self.select(sample)
        elif self.unanswered_only.get():
            self.status.set("This screenshot is already answered. Turn off the filter to browse it.")

    def choose(self, choice):
        if self.closed or self.sample is None:
            return
        try:
            save_label(self.sample, choice)
            label = read_labels(self.sample["frame_dir"])[self.sample["train_id"]]
        except (OSError, ValueError) as exc:
            self.messagebox.showerror("Could not save label", str(exc), parent=self.root)
            return
        self.sample["label"] = label
        self.label_overrides[self.sample["sample_id"]] = label
        self.labeled_count += 1
        self.draw_timeline()
        self.next_unlabeled()

    def toggle_image_server(self):
        if self.image_server is not None:
            self.stop_image_server()
            self.status.set("Image server stopped.")
            return
        try:
            self.recordings.mkdir(parents=True, exist_ok=True)
            handler = partial(SimpleHTTPRequestHandler, directory=str(self.recordings.resolve()))
            server = ThreadingHTTPServer(("0.0.0.0", 8000), handler)
        except OSError as exc:
            self.messagebox.showerror("Could not start image server", f"Port 8000 may already be in use.\n{exc}", parent=self.root)
            return
        self.image_server = server
        self.server_thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        try:
            self.server_thread.start()
        except RuntimeError as exc:
            server.server_close()
            self.image_server = self.server_thread = None
            self.messagebox.showerror("Could not start image server", str(exc), parent=self.root)
            return
        self.server_button.configure(text="Stop image server")
        self.messagebox.showinfo(
            "Image server running",
            "On this computer: http://localhost:8000/\n\n"
            "Your brother can open http://YOUR-LAN-IP:8000/ on the same network.\n"
            "Find your LAN IP with: hostname -I\n\n"
            "Browse a recording, then frames/frame_NUMBER/frame.jpg for the full image.\n"
            "Crops, scan results, labels, and existing files in recordings are also shared.\n"
            "This is a local network server with no login. Closing the GUI stops it.",
            parent=self.root)

    def stop_image_server(self):
        if self.image_server is None:
            return
        self.image_server.shutdown()
        self.image_server.server_close()
        self.server_thread.join(timeout=1)
        self.image_server = self.server_thread = None
        self.server_button.configure(text="Start image server")

    def close(self):
        self.closed = True
        if self.poll_id is not None:
            self.root.after_cancel(self.poll_id)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.stop_image_server()
        self.root.destroy()
