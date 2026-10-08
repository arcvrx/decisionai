"""Tkinter screenshot labeling with clickable, scrollable detection history."""

from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time

from label import (CHOICES, TimelineReader, apply_box_preset, frame_annotations,
                   load_samples, read_box_presets, read_labels, save_box_preset,
                   save_frame_boxes, save_label, test_gemini_frame)

ALL_RECORDINGS = "All recordings"

def review_counts(samples):
    unanswered = sum(not s.get("label") for s in samples)
    skipped = sum(bool(s.get("label")) and s["label"]["status"] == "skipped" for s in samples)
    return len(samples) - unanswered - skipped, unanswered, skipped


class BoxEditor:
    """Edit separate train rectangles in original image coordinates."""

    def __init__(self, reviewer):
        import tkinter as tk
        from tkinter import ttk
        self.reviewer = reviewer
        self.frame_dir = reviewer.sample["frame_dir"]
        report = frame_annotations(self.frame_dir)
        self.boxes = [{"train_id": d["train_id"], "bbox": list(d["bbox"])} for d in report["detections"]]
        self.active = next((b for b in self.boxes if str(b["train_id"]) == reviewer.sample["train_id"]), None)
        self.next_id = max(report.get("next_train_id", 1), max((b["train_id"] for b in self.boxes), default=0) + 1)
        with reviewer.Image.open(self.frame_dir / "frame.jpg") as source:
            self.image = source.convert("RGB")
        self.drag = None
        self.adding = False
        self.window = tk.Toplevel(reviewer.root)
        self.window.title("Correct train boxes")
        self.window.geometry("1200x800")
        self.window.transient(reviewer.root)
        self.window.grab_set()
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)
        ttk.Label(self.window, text="Click a box to select it. Drag a corner to resize, or drag inside to move. New box: drag around one train.", padding=10).grid(row=0, column=0, sticky="w")
        self.canvas = tk.Canvas(self.window, background="#191919", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda event: self.draw())
        self.canvas.bind("<Button-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.motion)
        self.canvas.bind("<ButtonRelease-1>", lambda event: self.release())
        controls = ttk.Frame(self.window, padding=10)
        controls.grid(row=2, column=0, sticky="ew")
        ttk.Button(controls, text="New box", command=self.new_box).pack(side="left")
        ttk.Button(controls, text="Delete selected box", command=self.delete_box).pack(side="left", padx=8)
        self.reviewed = tk.BooleanVar(value=report.get("boxes_reviewed", False))
        ttk.Checkbutton(controls, text="I checked every visible train in this frame", variable=self.reviewed).pack(side="left", padx=8)
        ttk.Button(controls, text="Save corrections", command=self.save).pack(side="right")
        ttk.Button(controls, text="Cancel", command=self.close).pack(side="right", padx=8)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda event: self.close())

    def draw(self):
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        resized = self.image.copy()
        resized.thumbnail((width, height), self.reviewer.Image.Resampling.LANCZOS)
        self.sx, self.sy = resized.width / self.image.width, resized.height / self.image.height
        self.ox, self.oy = (width - resized.width) / 2, (height - resized.height) / 2
        self.photo = self.reviewer.ImageTk.PhotoImage(resized, master=self.window)
        self.canvas.delete("all")
        self.canvas.create_image(self.ox, self.oy, image=self.photo, anchor="nw")
        for box in self.boxes:
            x1, y1, x2, y2 = box["bbox"]
            coords = (self.ox + x1 * self.sx, self.oy + y1 * self.sy, self.ox + x2 * self.sx, self.oy + y2 * self.sy)
            color = "lime" if box is self.active else "orange"
            self.canvas.create_rectangle(*coords, outline=color, width=2)
            self.canvas.create_text(coords[0] + 4, coords[1] + 4, text=f"Train {box['train_id']}", anchor="nw", fill=color)
            if box is self.active:
                for x, y in ((coords[0], coords[1]), (coords[2], coords[1]), (coords[2], coords[3]), (coords[0], coords[3])):
                    self.canvas.create_rectangle(x-4, y-4, x+4, y+4, fill="lime", outline="black")

    def image_point(self, event):
        return (max(0, min(self.image.width, (event.x - self.ox) / self.sx)),
                max(0, min(self.image.height, (event.y - self.oy) / self.sy)))

    def new_box(self):
        self.adding = True
        self.canvas.configure(cursor="crosshair")

    def press(self, event):
        if not hasattr(self, "sx"):
            return
        x, y = self.image_point(event)
        if self.adding:
            self.active = {"train_id": self.next_id, "bbox": [x, y, x, y]}
            self.next_id += 1
            self.boxes.append(self.active)
            self.drag = ("resize", (x, y))
            self.adding = False
            self.canvas.configure(cursor="")
        else:
            if self.active:
                x1, y1, x2, y2 = self.active["bbox"]
                corners = [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
                for i, (cx, cy) in enumerate(corners):
                    if abs(x-cx)*self.sx <= 10 and abs(y-cy)*self.sy <= 10:
                        self.drag = ("resize", corners[(i+2) % 4])
                        return
            containing = [b for b in self.boxes if b["bbox"][0] <= x <= b["bbox"][2] and b["bbox"][1] <= y <= b["bbox"][3]]
            if containing:
                self.active = min(containing, key=lambda b: (b["bbox"][2]-b["bbox"][0])*(b["bbox"][3]-b["bbox"][1]))
                self.drag = ("move", (x, y), list(self.active["bbox"]))
        self.draw()

    def motion(self, event):
        if not self.drag or self.active is None:
            return
        x, y = self.image_point(event)
        if self.drag[0] == "resize":
            ax, ay = self.drag[1]
            self.active["bbox"] = [min(ax,x), min(ay,y), max(ax,x), max(ay,y)]
        else:
            ax, ay = self.drag[1]
            x1, y1, x2, y2 = self.drag[2]
            dx = max(-x1, min(self.image.width-x2, x-ax))
            dy = max(-y1, min(self.image.height-y2, y-ay))
            self.active["bbox"] = [x1+dx, y1+dy, x2+dx, y2+dy]
        self.reviewed.set(False)
        self.draw()

    def release(self):
        self.drag = None

    def delete_box(self):
        if self.active is not None:
            self.boxes.remove(self.active)
            self.active = self.boxes[0] if self.boxes else None
            self.reviewed.set(False)
            self.draw()

    def save(self):
        try:
            report = save_frame_boxes(self.frame_dir, self.boxes, self.reviewed.get())
        except (OSError, ValueError) as exc:
            self.reviewer.messagebox.showerror("Could not save boxes", str(exc), parent=self.window)
            return
        selected_id = str(self.active["train_id"]) if self.active else None
        self.close()
        self.reviewer.boxes_saved(self.frame_dir, selected_id, report)

    def close(self):
        self.reviewer.editor = None
        self.window.destroy()


class LabelReviewer:
    def __init__(self, root, recordings, include_skipped=False):
        import tkinter as tk
        from tkinter import messagebox, simpledialog, ttk
        from PIL import Image, ImageDraw, ImageTk

        self.root = root
        self.recordings = Path(recordings)
        self.include_skipped = include_skipped
        self.messagebox = messagebox
        self.simpledialog = simpledialog
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
        self.gemini_executor = ThreadPoolExecutor(max_workers=1)
        self.gemini_future = None
        self.future = None
        self.next_scan = 0
        self.pixels_per_second = 3
        self.run_id = tk.StringVar(value=ALL_RECORDINGS)
        self.timeline_run = ""
        self.chart_mode = tk.StringVar(value="Activity")
        self.follow_live = tk.BooleanVar(value=False)
        self.unanswered_only = tk.BooleanVar(value=True)
        self.review_summary = tk.StringVar()
        self.preset_name = tk.StringVar()
        self.auto_preset = tk.BooleanVar(value=False)
        try:
            self.box_presets = read_box_presets(self.recordings)
        except (OSError, ValueError):
            self.box_presets = {}
        self.editor = None
        self.discard_scan = False
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
        self.gemini_button = ttk.Button(toolbar, text="Test full frame: Gemini", command=self.start_gemini_test)
        self.gemini_button.pack(side="right", padx=4)

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
        self.edit_button = ttk.Button(choices, text="Edit / add / delete train boxes", command=self.edit_boxes)
        self.edit_button.pack(fill="x", pady=(0, 10))
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
        preset_panel = ttk.LabelFrame(choices, text="Reusable train boxes", padding=8)
        preset_panel.pack(fill="x", pady=(10, 0))
        self.preset_box = ttk.Combobox(preset_panel, textvariable=self.preset_name,
                                      values=list(self.box_presets), state="readonly")
        self.preset_box.pack(fill="x")
        preset_actions = ttk.Frame(preset_panel)
        preset_actions.pack(fill="x", pady=(5, 0))
        ttk.Button(preset_actions, text="Save box…", command=self.save_preset).pack(side="left", fill="x", expand=True)
        ttk.Button(preset_actions, text="Apply", command=self.apply_preset).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(preset_actions, text="Add train", command=lambda: self.apply_preset(add=True)).pack(side="left", fill="x", expand=True)
        ttk.Checkbutton(preset_panel, text="Apply preset when labeling (1–6)", variable=self.auto_preset).pack(anchor="w", pady=(5, 0))
        ttk.Label(preset_panel, text="Fixed image area; check it fits each train.", wraplength=280).pack(anchor="w")
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
            button.state(["!disabled"] if self.sample and not self.sample.get("deleted") else ["disabled"])
        self.previous_button.state(["!disabled"] if index > 0 else ["disabled"])
        self.next_button.state(["!disabled"] if 0 <= index < len(self.visible) - 1 else ["disabled"])
        self.edit_button.state(["!disabled"] if self.sample else ["disabled"])
        self.gemini_button.state(["!disabled"] if self.sample and self.gemini_future is None else ["disabled"])

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
        if self.gemini_future is not None and self.gemini_future.done():
            future, self.gemini_future = self.gemini_future, None
            try:
                report, output_path = future.result()
                lines = [f"{key.replace('_', ' ')}: {percentage:.1f}%" for key, percentage in report["percentages"].items()]
                self.messagebox.showinfo("Gemini full-frame results", f"Frame: {report['frame_path']}\n\n" + "\n".join(lines)
                    + f"\n\n{report['explanation']}\n\nModel estimates, not calibrated probabilities.\nSaved: {output_path}", parent=self.root)
            except Exception as exc:
                self.messagebox.showerror("Gemini test failed", str(exc), parent=self.root)
            self.refresh_details()
        if self.editor is not None:
            self.poll_id = self.root.after(200, self.poll)
            return
        if self.future is not None and self.future.done():
            if self.discard_scan:
                self.future = None
                self.discard_scan = False
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
        if self.closed or self.sample is None or self.sample.get("deleted"):
            return
        if self.auto_preset.get() and choice[3] == "labeled" and not self.apply_preset():
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

    def start_gemini_test(self):
        if self.sample is None or self.gemini_future is not None:
            return
        self.gemini_future = self.gemini_executor.submit(test_gemini_frame, self.sample["frame_path"],
                                                       Path("outputs") / f"gemini_{time.time_ns()}")
        self.gemini_button.state(["disabled"])
        self.status.set("Testing the full frame with Gemini...")

    def save_preset(self):
        if self.sample is None:
            return
        name = self.simpledialog.askstring("Save box preset", "Name this box (for example: left freight or parked Surfliner):",
                                           initialvalue=self.preset_name.get(), parent=self.root)
        if not name or not name.strip():
            return
        try:
            self.box_presets = save_box_preset(self.recordings, name, self.sample)
        except (OSError, ValueError) as exc:
            self.messagebox.showerror("Could not save preset", str(exc), parent=self.root)
            return
        self.preset_box.configure(values=list(self.box_presets))
        self.preset_name.set(name.strip())
        self.status.set(f"Saved reusable box: {name.strip()}")

    def apply_preset(self, add=False):
        if self.sample is None:
            return False
        preset = self.box_presets.get(self.preset_name.get())
        if preset is None:
            self.messagebox.showerror("Choose a box preset", "Save or select a box preset first.", parent=self.root)
            return False
        try:
            frame_dir = self.sample["frame_dir"]
            pending_only = self.unanswered_only.get()
            train_id, report = apply_box_preset(self.sample, preset, add=add)
            self.boxes_saved(frame_dir, train_id, report)
            self.unanswered_only.set(pending_only)
            self.visible = self.filtered_samples()
            self.refresh_details()
        except (OSError, ValueError, KeyError) as exc:
            self.messagebox.showerror("Could not apply preset", str(exc), parent=self.root)
            return False
        return True

    def edit_boxes(self):
        if self.sample is None or self.editor is not None:
            return
        self.follow_live.set(False)
        try:
            self.editor = BoxEditor(self)
        except (OSError, ValueError) as exc:
            self.messagebox.showerror("Could not open box editor", str(exc), parent=self.root)

    def boxes_saved(self, frame_dir, train_id, report):
        # Discard a filesystem scan started before these corrections.
        self.discard_scan = self.future is not None
        self.next_scan = 0
        for detection in report["detections"]:
            key = detection["question_id"]
            self.label_overrides.pop(key, None)
        for key in list(self.label_overrides):
            if f"/{frame_dir.name}/" in key and key.startswith(frame_dir.parent.parent.name + "/"):
                del self.label_overrides[key]
        self.samples = load_samples(self.recordings)
        target = next((s for s in self.samples if s["frame_dir"] == frame_dir and s["train_id"] == train_id), None)
        # Keep the corrected frame accessible even if its chosen box was labeled.
        self.unanswered_only.set(False)
        self.visible = self.filtered_samples()
        if target:
            self.select(target)
        elif self.visible:
            self.select(self.visible[0])
        else:
            self.sample = None
            self.images.clear()
            self.refresh_details()
            self.draw_image(self.frame_canvas)
            self.draw_image(self.crop_canvas)
        self.status.set("Boxes saved. Changed crops need a new operator label. Original detector results are preserved.")

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
        self.gemini_executor.shutdown(wait=False, cancel_futures=True)
        self.stop_image_server()
        self.root.destroy()
