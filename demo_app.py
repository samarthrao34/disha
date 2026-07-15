"""Tkinter professor-demo application for DISHA."""

import json
import os
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from disha.live_capture import capture_webcam_frame, record_microphone
from disha.runtime import DishaEngine


class DishaDemo(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DISHA - Multimodal Emotion-Aware Research Demo")
        self.geometry("1040x760")
        self.minsize(900, 650)
        self.engine = DishaEngine()
        self.capture_dir = ROOT / "data" / "processed" / "live"
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.temporary_captures: set[Path] = set()
        self.last_result = None
        self.image_path = tk.StringVar()
        self.audio_path = tk.StringVar()
        self.status = tk.StringVar(value="Ready")
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _build(self) -> None:
        style = ttk.Style(self)
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("Sub.TLabel", font=("Segoe UI", 10))

        outer = ttk.Frame(self, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="DISHA", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "Research prototype - live/file face, speech and text; "
                "quality-aware SUTRA fusion; session tracking; bounded safety actions"
            ),
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(0, 12))

        ttk.Label(outer, text="User message").pack(anchor="w")
        self.text = tk.Text(outer, height=5, wrap="word", font=("Segoe UI", 11))
        self.text.pack(fill="x", pady=(4, 10))

        files = ttk.Frame(outer)
        files.pack(fill="x")
        ttk.Button(files, text="Choose face image", command=self.choose_image).grid(
            row=0, column=0, padx=(0, 8), pady=4
        )
        ttk.Label(files, textvariable=self.image_path).grid(row=0, column=1, sticky="w")
        ttk.Button(files, text="Capture webcam", command=self.capture_face).grid(
            row=0, column=2, padx=8, pady=4
        )
        ttk.Button(files, text="Choose speech WAV", command=self.choose_audio).grid(
            row=1, column=0, padx=(0, 8), pady=4
        )
        ttk.Label(files, textvariable=self.audio_path).grid(row=1, column=1, sticky="w")
        ttk.Button(files, text="Record 4 seconds", command=self.record_audio).grid(
            row=1, column=2, padx=8, pady=4
        )
        files.columnconfigure(1, weight=1)

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=10)
        self.analyze_button = ttk.Button(
            controls, text="Analyze with DISHA", command=self.analyze
        )
        self.analyze_button.pack(side="left")
        ttk.Button(controls, text="Clear session", command=self.clear).pack(side="left", padx=8)
        ttk.Button(
            controls, text="Export sanitized report", command=self.export_report
        ).pack(side="left")
        ttk.Label(controls, textvariable=self.status).pack(side="right")

        self.output = tk.Text(outer, wrap="word", font=("Consolas", 10), state="disabled")
        self.output.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text=(
                "DISHA is not a therapist or diagnostic system. Crisis behavior is a "
                "conservative research safeguard, not a validated clinical service."
            ),
            foreground="#8a3b12",
            wraplength=960,
        ).pack(anchor="w", pady=(10, 0))

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.image_path.set(path)

    def choose_audio(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("WAV audio", "*.wav")])
        if path:
            self.audio_path.set(path)

    def capture_face(self) -> None:
        target = self.capture_dir / f"webcam_{time.time_ns()}.jpg"
        self.status.set("Capturing webcam...")
        threading.Thread(target=self._capture_face_worker, args=(target,), daemon=True).start()

    def _capture_face_worker(self, target: Path) -> None:
        try:
            path = capture_webcam_frame(target)
            self.temporary_captures.add(path)
            self.after(
                0,
                lambda: self._capture_finished(
                    self.image_path, path, "Webcam frame captured"
                ),
            )
        except Exception as exc:
            self.after(0, lambda: self._capture_failed(exc))

    def record_audio(self) -> None:
        target = self.capture_dir / f"microphone_{time.time_ns()}.wav"
        self.status.set("Recording for 4 seconds...")
        threading.Thread(target=self._record_audio_worker, args=(target,), daemon=True).start()

    def _record_audio_worker(self, target: Path) -> None:
        try:
            path = record_microphone(target)
            self.temporary_captures.add(path)
            self.after(
                0,
                lambda: self._capture_finished(
                    self.audio_path, path, "Microphone recording captured"
                ),
            )
        except Exception as exc:
            self.after(0, lambda: self._capture_failed(exc))

    def _capture_finished(self, variable: tk.StringVar, path: Path, status: str) -> None:
        variable.set(str(path))
        self.status.set(status)

    def _capture_failed(self, exc: Exception) -> None:
        self.status.set("Capture failed")
        messagebox.showerror("DISHA capture", str(exc))

    def clear(self) -> None:
        self.text.delete("1.0", "end")
        self.image_path.set("")
        self.audio_path.set("")
        self.last_result = None
        self.engine.reset_session()
        self._delete_temporary_captures()
        self._show("")
        self.status.set("Ready - session cleared")

    def analyze(self) -> None:
        text = self.text.get("1.0", "end").strip()
        if not any((text, self.image_path.get(), self.audio_path.get())):
            messagebox.showinfo("DISHA", "Enter text or choose/capture image or audio input.")
            return
        self.analyze_button.configure(state="disabled")
        self.status.set("Analyzing... first model load can take about 30 seconds")
        threading.Thread(target=self._analyze_worker, args=(text,), daemon=True).start()

    def _analyze_worker(self, text: str) -> None:
        try:
            result = self.engine.process(
                text=text or None,
                image_path=self.image_path.get() or None,
                audio_path=self.audio_path.get() or None,
            )
            self.last_result = result
            trace = result.reasoning_trace
            lines = [
                f"DISHA: {result.response}",
                "",
                f"Action: {result.action}",
                f"Dominant emotion: {trace.get('dominant_emotion', 'n/a')}",
                f"Crisis level: {trace.get('crisis_level', 'low')}",
                f"Modalities used: {', '.join(trace.get('modalities_used', [])) or 'none'}",
                f"Measured latency: {result.latency_ms:.2f} ms",
                "",
                "Fused probabilities:",
                json.dumps(trace.get("fused_probabilities", {}), indent=2),
                "",
                "Session state:",
                json.dumps(result.session_state, indent=2),
                "",
                "Evidence quality:",
            ]
            for item in result.evidence:
                uncertainty = item.get("predictive_entropy")
                uncertainty_text = "n/a" if uncertainty is None else f"{uncertainty:.3f}"
                lines.append(
                    f"- {item['modality']}: reliability={item['reliability_score']:.3f}, "
                    f"uncertainty={uncertainty_text}, model={item['model_name']}"
                )
            self.after(0, lambda: self._finish("\n".join(lines)))
        except Exception as exc:
            self.after(0, lambda: self._finish(f"ERROR: {exc}"))

    def export_report(self) -> None:
        if self.last_result is None:
            messagebox.showinfo("DISHA", "Run an analysis before exporting a report.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON report", "*.json")],
            initialfile="disha_sanitized_session_report.json",
        )
        if path:
            Path(path).write_text(
                json.dumps(self.last_result.to_dict(), indent=2), encoding="utf-8"
            )
            self.status.set("Sanitized report exported")

    def _finish(self, content: str) -> None:
        self._show(content)
        self.status.set("Ready")
        self.analyze_button.configure(state="normal")

    def _show(self, content: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", content)
        self.output.configure(state="disabled")

    def _delete_temporary_captures(self) -> None:
        for path in tuple(self.temporary_captures):
            try:
                path.unlink(missing_ok=True)
            finally:
                self.temporary_captures.discard(path)

    def close(self) -> None:
        self._delete_temporary_captures()
        self.destroy()


if __name__ == "__main__":
    os.chdir(ROOT)
    DishaDemo().mainloop()
