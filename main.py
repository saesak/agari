from __future__ import annotations

import os
import time
import tkinter as tk
from tkinter import ttk
from typing import Any

import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image, ImageTk

# --- Paths + constants ---

HERE: str = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH: str = os.path.join(HERE, "face_landmarker.task")

PREVIEW_W: int = 320
PREVIEW_H: int = 240
TICK_MS: int = 33

# Peak-over-window for fast-gesture detection. At 30fps, 3 frames ≈ 100ms —
# catches brief peaks that would be missed by instantaneous thresholding.
SCORE_WINDOW: int = 3

ACTION_CLICK: str = "Click"
ACTION_SCROLL_DOWN: str = "Scroll down"
ACTION_SCROLL_UP: str = "Scroll up"
ACTIONS: list[str] = [ACTION_CLICK, ACTION_SCROLL_DOWN, ACTION_SCROLL_UP]

# Mouth-center landmarks (upper/lower inner lip) — for the preview dot only.
MOUTH_UPPER_LM: int = 13
MOUTH_LOWER_LM: int = 14


# --- MediaPipe ---


def make_landmarker() -> vision.FaceLandmarker:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"face_landmarker.task not found at {MODEL_PATH} — "
            "the model ships with the repo, did you delete it?"
        )
    base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.FaceLandmarkerOptions(
        base_options=base,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def jaw_open_score(result: Any) -> float:
    if not result.face_blendshapes:
        return 0.0
    for bs in result.face_blendshapes[0]:
        if bs.category_name == "jawOpen":
            return float(bs.score)
    return 0.0


# --- App ---


class App:
    def __init__(
        self,
        root: tk.Tk,
        cap: cv2.VideoCapture,
        landmarker: vision.FaceLandmarker,
    ) -> None:
        self.root: tk.Tk = root
        self.cap: cv2.VideoCapture = cap
        self.landmarker: vision.FaceLandmarker = landmarker

        self.prev_open: bool = False
        self.last_trigger: float = 0.0
        self._alive: bool = True
        self.score_history: list[float] = []
        self._after_id: str | None = None

        self.enabled: tk.BooleanVar = tk.BooleanVar(value=True)
        self.action: tk.StringVar = tk.StringVar(value=ACTION_SCROLL_DOWN)
        self.threshold: tk.DoubleVar = tk.DoubleVar(value=0.4)
        self.scroll_amount: tk.IntVar = tk.IntVar(value=5)
        self.cooldown: tk.DoubleVar = tk.DoubleVar(value=0.4)

        self._build()
        self._after_id = self.root.after(TICK_MS, self._tick)

    def _build(self) -> None:
        for c in self.root.winfo_children():
            c.destroy()
        self.root.title("agari")
        self.root.resizable(False, False)

        pad: dict[str, int] = {"padx": 8, "pady": 4}
        self._placeholder = ImageTk.PhotoImage(
            Image.new("RGB", (PREVIEW_W, PREVIEW_H), color=(0, 0, 0))
        )
        self.preview = tk.Label(self.root, image=self._placeholder, bg="black")
        self.preview.grid(row=0, column=0, columnspan=2, **pad)

        meter_frame = tk.Frame(self.root)
        meter_frame.grid(row=1, column=0, columnspan=2, sticky="ew", **pad)
        tk.Label(meter_frame, text="jaw").pack(side=tk.LEFT)
        self.meter = ttk.Progressbar(
            meter_frame, length=220, maximum=1.0, mode="determinate"
        )
        self.meter.pack(side=tk.LEFT, padx=6)
        self.score_label = tk.Label(meter_frame, text="0.00", width=5)
        self.score_label.pack(side=tk.LEFT)

        tk.Label(self.root, text="Action").grid(row=2, column=0, sticky="e", **pad)
        ttk.Combobox(
            self.root, textvariable=self.action, values=ACTIONS,
            state="readonly", width=18,
        ).grid(row=2, column=1, sticky="w", **pad)

        tk.Label(self.root, text="Scroll step").grid(row=3, column=0, sticky="e", **pad)
        tk.Scale(
            self.root, from_=1, to=30, orient=tk.HORIZONTAL,
            variable=self.scroll_amount, length=220,
        ).grid(row=3, column=1, sticky="w", **pad)

        tk.Label(self.root, text="Threshold").grid(row=4, column=0, sticky="e", **pad)
        tk.Scale(
            self.root, from_=0.15, to=0.9, resolution=0.01, orient=tk.HORIZONTAL,
            variable=self.threshold, length=220,
        ).grid(row=4, column=1, sticky="w", **pad)

        tk.Label(self.root, text="Cooldown (s)").grid(row=5, column=0, sticky="e", **pad)
        tk.Scale(
            self.root, from_=0.1, to=1.5, resolution=0.05, orient=tk.HORIZONTAL,
            variable=self.cooldown, length=220,
        ).grid(row=5, column=1, sticky="w", **pad)

        tk.Checkbutton(
            self.root, text="Active", variable=self.enabled,
            font=("Helvetica", 14, "bold"),
        ).grid(row=6, column=0, columnspan=2, **pad)

    def _tick(self) -> None:
        if not self._alive:
            return
        try:
            ok, frame = self.cap.read()
            if not ok:
                return

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)
            score: float = jaw_open_score(result)

            if not result.face_blendshapes:
                self.score_history = []
                self.prev_open = False
                effective_score: float = 0.0
            else:
                self.score_history.append(score)
                if len(self.score_history) > SCORE_WINDOW:
                    self.score_history.pop(0)
                effective_score = max(self.score_history)

            threshold: float = self.threshold.get()
            is_open: bool = effective_score > threshold

            self.meter["value"] = effective_score
            self.score_label.config(text=f"{effective_score:.2f}")

            if self.enabled.get() and is_open and not self.prev_open:
                now: float = time.monotonic()
                if now - self.last_trigger >= self.cooldown.get():
                    try:
                        self._fire()
                    except Exception as e:
                        print(f"agari: action failed: {e}", flush=True)
                    self.last_trigger = now
            self.prev_open = is_open

            # Preview overlays — dot on mouth center + green border when open.
            if result.face_landmarks:
                lms = result.face_landmarks[0]
                h, w = rgb.shape[:2]
                mx: int = int((lms[MOUTH_UPPER_LM].x + lms[MOUTH_LOWER_LM].x) / 2 * w)
                my: int = int((lms[MOUTH_UPPER_LM].y + lms[MOUTH_LOWER_LM].y) / 2 * h)
                cv2.circle(rgb, (mx, my), 3, (80, 180, 255), -1)
            if is_open:
                cv2.rectangle(
                    rgb, (0, 0), (rgb.shape[1] - 1, rgb.shape[0] - 1), (0, 255, 0), 6
                )

            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.preview.config(image=photo)
            self.preview.image = photo
        except tk.TclError:
            # Widgets gone mid-tick — just stop rescheduling
            return
        except Exception as e:
            print(f"agari: tick error: {e}", flush=True)
        finally:
            if self._alive:
                self._after_id = self.root.after(TICK_MS, self._tick)

    def _fire(self) -> None:
        action: str = self.action.get()
        if action == ACTION_CLICK:
            pyautogui.click()
        elif action == ACTION_SCROLL_DOWN:
            pyautogui.scroll(-self.scroll_amount.get())
        elif action == ACTION_SCROLL_UP:
            pyautogui.scroll(self.scroll_amount.get())

    def _on_close(self) -> None:
        self._alive = False
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None


# --- Entry point ---


def main() -> None:
    root = tk.Tk()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_H)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("could not open camera")

    landmarker: vision.FaceLandmarker | None = None
    try:
        landmarker = make_landmarker()
        app = App(root, cap, landmarker)
    except BaseException:
        cap.release()
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass
        raise

    def on_close() -> None:
        app._on_close()
        try:
            cap.release()
            landmarker.close()
        finally:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
