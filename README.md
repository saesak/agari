# agari

<p align="center">
  <img src="agari.gif" alt="agari" width="320">
</p>

*agari* (아가리) — Korean slang for mouth.

Hands-free manga and webtoon reader. Open your mouth, page scrolls.

For manga websites that require clicking, webtoons that require scrolling, and dense PDFs — without destroying your wrist. MediaPipe Face Landmarker reads the `jawOpen` blendshape; a quick mouth-open fires a click or scroll via `pyautogui`. Works in whatever reader you have in the foreground — Webtoons, MangaDex, Tachidesk, PDFs. Zero calibration, runs 100% locally, no network calls ever.

## Install

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/saesak/agari.git
cd agari
uv sync
```

`uv sync` creates `.venv/` and installs the dependencies (mediapipe, opencv-python, pyautogui, pillow, numpy). The MediaPipe model file ships in the repo, so this is the only setup step — no separate downloads.

## Run

```bash
uv run main.py
```

First launch will prompt for camera (and Accessibility on macOS — see below).

## Permissions

The app needs camera access to see your face, and input-synthesis permission to send clicks/scrolls to other apps.

- **macOS**: Camera + Accessibility (System Settings → Privacy & Security → Accessibility → allow your terminal / Python)
- **Windows**: Camera only — input synthesis works without extra setup
- **Linux (X11)**: Camera only — `pyautogui` works through Xlib
- **Linux (Wayland)**: input synthesis is blocked by the compositor for security. Run under XWayland, or install `ydotool` and route through it. Out-of-the-box pyautogui won't drive the cursor.

Heads-up on whichever permission your OS calls it: granting input synthesis lets the running program type and click anywhere — Terminal, system dialogs (including "Allow"), your password manager, etc. This is true for anything using `pyautogui` or similar — only grant it to code you've actually read.

## Modes

- **Click** — one click per mouth-open gesture
- **Scroll down** — one scroll step per gesture
- **Scroll up** — same, reversed

Threshold, scroll step, and cooldown are all live-editable in the UI.

## Model file

`face_landmarker.task` is MediaPipe's float16 face landmarker, vendored so the app runs offline. If you're cloning a fork and want to confirm the binary hasn't been swapped, verify against the original:

```bash
shasum -a 256 face_landmarker.task
# 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
```

Source: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
