# IFF — Identification Friend or Foe

> A real-time surveillance system that detects people on a camera feed, recognizes known faces as **FRIEND**, flags unknown faces as **FOE**, and raises an alert when an intruder is confirmed.

Internally branded **BorderGuard**, this project combines [YOLOv8](https://github.com/ultralytics/ultralytics) person detection with [DeepFace](https://github.com/serengil/deepface) (ArcFace) face recognition to build a lightweight "Identification Friend or Foe" pipeline that runs on a regular laptop — no GPU required.

---

## Features

- **Real-time person detection** using YOLOv8 (`yolov8n` / `yolov8s`).
- **Friend / Foe classification** — faces are matched against a folder of known "friend" reference images using ArcFace embeddings and cosine similarity.
- **Lightweight tracker** — assigns a stable ID to each person across frames so labels stay consistent.
- **Confirmation logic** — a person must stay close and be classified as FOE for a set time before an alert fires, which cuts down on false alarms.
- **Smoothing & momentum** — similarity scores are averaged over recent frames and require consecutive agreeing frames before a label is committed.
- **Multiple alert modes** — local sound + image popup, or email alerts with snapshots attached.
- **CSV logging** — every detection is written to `detections_log.csv` with timestamp, track ID, label, similarity score, and bounding box.
- **Snapshots** — full-frame captures and a face "montage" of detected foes are saved to a `snapshots/` folder.
- **Multiple camera sources** — built-in webcam, Android IP Webcam (MJPEG stream or JPEG polling fallback), or an RTSP/HTTP URL.

---

## Project structure

| File | Description |
|------|-------------|
| `iff.py` | **BorderGuard LIVE** — person detection tuned for an Android IP Webcam feed, with pose confirmation, animal suppression, and region/geofence masking. |
| `iff_final.py` | **Full version** — FRIEND/FOE/HUMAN classification with ArcFace, plus **email alerts** (SMTP) that attach a snapshot and a face montage. |
| `iff1.py` | **Local-only version** — same FRIEND/FOE pipeline, but instead of email it plays an alert sound and opens the snapshot in your default image viewer. |
| `detections_log.csv` | Example output log of detections. |
| `yolov8n.pt`, `yolov8s.pt`, `yolov8n-pose.pt` | Pretrained YOLOv8 model weights. |

---

## Installation

Requires **Python 3.9+**.

```bash
# Clone the repository
git clone https://github.com/<your-username>/iff.git
cd iff

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# Install dependencies
pip install ultralytics opencv-python numpy requests torch deepface python-dotenv
```

> **Note:** `deepface` is only needed for the FRIEND/FOE scripts (`iff_final.py`, `iff1.py`). The first run will download the ArcFace model automatically.

---

## Setup

### 1. Add friend reference images

Create a folder named `frnd_imgs/` and add clear, well-lit face photos of people who should be recognized as **FRIEND**. Name each file with the person's name first, e.g.:

```
frnd_imgs/
├── alex_1.jpeg
├── alex_2.jpeg
├── sam_1.jpeg
└── sam_2.jpeg
```

The text before the first underscore is used as the displayed name (`alex_1.jpeg` → "Alex"). Two or three photos per person improves accuracy.

### 2. Create a `.env` file

The scripts read configuration from a `.env` file in the project folder. Create one like this:

```ini
# Camera source: 0 for built-in webcam, or a stream URL
CAMERA_URL=0

# Model and friend images
LOCAL_MODEL=yolov8n.pt
FRIEND_DIR=frnd_imgs
FRIEND_IMGS=*               # "*" loads every image in FRIEND_DIR

# Detection tuning (optional — these are the defaults)
FRIEND_THRESHOLD=0.42
CONFIRMATION_TIME=2.0
ALERT_COOLDOWN=45.0

# Email alerts (only needed for iff_final.py)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=465
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password
TO_EMAIL=recipient@gmail.com
```

> **Security tip:** never commit your `.env` file. Add it to `.gitignore`. For Gmail, use an [App Password](https://support.google.com/accounts/answer/185833), not your real password.

---

## Usage

Run whichever variant fits your setup:

```bash
# Local-only alerts (sound + image popup)
python iff1.py

# Email alerts with snapshots
python iff_final.py

# IP-webcam-focused live detection
python iff.py
```

A window opens showing the camera feed with labeled bounding boxes:

- 🟡 **HUMAN** — a person detected but too far away or not yet classified.
- 🟢 **FRIEND** — a recognized face (the person's name is shown).
- 🔴 **FOE** — an unrecognized face.

**Keyboard controls:**

- `q` — quit
- `s` — save a manual snapshot

When a FOE stays close for longer than `CONFIRMATION_TIME`, the HUD switches to **🔴 Intruder Detected** and an alert is triggered (subject to `ALERT_COOLDOWN`).

---

## How it works

1. **Detect** — YOLOv8 finds every person in the frame.
2. **Track** — each person is matched to an existing track by center distance, keeping IDs stable across frames.
3. **Classify** — when a person is close enough (their bounding box covers a large enough share of the frame), the best face chip is extracted, lighting-normalized, and embedded with ArcFace. Cosine similarity against the friend embeddings decides FRIEND vs FOE.
4. **Confirm** — similarity is smoothed over recent frames, a label must hold for consecutive frames, and a FOE must remain close for a few seconds before it counts.
5. **Alert** — on a confirmed intruder, a snapshot and face montage are saved and an alert is sent (sound popup or email).
6. **Log** — every detection is appended to `detections_log.csv`.

---

## Configuration reference

| Variable | Default | Meaning |
|----------|---------|---------|
| `CAMERA_URL` | `0` | Camera source: device index or stream URL. |
| `LOCAL_MODEL` | `yolov8n.pt` | YOLO weights file. |
| `FRIEND_DIR` | `frnd_imgs` | Folder containing friend reference images. |
| `FRIEND_IMGS` | (list) | Specific files, or `*` to load the whole folder. |
| `CONF_PERSON_CONF` | `0.30` | Minimum YOLO confidence to count as a person. |
| `CLOSE_AREA_RATIO` | `0.03` | How much of the frame a person must fill before face classification runs. |
| `FRIEND_THRESHOLD` | `0.42` | Cosine similarity above which a face counts as FRIEND. |
| `CONFIRMATION_TIME` | `2.0` | Seconds a FOE must stay close before alerting. |
| `ALERT_COOLDOWN` | `45.0` | Minimum seconds between alerts. |

---

## Limitations

- Face recognition accuracy depends heavily on lighting and the quality of your reference images.
- CPU-only inference is fine for testing but limited in frame rate; a CUDA GPU is much faster.
- This is a hobby/learning project and is **not** a substitute for a professionally installed security system.

---

## License

This project is released under the [MIT License](LICENSE) — you are free to use, modify, and distribute it, provided the original copyright notice is kept.

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection.
- [DeepFace](https://github.com/serengil/deepface) for ArcFace face recognition.
- [OpenCV](https://opencv.org/) for video capture and image processing.
