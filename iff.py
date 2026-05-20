# ==================== BorderGuard v14 — LIVE Border Surveillance (Mac-friendly) ====================
# Low-latency live pipeline with Android IP Webcam (MJPEG + JPEG fallback), latest-frame capture,
# optional animal suppression, optional pose confirmation, region masking, snapshots + CSV logs.
# -----------------------------------------------------------------------------------------------
# Requirements:
#   pip install ultralytics opencv-python numpy requests torch
# (use opencv-python-headless if you want headless; add torchvision/torchaudio if needed)
#
# Notes:
# - Works on CPU; uses FP16 if CUDA available. macOS often runs CPU-only; that's fine for testing.
# - Designed for "miss-nothing" recall with a quick visual confirmation path.
# - On networks where MJPEG is flaky, automatic /shot.jpg polling fallback is extremely reliable.
# - Set IP_CAM_BASE to your phone’s URL ROOT shown by the IP Webcam app (same Wi-Fi or USB tether).
# -----------------------------------------------------------------------------------------------

import os, time, threading, queue, csv
from pathlib import Path
import numpy as np
import cv2
import urllib.parse, requests

# Ultralytics (load after env ready)
from ultralytics import YOLO

# Torch (optional CUDA/FP16)
try:
    import torch
    CUDA_OK = torch.cuda.is_available()
except Exception:
    CUDA_OK = False

# ---------------- CONFIG (EDIT THESE) ----------------
# 1) Android IP Webcam base URL (what the app shows on phone screen, e.g., http://192.168.1.50:8080)
IP_CAM_BASE           = "http://192.168.95.27:8080/"   # <--- CHANGE THIS
IP_USER               = None                         # e.g. "user" or None if auth disabled in app
IP_PASS               = None                         # e.g. "pass" or None

# Endpoints to try for MJPEG streaming (in order)
ENDPOINTS             = ["/video", "/stream.mjpeg"]
# JPEG polling fallback (very reliable on macOS)
JPEG_PATH             = "/shot.jpg"
USE_JPEG_FALLBACK     = True
JPEG_POLL_FPS         = 15                           # 10–15 is a good low-latency sweet spot

# 2) Models (use TensorRT only on supported Linux/NVIDIA setups)
USE_TRT               = False
MODEL_YOLO            = "yolov8s.pt"                # try 'yolov8n.pt' for max FPS
MODEL_YOLO_TRT        = "yolov8s.engine"            # if USE_TRT=True and you exported it
MODEL_POSE            = "yolov8n-pose.pt"           # for ambiguous confirmations

# 3) Sizes & thresholds
IMGSZ                 = 640
CONF_PERSON           = 0.25
NMS_IOU               = 0.7
MAX_DET               = 300
POSE_ENABLED          = True
POSE_CONF_CHECK       = (0.15, 0.35)                 # confirm via pose in this band

# 4) Animal suppression (border-specific)
ANIMAL_HARD_REJECT    = True
ANIMAL_CLASS_IDS      = [15,16,17,18,19,20,21,22,23] # COCO: cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe
ANIMAL_CONF           = 0.30
ANIMAL_EVERY_N        = 5
ANIMAL_IOU            = 0.25

# 5) Region mask (geofence) — list of polygons in relative coords (0..1). Empty = whole frame.
REGION_POLYGONS       = [
    # Example trapezoid covering lower half:
    # [(0.05,0.55), (0.95,0.55), (0.95,0.98), (0.05,0.98)]
]

# 6) Motion/light gating
BRIGHT_MEAN_MIN       = 35
BRIGHT_STD_MIN        = 18

# 7) Runtime & I/O
PRINT_EVERY_N         = 30
SNAP_DIR              = "snapshots"
CSV_LOG               = "detections_live_log.csv"
DRAW_LABELS           = True
DISPLAY_WINDOW        = True                      # set False on mac headless / SSH runs

# Optional: write output video (off by default)
WRITE_VIDEO           = False
OUTPUT_VIDEO          = "borderguard_live_output.avi"
OUTPUT_FPS            = 20

# ---------------- UTILITIES ----------------
Path(SNAP_DIR).mkdir(exist_ok=True)
frame_q = queue.Queue(maxsize=1)
stop_flag = False

def now_ts():
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())

def iou_xyxy(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0: return 0.0
    areaA = max(0, (a[2]-a[0])) * max(0, (a[3]-a[1]))
    areaB = max(0, (b[2]-b[0])) * max(0, (b[3]-b[1]))
    return inter / max(1e-9, (areaA + areaB - inter))

def center_xyxy(box):
    return ((box[0]+box[2])/2.0, (box[1]+box[3])/2.0)

def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        if ((y1 > y) != (y2 > y)):
            xinters = x1 + (y - y1) * (x2 - x1) / max(1e-9, (y2 - y1))
            if x < xinters:
                inside = not inside
    return inside

def in_any_region(cx, cy, W, H):
    if not REGION_POLYGONS: return True
    rx, ry = cx / max(1, W), cy / max(1, H)
    for poly in REGION_POLYGONS:
        if point_in_poly(rx, ry, poly): return True
    return False

# ---------------- CAMERA HELPERS (Android IP Webcam) ----------------
def build_url(path):
    base = IP_CAM_BASE.rstrip("/")
    if IP_USER and IP_PASS:
        u = urllib.parse.urlparse(base)
        host = u.hostname
        port = u.port or (443 if u.scheme == "https" else 80)
        netloc = f"{IP_USER}:{IP_PASS}@{host}:{port}"
        return urllib.parse.urlunparse((u.scheme, netloc, path, "", "", ""))
    return base + path

def try_open_with_backends(urls):
    # Try CAP_ANY then CAP_FFMPEG; add dummy mjpg param (helps some OpenCV builds)
    candidates = []
    for u in urls:
        candidates.append(u)
        candidates.append(u + ("&dummy=1.mjpg" if "?" in u else "?dummy=1.mjpg"))

    for u in candidates:
        for backend in (cv2.CAP_ANY, cv2.CAP_FFMPEG):
            cap = cv2.VideoCapture(u, backend)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if cap.isOpened():
                print(f"[IPCam] Opened {u} with backend={backend}")
                return cap
            cap.release()
    return None

def open_mjpeg_or_fallback():
    urls = [build_url(ep) for ep in ENDPOINTS]
    cap = try_open_with_backends(urls)
    if cap is not None:
        return ("mjpeg", cap)
    if USE_JPEG_FALLBACK:
        print("[IPCam] MJPEG failed → using JPEG polling fallback")
        sess = requests.Session()
        return ("jpeg", (sess, build_url(JPEG_PATH)))
    return (None, None)

def capture_loop(_unused):
    global stop_flag, frame_q
    mode, handle = open_mjpeg_or_fallback()
    if mode is None:
        raise RuntimeError("Cannot open IP Webcam stream. Check IP_CAM_BASE/auth/network.")

    if mode == "mjpeg":
        cap = handle
        while not stop_flag:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02); continue
            if not frame_q.empty():
                try: frame_q.get_nowait()
                except queue.Empty: pass
            frame_q.put(frame)
        cap.release()
        return

    # JPEG polling fallback (rock solid on macOS)
    sess, url = handle
    period = 1.0 / max(1, JPEG_POLL_FPS)
    while not stop_flag:
        t0 = time.time()
        try:
            r = sess.get(url, timeout=2)
            if r.status_code == 200:
                arr = np.frombuffer(r.content, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    if not frame_q.empty():
                        try: frame_q.get_nowait()
                        except queue.Empty: pass
                    frame_q.put(frame)
        except Exception:
            time.sleep(0.1)
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)

# ---------------- MODELS ----------------
def load_model():
    if USE_TRT and os.path.exists(MODEL_YOLO_TRT):
        m = YOLO(MODEL_YOLO_TRT)
    else:
        m = YOLO(MODEL_YOLO)
    try: m.fuse()
    except Exception: pass
    return m

def load_pose_model():
    try:
        pm = YOLO(MODEL_POSE)
        pm.fuse()
        return pm
    except Exception:
        return None

# ---------------- INFERENCE HELPERS ----------------
def run_person(model, img, device, half):
    r = model.predict(img, classes=[0], conf=CONF_PERSON, imgsz=IMGSZ,
                      device=device, half=half, iou=NMS_IOU, max_det=MAX_DET,
                      verbose=False)[0]
    if not hasattr(r, 'boxes') or r.boxes is None:
        return np.zeros((0,4)), np.zeros((0,))
    return r.boxes.xyxy.detach().cpu().numpy(), r.boxes.conf.detach().cpu().numpy()

def run_animals(model, img, device, half):
    r = model.predict(img, classes=ANIMAL_CLASS_IDS, conf=ANIMAL_CONF, imgsz=IMGSZ,
                      device=device, half=half, iou=0.5, max_det=MAX_DET,
                      verbose=False)[0]
    if not hasattr(r, 'boxes') or r.boxes is None:
        return np.zeros((0,4), dtype=np.float32)
    return r.boxes.xyxy.detach().cpu().numpy()

def pose_confirms(pose_model, crop, device, half):
    if pose_model is None:
        return True
    r = pose_model.predict(crop, imgsz=384, conf=0.25, device=device, half=half, verbose=False)[0]
    return (hasattr(r, 'keypoints') and r.keypoints is not None and len(r.keypoints) > 0)

# ---------------- MAIN ----------------
def main():
    global stop_flag
    device = 0 if CUDA_OK else None
    half = bool(CUDA_OK)

    model = load_model()
    pose_model = load_pose_model() if POSE_ENABLED else None

    # Warmup (builds kernels)
    warm_shape = (IMGSZ, IMGSZ, 3)
    try:
        _ = model.predict(np.zeros(warm_shape, dtype=np.uint8), imgsz=IMGSZ,
                          device=device, half=half, verbose=False)
        if pose_model is not None:
            _ = pose_model.predict(np.zeros((384,384,3), dtype=np.uint8), imgsz=384,
                                   device=device, half=half, verbose=False)
        if CUDA_OK:
            torch.backends.cudnn.benchmark = True
    except Exception as e:
        print(f"Warmup warning: {e}")

    # Start capture
    t = threading.Thread(target=capture_loop, args=(None,), daemon=True)
    t.start()

    writer = None
    W = H = None

    # CSV header
    with open(CSV_LOG, "w", newline="") as f:
        csv.writer(f).writerow(["ts_utc", "frame_id", "humans", "snap_path"])

    print("\n🎥 BorderGuard v14 — LIVE started (Android IP Webcam). Press 'q' to quit.\n")

    frame_id = 0
    t0 = time.time()

    try:
        while True:
            try:
                frame = frame_q.get(timeout=2.0)
            except queue.Empty:
                continue

            frame_id += 1
            H, W = frame.shape[:2]

            # simple light/variance gate
            m, s = frame.mean(), frame.std()
            if m < BRIGHT_MEAN_MIN or s < BRIGHT_STD_MIN:
                if DISPLAY_WINDOW:
                    cv2.putText(frame, "LOW LIGHT / LOW VARIANCE - SKIP", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    cv2.imshow("BorderGuard LIVE", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            # scale to IMGSZ for detection (keep aspect)
            max_side = max(H, W)
            scale = min(IMGSZ / max_side, 1.0)
            det_in = cv2.resize(frame, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_LINEAR) if scale < 1.0 else frame

            p_boxes, p_scores = run_person(model, det_in, device, half)
            if scale < 1.0 and p_boxes.shape[0]:
                p_boxes = p_boxes / np.array([scale, scale, scale, scale], dtype=np.float32)

            humans = []
            results = []

            # optional animal rejection every N frames
            a_boxes = np.zeros((0,4), dtype=np.float32)
            if ANIMAL_HARD_REJECT and (frame_id % ANIMAL_EVERY_N == 0) and p_boxes.shape[0] > 0:
                a_boxes = run_animals(model, det_in, device, half)
                if scale < 1.0 and a_boxes.shape[0]:
                    a_boxes = a_boxes / np.array([scale, scale, scale, scale], dtype=np.float32)

            for (box, conf) in zip(p_boxes, p_scores):
                if conf < 0.10:
                    continue
                x1,y1,x2,y2 = [int(v) for v in box]
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
                if x2 <= x1 or y2 <= y1: continue

                cx, cy = center_xyxy((x1,y1,x2,y2))
                if not in_any_region(cx, cy, W, H): continue

                # pose confirm if ambiguous
                if POSE_ENABLED and (POSE_CONF_CHECK[0] <= conf <= POSE_CONF_CHECK[1]):
                    crop = frame[y1:y2, x1:x2]
                    try:
                        if not pose_confirms(pose_model, crop, device, half):
                            continue
                    except Exception:
                        pass

                # animal overlap reject
                rejected = False
                if a_boxes.shape[0] > 0:
                    for ab in a_boxes:
                        if iou_xyxy(ab, (x1,y1,x2,y2)) >= ANIMAL_IOU:
                            rejected = True; break
                if rejected: continue

                humans.append((x1,y1,x2,y2))
                results.append(((x1,y1,x2,y2), float(conf)))

            # Draw HUD
            if DRAW_LABELS:
                for (x1,y1,x2,y2), conf in results:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, f"HUMAN {conf:.2f}", (x1, max(18, y1-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # Draw geofences
                for poly in REGION_POLYGONS:
                    pts = np.array([[int(px*W), int(py*H)] for (px,py) in poly], dtype=np.int32)
                    cv2.polylines(frame, [pts], isClosed=True, color=(255,255,0), thickness=2)

                cv2.rectangle(frame, (0,0), (280,48), (0,0,0), -1)
                cv2.putText(frame, f"Humans: {len(humans)}", (10,32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            # Snapshots & logs on positive
            snap_path = ""
            if len(humans) > 0:
                snap_name = f"snap_{now_ts()}_{frame_id:06d}.jpg"
                snap_path = str(Path(SNAP_DIR) / snap_name)
                try:
                    cv2.imwrite(snap_path, frame)
                except Exception:
                    snap_path = ""
                with open(CSV_LOG, "a", newline="") as f:
                    csv.writer(f).writerow([now_ts(), frame_id, len(humans), snap_path])

            # Optional writer
            if WRITE_VIDEO:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, OUTPUT_FPS, (W,H))
                writer.write(frame)

            # Display
            if DISPLAY_WINDOW:
                cv2.imshow("BorderGuard LIVE", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_id % PRINT_EVERY_N == 0:
                fps_now = frame_id / max(1e-6, (time.time() - t0))
                print(f"Processed {frame_id} frames | ~{fps_now:.1f} FPS | Humans={len(humans)}")

    finally:
        stop_flag = True
        if DISPLAY_WINDOW:
            try: cv2.destroyAllWindows()
            except Exception: pass
        if 'writer' in locals() and writer is not None:
            writer.release()
        print("\n✅ BorderGuard v14 — LIVE stopped. Logs saved to:", CSV_LOG)

if __name__ == "__main__":
    # Sanity: warn if base URL looks untouched
    if IP_CAM_BASE.startswith("http://192.168.1.50"):
        print("[Hint] Set IP_CAM_BASE to the exact URL shown in the IP Webcam app (phone & Mac on same network).")
    main()
