import os
import time
import csv
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------- CONFIG ----------------
CAMERA_URL = os.getenv("CAMERA_URL", "0")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "yolov8n.pt")
FRIEND_IMGS_RAW = os.getenv("FRIEND_IMGS", "frnd_imgs")
FRIEND_DIR = os.getenv("FRIEND_DIR", "frnd_imgs").strip()

SNAP_DIR = "snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)

# Detection params
CONF_PERSON_CONF   = float(os.getenv("CONF_PERSON_CONF", 0.30))
CLOSE_AREA_RATIO   = float(os.getenv("CLOSE_AREA_RATIO", 0.03))
FAR_AREA_RATIO     = float(os.getenv("FAR_AREA_RATIO", 0.02))
FRIEND_THRESHOLD   = float(os.getenv("FRIEND_THRESHOLD", 0.42))
CONFIRMATION_TIME  = float(os.getenv("CONFIRMATION_TIME", 2.0))
ALERT_COOLDOWN     = float(os.getenv("ALERT_COOLDOWN", 45.0))
CONFIRM_FRAMES     = int(os.getenv("CONFIRM_FRAMES", 3))
MAX_MISSING_FRAMES = int(os.getenv("MAX_MISSING_FRAMES", 10))
MATCH_CENTER_DIST  = int(os.getenv("MATCH_CENTER_DIST", 90))
FACE_MODEL         = os.getenv("FACE_MODEL", "ArcFace")
CSV_LOG = "detections_log.csv"

# Parse FRIEND_IMGS env
if FRIEND_IMGS_RAW.strip() == "":
    FRIEND_IMGS_LIST = []
elif FRIEND_IMGS_RAW.strip() == "*":
    FRIEND_IMGS_LIST = ["*"]
elif os.path.isdir(FRIEND_IMGS_RAW) or FRIEND_IMGS_RAW.lower().endswith((".jpg", ".jpeg", ".png")):
    FRIEND_IMGS_LIST = [FRIEND_IMGS_RAW]
else:
    FRIEND_IMGS_LIST = [p.strip() for p in FRIEND_IMGS_RAW.split(",") if p.strip()]

print(f"[CONFIG] CAMERA_URL={CAMERA_URL}  LOCAL_MODEL={LOCAL_MODEL}  FRIEND_DIR={FRIEND_DIR}")

DEEPFACE_OK = False
try:
    from deepface import DeepFace
    DEEPFACE_OK = True
    print("[INFO] DeepFace imported.")
except Exception as e:
    print("[WARN] DeepFace not available:", e)
    DEEPFACE_OK = False

# ---------------- UTILITIES ----------------
def now_str():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())

def ensure_csv_header(path):
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["time","track_id","label","similarity","x1","y1","x2","y2","area_ratio"])
    return f, w

# ---------------- Local alert helpers ----------------
def local_play_sound():
    """Best-effort short alert sound. macOS uses afplay; fallback to terminal bell."""
    try:
        if os.name == "posix" and "darwin" in os.uname().sysname.lower():
            sound = "/System/Library/Sounds/Glass.aiff"
            if os.path.exists(sound):
                os.system(f"afplay '{sound}' &")
                return True
        # fallback bell
        print("\a", end="", flush=True)
        return True
    except Exception:
        return False

def open_image_default(path):
    """Open image file with default system viewer."""
    try:
        if os.name == "posix" and "darwin" in os.uname().sysname.lower():
            os.system(f"open '{path}' &")
        elif os.name == "posix":
            os.system(f"xdg-open '{path}' &")
        else:
            # Windows
            os.system(f"start \"\" \"{path}\"")
    except Exception:
        pass

def local_alert(full_snap_path, montage_path=None, text=None):
    """Local-only alert: play sound and open saved images."""
    print("[ALERT] Local alert:", text or "Intruder detected")
    local_play_sound()
    if montage_path and os.path.exists(montage_path):
        open_image_default(montage_path)
    elif full_snap_path and os.path.exists(full_snap_path):
        open_image_default(full_snap_path)

# ---------------- FACE HELPERS ----------------
def extract_best_face_chip(img_bgr, detector="retinaface", align=True):
    try:
        faces = DeepFace.extract_faces(img_path=img_bgr, align=align, detector_backend=detector, enforce_detection=False)
        if not faces:
            return None
        best = max(
            faces,
            key=lambda f: f.get("facial_area", {}).get("w",0) * f.get("facial_area", {}).get("h",0)
        )
        chip_rgb = best["face"]
        chip_rgb = (np.clip(chip_rgb, 0, 1) * 255).astype(np.uint8)
        return chip_rgb
    except Exception:
        return None

def enhance_face_lighting(face_rgb):
    try:
        lab = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    except Exception:
        return face_rgb

def _embed_face_rgb(face_rgb):
    rep = DeepFace.represent(img_path=face_rgb, model_name=FACE_MODEL, detector_backend="skip", enforce_detection=False)
    emb = np.array(rep[0]["embedding"], dtype=np.float32)
    emb /= (np.linalg.norm(emb) + 1e-9)
    return emb

# ---------------- FRIEND PATH RESOLUTION ----------------
def _resolve_friend_paths(friend_imgs_list, friend_dir):
    import glob
    out = []
    def add_dir(d):
        patterns = ["*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"]
        for pat in patterns:
            out.extend(sorted(glob.glob(os.path.join(d, pat))))
    if len(friend_imgs_list) == 1 and friend_imgs_list[0] == "*":
        add_dir(friend_dir)
        return out
    for p in friend_imgs_list:
        p = p.strip()
        if not p:
            continue
        if os.path.isabs(p):
            if os.path.isdir(p): add_dir(p)
            elif os.path.isfile(p): out.append(p)
            else: print(f"[WARN] Friend path not found: {p}")
            continue
        if os.path.isdir(p):
            add_dir(p); continue
        if os.path.isfile(p):
            out.append(p); continue
        cand = os.path.join(friend_dir, p)
        if os.path.isdir(cand):
            add_dir(cand); continue
        if os.path.isfile(cand):
            out.append(cand); continue
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cand2 = os.path.join(base_dir, friend_dir, p)
        if os.path.isdir(cand2):
            add_dir(cand2); continue
        if os.path.isfile(cand2):
            out.append(cand2); continue
        print(f"[WARN] Friend image not found (skipped): {p}")
    return out

FRIEND_IMGS = _resolve_friend_paths(FRIEND_IMGS_LIST, FRIEND_DIR)
print(f"[STATUS] Looking for friend images in: {FRIEND_DIR}")
print(f"[STATUS] Resolved {len(FRIEND_IMGS)} friend images.")

# ---------------- FRIEND EMBEDDINGS ----------------
friend_embs = []
friend_names = []

def _avg_augmented_embedding_from_path(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError(f"Cannot read {img_path}")
    chip = extract_best_face_chip(img_bgr, detector="retinaface")
    if chip is None:
        raise ValueError(f"No face found in {img_path}")
    chip = enhance_face_lighting(chip)
    e1 = _embed_face_rgb(chip)
    e2 = _embed_face_rgb(cv2.flip(chip, 1))
    avg = (e1 + e2) / 2.0
    avg /= (np.linalg.norm(avg) + 1e-9)
    return avg

def _name_from_path(p):
    base = os.path.basename(p)
    root, _ = os.path.splitext(base)
    return root.split("_")[0].capitalize()

if DEEPFACE_OK:
    if not FRIEND_IMGS:
        print("[WARN] No FRIEND_IMGS resolved. Matching will be OFF.")
    for fp in FRIEND_IMGS:
        try:
            print(f"🔄 Encoding friend: {fp}")
            emb = _avg_augmented_embedding_from_path(fp)
            friend_embs.append(emb)
            friend_names.append(_name_from_path(fp))
            print(f"✅ Encoded: {fp}")
        except Exception as e:
            print(f"[WARN] Failed to encode {fp}: {e}")
else:
    print("[WARN] DeepFace unavailable; friend matching disabled.")

if len(friend_embs) == 0:
    DEEPFACE_OK = False
    print("[WARN] No friend embeddings loaded. Friend matching OFF.")
else:
    print(f"[STATUS] DeepFace OK: {DEEPFACE_OK} | friend_embs: {len(friend_embs)}")

def load_yolo():
    print("🔄 Loading YOLO model...")
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(f"Failed to import ultralytics: {e}")
    if not os.path.exists(LOCAL_MODEL):
        raise FileNotFoundError(f"Model file not found: {LOCAL_MODEL} (cwd={os.getcwd()})")
    m = YOLO(LOCAL_MODEL)
    print("✅ YOLO ready.")
    return m

# ---------------- HELPERS ----------------
def area_ratio(x1,y1,x2,y2,W,H):
    return max(0, (x2-x1)*(y2-y1)) / (W*H + 1e-9)

def cosine_sim(u,v):
    return float(np.dot(u,v))

def classify_close_crop(person_crop):
    if not DEEPFACE_OK or person_crop is None or person_crop.size == 0:
        return "UNKNOWN", 0.0, -1
    try:
        chip = extract_best_face_chip(person_crop, detector="retinaface")
        if chip is None or chip.size == 0:
            return "UNKNOWN", 0.0, -1
        h, w = chip.shape[:2]
        if min(h, w) < 96:
            return "UNKNOWN", 0.0, -1
        chip = enhance_face_lighting(chip)
        emb  = _embed_face_rgb(chip)
        if not friend_embs:
            return "UNKNOWN", 0.0, -1
        sims = [cosine_sim(emb, f) for f in friend_embs]
        best_idx = int(np.argmax(sims))
        max_sim = float(sims[best_idx])
        label = "FRIEND" if max_sim >= FRIEND_THRESHOLD else "FOE"
        return label, max_sim, best_idx
    except Exception:
        return "UNKNOWN", 0.0, -1

def save_foe_montage(crops, out_path):
    try:
        crops = [c for c in crops if c is not None and c.size > 0]
        if not crops:
            return None
        crops = [cv2.resize(c, (128,128)) for c in crops[:12]]
        row_len = min(6, len(crops))
        rows = []
        for i in range(0, len(crops), row_len):
            row = crops[i:i+row_len]
            while len(row) < row_len:
                row.append(np.zeros_like(crops[0]))
            rows.append(cv2.hconcat(row))
        canvas = rows[0] if len(rows)==1 else cv2.vconcat(rows)
        cv2.imwrite(out_path, canvas)
        return out_path
    except Exception as e:
        print("[WARN] Montage failed:", e)
        return None

def center_of_box(x1,y1,x2,y2):
    return int((x1+x2)//2), int((y1+y2)//2)

# ---------------- TRACKER ----------------
NEXT_TRACK_ID = 1
tracks = {}

def match_track(center):
    best_id, best_d = None, 1e9
    for tid, t in tracks.items():
        cx, cy = t['center']
        d = np.hypot(center[0]-cx, center[1]-cy)
        if d < best_d and d <= MATCH_CENTER_DIST:
            best_id, best_d = tid, d
    return best_id

# ---------------- MAIN LOOP ----------------
def run_camera():
    global NEXT_TRACK_ID, tracks

    # Load model once
    try:
        model = load_yolo()
    except Exception as e:
        print("[FATAL] YOLO load failed:", e)
        print("-> Install ultralytics and place model at:", LOCAL_MODEL)
        return

    # Choose device
    try:
        import torch
        DEVICE = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        DEVICE = "cpu"
    USE_HALF = (DEVICE != "cpu")
    print("[INFO] Inference device:", DEVICE, " use_half:", USE_HALF)

    csv_file, csv_writer = ensure_csv_header(CSV_LOG)

    # Normalize src
    src = CAMERA_URL
    try:
        if str(src).strip().isdigit():
            src = int(str(src).strip())
    except Exception:
        pass

    # Open camera (mac-friendly)
    def try_open(attempt):
        try:
            uname = os.uname().sysname.lower()
        except Exception:
            uname = ""
        backends = [cv2.CAP_ANY]
        if "darwin" in uname:
            backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_QT, cv2.CAP_ANY]
        for b in backends:
            try:
                c = cv2.VideoCapture(attempt, b) if isinstance(attempt, int) else cv2.VideoCapture(attempt)
                time.sleep(0.25)
                if c.isOpened():
                    print(f"[OK] Opened source {attempt} with backend {b}")
                    return c
                else:
                    try: c.release()
                    except: pass
            except Exception as e:
                print(f"[WARN] open {attempt} backend {b} error: {e}")
        return None

    cap = None
    candidates = [src] if not isinstance(src, int) else [src, 0, 1, 2, 3]
    for attempt in candidates:
        print(f"[INFO] Attempting to open camera source: {attempt}")
        cap = try_open(attempt)
        if cap is not None:
            break
    if cap is None or not cap.isOpened():
        print("[ERROR] Could not open any camera source. Check CAMERA_URL and permissions.")
        return

    print("🎥 Connected!")
    # Set lower capture resolution for improved throughput (still processing every frame)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        print("[INFO] Requested camera resolution 640x360")
    except Exception:
        pass

    # Quick GUI test
    ok, test_frame = cap.read()
    if not ok or test_frame is None:
        print("[ERROR] Camera opened but failed to read frame.")
        try: cap.release()
        except: pass
        return
    print("[INFO] Camera read test frame OK — starting loop.")

    last_alert_time = 0
    system_status = "🟢 Secure"

    # main loop - process EVERY frame (no skipping)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] Frame read failed. Attempting to reopen.")
            try: cap.release()
            except: pass
            cap = None
            # try reopen
            for attempt in candidates:
                cap = try_open(attempt)
                if cap is not None: break
            if cap is None:
                print("[FATAL] Reopen failed — exiting.")
                break
            continue

        H, W = frame.shape[:2]

        # Perform inference on a resized copy to reduce model workload, while treating each frame.
        # Resize so longer side == 640 (or smaller if frame already small). Preserve aspect ratio.
        long_side = max(H, W)
        target_long = 640
        if long_side > target_long:
            scale = target_long / float(long_side)
            small_w = max(1, int(W * scale))
            small_h = max(1, int(H * scale))
            small_frame = cv2.resize(frame, (small_w, small_h))
        else:
            small_frame = frame

        # Run prediction on the smaller frame — we still process every frame, but model work is reduced
        try:
            # ultralytics predict accepts numpy BGR frames; pass imgsz=target_long for consistent scaling
            results = model.predict(small_frame, conf=CONF_PERSON_CONF, device=DEVICE, imgsz=640, half=USE_HALF, verbose=False)
            r = results[0]
            boxes_raw = r.boxes.xyxy.cpu().numpy() if getattr(r, 'boxes', None) is not None else np.zeros((0,4))
        except Exception as e:
            print("[ERROR] Model prediction failed:", e)
            # avoid crashing — continue to next frame
            time.sleep(0.01)
            continue

        # Map boxes back to original frame coordinates
        boxes = []
        sh, sw = small_frame.shape[:2]
        scale_x = W / float(sw)
        scale_y = H / float(sh)
        for i in range(len(boxes_raw)):
            x1, y1, x2, y2 = boxes_raw[i]
            x1 = int(max(0, x1 * scale_x)); y1 = int(max(0, y1 * scale_y))
            x2 = int(min(W-1, x2 * scale_x)); y2 = int(min(H-1, y2 * scale_y))
            boxes.append([x1, y1, x2, y2])
        boxes = np.array(boxes) if boxes else np.zeros((0,4))

        any_confirmed_foe_close = False
        foe_crops = []

        # age tracks
        for t in tracks.values():
            t['missing'] = t.get('missing', 0) + 1

        # handle detections
        for i in range(len(boxes)):
            x1,y1,x2,y2 = boxes[i].astype(int)
            x1,y1 = max(0,x1), max(0,y1)
            x2,y2 = min(W-1,x2), min(H-1,y2)
            ar = area_ratio(x1,y1,x2,y2,W,H)
            cx, cy = center_of_box(x1,y1,x2,y2)

            tid = match_track((cx,cy))
            if tid is None:
                tid = NEXT_TRACK_ID; NEXT_TRACK_ID += 1
                tracks[tid] = {'bbox':(x1,y1,x2,y2),'center':(cx,cy),
                               'seen':1,'missing':0,'label':'HUMAN','sim':0.0,
                               'close_since':None,'sim_hist':[]}
            else:
                t = tracks[tid]
                t['bbox']=(x1,y1,x2,y2); t['center']=(cx,cy)
                t['seen']=t.get('seen',0)+1; t['missing']=0

            t = tracks[tid]

            # default HUMAN
            label, sim = "HUMAN", 0.0
            color = (255,255,0)
            best_idx_for_box = -1

            # if close -> run FRIEND/FOE
            if ar >= CLOSE_AREA_RATIO:
                crop = frame[y1:y2, x1:x2]
                label, sim, best_idx_for_box = classify_close_crop(crop)
                if label == "FRIEND":
                    color = (0,255,0)
                    t['close_since'] = None
                elif label == "FOE":
                    color = (0,0,255)
                    if t['close_since'] is None:
                        t['close_since'] = time.time()
                    foe_crops.append(crop.copy())
                else:
                    # UNKNOWN -> treat as HUMAN (don’t escalate)
                    label = "HUMAN"
                    color = (255,255,0)
                    t['close_since'] = None
            else:
                label = "HUMAN"
                t['close_since'] = None

            # smoothing history
            if label in ("FRIEND","FOE"):
                hist = t.get('sim_hist', [])
                hist.append(sim)
                if len(hist) > 8: hist.pop(0)
                t['sim_hist'] = hist
                smoothed = float(np.mean(hist[-5:])) if hist else sim
                label = "FRIEND" if smoothed >= FRIEND_THRESHOLD else "FOE"
                sim = smoothed

            # momentum: require consecutive frames for FRIEND/FOE
            t['friend_streak'] = 1 + t.get('friend_streak', 0) if label == "FRIEND" else 0
            t['foe_streak']    = 1 + t.get('foe_streak', 0)    if label == "FOE"    else 0

            COMMIT_N = 2
            committed_label = label
            if label == "FRIEND" and t['friend_streak'] < COMMIT_N:
                committed_label = "HUMAN"
            if label == "FOE" and t['foe_streak'] < COMMIT_N:
                committed_label = "HUMAN"

            t['label'] = committed_label
            t['sim'] = sim
            confirmed = t.get('seen',0) >= CONFIRM_FRAMES

            # Draw tag (show name only for FRIEND)
            name_suffix = ""
            if committed_label == "FRIEND" and best_idx_for_box is not None and 0 <= best_idx_for_box < len(friend_names):
                name_suffix = f":{friend_names[best_idx_for_box]}"
            tag = (committed_label if confirmed else f"maybe:{committed_label}") + name_suffix + (f" {sim:.2f}" if committed_label!="HUMAN" else "")
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, tag, (x1, max(20,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # log
            csv_writer.writerow([time.ctime(), tid, committed_label, f"{sim:.3f}", x1,y1,x2,y2, f"{ar:.5f}"])
            csv_file.flush()

            # FOE confirmation timing
            if confirmed and committed_label=="FOE" and t['close_since'] and (time.time() - t['close_since'] >= CONFIRMATION_TIME):
                any_confirmed_foe_close = True

        # cleanup dead tracks
        dead = [tid for tid,t in tracks.items() if t.get('missing',0) > MAX_MISSING_FRAMES]
        for tid in dead: del tracks[tid]

        # alerts
        if any_confirmed_foe_close:
            system_status = "🔴 Intruder Detected"
            if time.time() - last_alert_time >= ALERT_COOLDOWN:
                full_snap = os.path.join(SNAP_DIR, f"foe_full_{now_str()}.jpg")
                cv2.imwrite(full_snap, frame)
                montage_path = os.path.join(SNAP_DIR, f"foe_faces_{now_str()}.jpg")
                montage = save_foe_montage(foe_crops, montage_path)
                # Local-only alert (no email)
                local_alert(full_snap, montage, text=f"Unknown person(s) detected at {time.ctime()}")
                last_alert_time = time.time()
        else:
            system_status = "🟢 Secure"

        # HUD
        cv2.rectangle(frame, (0,0), (420,40), (0,0,0), -1)
        cv2.putText(frame, system_status, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0) if "Secure" in system_status else (0,0,255), 2)

        # Display
        cv2.imshow("BorderGuard", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(os.path.join(SNAP_DIR, f"manual_{now_str()}.jpg"), frame)

    csv_file.close(); cap.release(); cv2.destroyAllWindows()
    print("[INFO] Exiting.")

if __name__ == "__main__":
    run_camera()