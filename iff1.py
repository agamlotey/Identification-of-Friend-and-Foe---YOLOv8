# borderguard_final.py
# BorderGuard — secure SMTP via .env, multi-person detection, FRIEND/FOE/HUMAN labels,
# FOE confirmation with full-frame + face-montage email
# (Agam / Dev / Furmaan presets + ArcFace face-chips + smoothing)

import os, time, ssl, csv, cv2, numpy as np
from email.message import EmailMessage
from ultralytics import YOLO
from dotenv import load_dotenv

# ---------------- LOAD CONFIG FROM .env ----------------
load_dotenv()  # reads .env in working directory

# SMTP / Email config
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))
SMTP_SSL  = os.getenv("SMTP_SSL", "True").lower() in ("true", "1", "yes")
SMTP_USER = os.getenv("SMTP_USER")            # e.g. your_gmail@gmail.com
SMTP_PASS = (os.getenv("SMTP_PASS") or "").replace(" ", "")  # strip spaces if pasted
TO_EMAIL  = os.getenv("TO_EMAIL")

# App-specific config
CAMERA_URL = os.getenv("CAMERA_URL", "http://10.39.95.133:8080/video")

# Default friend images list (can be overridden via .env).
# Tip: set FRIEND_IMGS="*" in .env to load all jpg/pngs from FRIEND_DIR.
FRIEND_IMGS = [p.strip() for p in os.getenv(
    "FRIEND_IMGS",
    "agam_1.jpeg,agam_2.jpeg,agam_3.jpeg,dev_1.jpeg,dev_2.jpeg,dev_3.jpeg,furmaan_1.jpeg,furmaan_2.jpeg,furmaan_3.jpeg"
).split(",") if p.strip()]

# Directory containing friend images
FRIEND_DIR = os.getenv("FRIEND_DIR", "frnd_imgs").strip()

LOCAL_MODEL = os.getenv("LOCAL_MODEL", "yolov8n.pt")

SNAP_DIR = "snapshots"
os.makedirs(SNAP_DIR, exist_ok=True)

# Detection/behavior params
CONF_PERSON_CONF   = float(os.getenv("CONF_PERSON_CONF", 0.35))  # YOLO person conf
CLOSE_AREA_RATIO   = float(os.getenv("CLOSE_AREA_RATIO", 0.06))  # >= close -> classify FRIEND/FOE
FAR_AREA_RATIO     = float(os.getenv("FAR_AREA_RATIO", 0.03))    # < far -> HUMAN
FRIEND_THRESHOLD   = float(os.getenv("FRIEND_THRESHOLD", 0.55))  # cosine sim threshold (ArcFace)
CONFIRMATION_TIME  = float(os.getenv("CONFIRMATION_TIME", 2.0))  # seconds FOE must remain close
ALERT_COOLDOWN     = float(os.getenv("ALERT_COOLDOWN", 45.0))    # seconds between emails
CONFIRM_FRAMES     = int(os.getenv("CONFIRM_FRAMES", 3))         # frames to confirm a track
MAX_MISSING_FRAMES = int(os.getenv("MAX_MISSING_FRAMES", 10))    # drop track if missing too long
MATCH_CENTER_DIST  = int(os.getenv("MATCH_CENTER_DIST", 90))     # px center distance for track match

FACE_MODEL         = os.getenv("FACE_MODEL", "ArcFace")          # ArcFace > Facenet for ID

CSV_LOG = "detections_log.csv"

# ---------------- DEEPFACE IMPORT ----------------
try:
    from deepface import DeepFace
    DEEPFACE_OK = True
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

def send_email(subject, body, attachment_paths=None):
    """SMTP sender; supports SSL (465) or STARTTLS (587) based on .env."""
    if not (SMTP_USER and SMTP_PASS and TO_EMAIL):
        print("❌ Email credentials missing in .env. Skipping email.")
        return False

    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"] = TO_EMAIL
    msg["Subject"] = subject
    msg.set_content(body)

    if attachment_paths:
        for p in attachment_paths:
            if p and os.path.exists(p):
                with open(p, "rb") as f:
                    msg.add_attachment(f.read(), maintype="image", subtype="jpeg",
                                       filename=os.path.basename(p))
    try:
        import smtplib
        if SMTP_SSL:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context()) as s:
                s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
                s.ehlo()
                s.starttls(context=ssl.create_default_context())
                s.ehlo()
                s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
        print(f"📩 Email sent to {TO_EMAIL}")
        return True
    except Exception as e:
        print("❌ Email send failed:", e)
        return False

# ---------------- FACE HELPERS (chips + lighting + ArcFace) ----------------
def extract_best_face_chip(img_bgr, detector="retinaface", align=True):
    """
    Returns an aligned RGB face chip (np.uint8) or None.
    DeepFace.extract_faces returns list of dicts with 'face' in RGB [0..1].
    """
    try:
        faces = DeepFace.extract_faces(
            img_path=img_bgr, align=align, detector_backend=detector, enforce_detection=False
        )
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
    """Light normalization to reduce mismatch. RGB in, RGB out."""
    try:
        lab = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    except Exception:
        return face_rgb

def _embed_face_rgb(face_rgb):
    rep = DeepFace.represent(
        img_path=face_rgb, model_name=FACE_MODEL,
        detector_backend="skip", enforce_detection=False
    )
    emb = np.array(rep[0]["embedding"], dtype=np.float32)
    emb /= (np.linalg.norm(emb) + 1e-9)
    return emb

# ---------------- FRIEND PATH RESOLUTION (robust) ----------------
def _resolve_friend_paths(friend_imgs_list, friend_dir):
    import glob
    out = []

    def add_dir(d):
        patterns = ["*.jpg","*.jpeg","*.png","*.JPG","*.JPEG","*.PNG"]
        for pat in patterns:
            out.extend(sorted(glob.glob(os.path.join(d, pat))))

    # Case 1: wildcard = load everything in FRIEND_DIR
    if len(friend_imgs_list) == 1 and friend_imgs_list[0] == "*":
        add_dir(friend_dir)
        return out

    # Case 2: explicit items (file or folder, abs or relative)
    for p in friend_imgs_list:
        p = p.strip()
        if not p:
            continue

        # absolute path
        if os.path.isabs(p):
            if os.path.isdir(p): add_dir(p)
            elif os.path.isfile(p): out.append(p)
            else: print(f"[WARN] Friend path not found: {p}")
            continue

        # relative to CWD
        if os.path.isdir(p):
            add_dir(p); continue
        if os.path.isfile(p):
            out.append(p); continue

        # relative to FRIEND_DIR
        cand = os.path.join(friend_dir, p)
        if os.path.isdir(cand):
            add_dir(cand); continue
        if os.path.isfile(cand):
            out.append(cand); continue

        # relative to script dir + FRIEND_DIR
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cand2 = os.path.join(base_dir, friend_dir, p)
        if os.path.isdir(cand2):
            add_dir(cand2); continue
        if os.path.isfile(cand2):
            out.append(cand2); continue

        print(f"[WARN] Friend image not found (skipped): {p}")

    return out

FRIEND_IMGS = _resolve_friend_paths(FRIEND_IMGS, FRIEND_DIR)
print(f"[STATUS] Looking for friend images in: {FRIEND_DIR}")
print(f"[STATUS] Resolved {len(FRIEND_IMGS)} friend images.")

# ---------------- FRIEND EMBEDDINGS (robust) ----------------
friend_embs = []
friend_names = []  # parallel to friend_embs

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
    # e.g., frnd_imgs/agam_1.jpeg -> "Agam"
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

# ---------------- YOLO ----------------
def load_yolo():
    print("🔄 Loading YOLO model...")
    m = YOLO(LOCAL_MODEL)
    print("✅ YOLO ready.")
    return m

# ---------------- HELPERS ----------------
def area_ratio(x1,y1,x2,y2,W,H):
    return max(0, (x2-x1)*(y2-y1)) / (W*H + 1e-9)

def cosine_sim(u,v):
    return float(np.dot(u,v))

def classify_close_crop(person_crop):
    """Return ('FRIEND'|'FOE'|'UNKNOWN', similarity, best_idx)."""
    if not DEEPFACE_OK or person_crop is None or person_crop.size == 0:
        return "UNKNOWN", 0.0, -1
    try:
        chip = extract_best_face_chip(person_crop, detector="retinaface")
        if chip is None or chip.size == 0:
            return "UNKNOWN", 0.0, -1

        h, w = chip.shape[:2]
        if min(h, w) < 96:   # lower to 64 if your faces are small
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

# ---------------- LIGHTWEIGHT TRACKER ----------------
NEXT_TRACK_ID = 1
# id -> {bbox, center, seen, missing, label, sim, close_since, sim_hist}
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
    model = load_yolo()
    cap, last_try = None, 0
    last_alert_time = 0
    system_status = "🟢 Secure"
    csv_file, csv_writer = ensure_csv_header(CSV_LOG)

    while True:
        # (re)connect
        if cap is None or not cap.isOpened():
            if time.time() - last_try < 1:
                time.sleep(0.2); continue
            last_try = time.time()
            src = CAMERA_URL
            try:
                if str(src).strip().isdigit():
                    src = int(str(src).strip())
            except Exception:
                pass
            print("🔌 Connecting:", src)
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int) else cv2.VideoCapture(src)
            if not cap.isOpened():
                print("⚠️ Failed. Retrying..."); cap=None; continue
            print("🎥 Connected!")

        ok, frame = cap.read()
        if not ok:
            print("⚠️ Frame read failed. Reconnecting...")
            cap.release(); cap=None; continue

        H, W = frame.shape[:2]

        # age tracks
        for t in tracks.values():
            t['missing'] = t.get('missing', 0) + 1

        # detect
        import torch
        DEVICE = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        res = model.predict(frame, classes=[0], conf=CONF_PERSON_CONF, verbose=False, device=DEVICE)[0]

        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0,4))

        any_confirmed_foe_close = False
        foe_crops = []

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

            # --- momentum: require consecutive frames for FRIEND/FOE ---
            t['friend_streak'] = 1 + t.get('friend_streak', 0) if label == "FRIEND" else 0
            t['foe_streak']    = 1 + t.get('foe_streak', 0)    if label == "FOE"    else 0

            COMMIT_N = 2  # need 2 consecutive frames to commit
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
                send_email(
                    "⚠️ Intruder Detected!",
                    f"Unknown person(s) detected at {time.ctime()} — see attachments.",
                    [full_snap, montage] if montage else [full_snap]
                )
                last_alert_time = time.time()
        else:
            system_status = "🟢 Secure"

        # HUD
        cv2.rectangle(frame, (0,0), (420,40), (0,0,0), -1)
        cv2.putText(frame, system_status, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0) if "Secure" in system_status else (0,0,255), 2)

        cv2.imshow("BorderGuard", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(os.path.join(SNAP_DIR, f"manual_{now_str()}.jpg"), frame)

    csv_file.close(); cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera()