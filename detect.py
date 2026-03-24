"""
=============================================================
 detect.py — Real-Time Drowsy Driver Detection (v2)
 Improvements:
   - Calibration step (personalized EAR baseline)
   - CNN confidence % displayed on screen
   - Session logger (saves alerts to CSV)
   - Better hand handling when not visible
   - Session summary on quit
=============================================================
 USAGE:
   python detect.py
 REQUIREMENTS:
   models/best_model.pth (run train.py first)
=============================================================
"""

import cv2, time, numpy as np, torch, torch.nn as nn
import mediapipe as mp, csv, os
from datetime import datetime
from torchvision import models, transforms
from collections import deque

try:
    import pygame
    pygame.mixer.init()
    AUDIO = True
except:
    AUDIO = False
    print("[WARN] pygame not found — audio alerts disabled")

# =============================================================
# CONFIG
# =============================================================
MODEL_PATH   = "./models/best_model.pth"
EAR_THRESH   = 0.25
EAR_CONSEC   = 1.5
HEAD_THRESH  = 20.0
HAND_WINDOW  = 10
DAI_W        = (0.5, 0.3, 0.2)
DAI_MILD     = 0.3
DAI_WARN     = 0.6
DAI_CRIT     = 0.8
IMG_SIZE     = 224
CALIB_SECS   = 3
LOG_DIR      = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

# =============================================================
# SESSION LOGGER
# =============================================================
session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path    = os.path.join(LOG_DIR, f"session_{session_id}.csv")
log_file    = open(log_path, "w", newline="")
log_writer  = csv.writer(log_file)
log_writer.writerow(["timestamp","alert_level","dai","ear","head_pitch","hand_score","cnn_conf"])

def log_event(level, dai, ear, pitch, hand, cnn):
    log_writer.writerow([
        datetime.now().strftime("%H:%M:%S.%f")[:-3],
        ["NONE","MILD","WARNING","CRITICAL"][level],
        f"{dai:.3f}", f"{ear:.3f}", f"{pitch:.1f}", f"{hand:.3f}", f"{cnn:.3f}"
    ])
    log_file.flush()

# =============================================================
# LOAD CNN
# =============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(m.fc.in_features, 256),
        nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)
    )
    try:
        m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"[CNN] Loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[WARN] Model not found — CNN disabled. Run train.py first.")
        return None
    m.eval()
    return m.to(device)

cnn_model = load_model()
cnn_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# =============================================================
# MEDIAPIPE
# =============================================================
mp_face  = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)
hands_det = mp_hands.Hands(max_num_hands=2,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

LEFT_EYE  = [362,385,387,263,373,380]
RIGHT_EYE = [33,160,158,133,153,144]

def calc_ear(landmarks, eye_pts, w, h):
    pts = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in eye_pts]
    A = np.linalg.norm(np.array(pts[1])-np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2])-np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0])-np.array(pts[3]))
    return (A+B)/(2.0*C) if C > 0 else 0.3

def calc_pitch(landmarks, w, h):
    nose  = landmarks[1]
    chin  = landmarks[152]
    fore  = landmarks[10]
    dy_nc = (chin.y - nose.y) * h
    dy_fn = (nose.y - fore.y) * h
    if dy_fn < 1: return 0.0
    ratio = dy_nc / dy_fn
    return min(max(0.0, (ratio-1.0)*40.0), 60.0)

# =============================================================
# STATE
# =============================================================
hand_buffer    = deque(maxlen=HAND_WINDOW)
dai_buffer     = deque(maxlen=20)
prev_wrist     = None
eye_close_t    = None
alert_cooldown = 0.0
fps_buffer     = deque(maxlen=30)

session_start  = time.time()
total_alerts   = {1:0, 2:0, 3:0}
total_dai      = []
drowsy_frames  = 0
total_frames   = 0

calibrating       = True
calib_start       = None
calib_ears        = []
ear_thresh_calib  = EAR_THRESH

# =============================================================
# ALERT
# =============================================================
def play_alert(level):
    if not AUDIO: return
    freq = {1:440, 2:880, 3:1200}.get(level, 440)
    dur  = {1:0.2, 2:0.4, 3:0.6}.get(level, 0.2)
    try:
        t   = np.linspace(0, dur, int(44100*dur))
        arr = (32767*np.sin(2*np.pi*freq*t)).astype(np.int16)
        stereo = np.column_stack([arr, arr])
        pygame.sndarray.make_sound(stereo).play()
    except: pass

# =============================================================
# OVERLAY
# =============================================================
def draw_overlay(frame, dai, ear, pitch, hand, alert_level,
                 fps, cnn_conf, calibrating, calib_remaining):
    h, w = frame.shape[:2]

    panel = frame.copy()
    cv2.rectangle(panel, (0,0), (340,230), (0,0,0), -1)
    cv2.addWeighted(panel, 0.55, frame, 0.45, 0, frame)

    color  = [(0,200,0),(0,165,255),(0,100,255),(0,0,220)][alert_level]
    status = ["ALERT","MILD","WARNING","CRITICAL"][alert_level]

    # DAI bar
    bx, by, bw, bh = 350, 15, 280, 22
    cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (40,40,40), -1)
    cv2.rectangle(frame, (bx,by), (bx+int(dai*bw),by+bh), color, -1)
    cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (180,180,180), 1)
    cv2.putText(frame, "Driver Attention Index", (bx, by-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
    cv2.putText(frame, f"{dai:.2f}", (bx+bw+8, by+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Metrics
    lines = [
        (f"EAR     : {ear:.3f}",              (200,200,200)),
        (f"Head    : {pitch:.1f} deg",         (200,200,200)),
        (f"Hand    : {hand:.2f}",              (200,200,200)),
        (f"CNN     : {cnn_conf*100:.0f}% drowsy",(200,200,200)),
        (f"FPS     : {fps:.0f}",               (150,150,150)),
    ]
    for i,(txt,c) in enumerate(lines):
        cv2.putText(frame, txt, (10, 30+i*26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1)

    cv2.putText(frame, status, (10, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    # Alert dots
    labels = ["MILD","WARN","CRIT"]
    threshs = [DAI_MILD, DAI_WARN, DAI_CRIT]
    for i,(lbl,thr) in enumerate(zip(labels,threshs)):
        dc = (0,200,0) if dai < thr else (0,0,220)
        cv2.circle(frame, (12+i*30, 210), 7, dc, -1)
        cv2.putText(frame, lbl, (2+i*30, 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (180,180,180), 1)

    if alert_level >= 3:
        cv2.rectangle(frame, (0,0), (w,h), (0,0,220), 6)

    if calibrating:
        cv2.rectangle(frame, (0, h//2-55), (w, h//2+55), (0,0,0), -1)
        cv2.putText(frame, "CALIBRATING", (w//2-110, h//2-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,220,255), 2)
        cv2.putText(frame, f"Keep eyes open — {calib_remaining:.1f}s remaining",
                    (w//2-175, h//2+28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
    return frame

# =============================================================
# SESSION SUMMARY
# =============================================================
def print_summary():
    duration   = time.time() - session_start
    avg_dai    = np.mean(total_dai) if total_dai else 0
    drowsy_pct = (drowsy_frames/total_frames*100) if total_frames > 0 else 0
    print("\n"+"="*50)
    print("  SESSION SUMMARY")
    print("="*50)
    print(f"  Duration       : {duration/60:.1f} minutes")
    print(f"  Avg DAI        : {avg_dai:.3f}")
    print(f"  Drowsy time    : {drowsy_pct:.1f}% of session")
    print(f"  Mild alerts    : {total_alerts[1]}")
    print(f"  Warning alerts : {total_alerts[2]}")
    print(f"  Critical alerts: {total_alerts[3]}")
    print(f"  Log saved      : {log_path}")
    print("="*50)
    log_file.close()

# =============================================================
# MAIN LOOP
# =============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("\n[DETECT] Keep eyes OPEN for 3-second calibration")
print("[DETECT] Press Q to quit\n")

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret: break
    h, w   = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    total_frames += 1

    # Face & EAR
    ear_val = 0.3; f_eye_score = 0.0; f_head_score = 0.0; pitch = 0.0
    face_res = face_mesh.process(rgb)

    if face_res.multi_face_landmarks:
        lm      = face_res.multi_face_landmarks[0].landmark
        l       = calc_ear(lm, LEFT_EYE,  w, h)
        r       = calc_ear(lm, RIGHT_EYE, w, h)
        ear_val = (l+r)/2.0

        # Calibration
        if calibrating:
            if calib_start is None: calib_start = time.time()
            calib_ears.append(ear_val)
            if time.time()-calib_start >= CALIB_SECS:
                ear_thresh_calib = max(0.18, np.mean(calib_ears)*0.75)
                calibrating = False
                print(f"[CALIB] Done. EAR threshold set to {ear_thresh_calib:.3f}")

        if not calibrating:
            if ear_val < ear_thresh_calib:
                if eye_close_t is None: eye_close_t = time.time()
                f_eye_score = min(1.0, (time.time()-eye_close_t)/EAR_CONSEC)
            else:
                eye_close_t = None; f_eye_score = 0.0

        pitch        = calc_pitch(lm, w, h)
        f_head_score = min(1.0, max(0.0, (pitch-5.0)/HEAD_THRESH))

    # Hands
    f_hand_score = 0.3
    hand_res = hands_det.process(rgb)
    if hand_res.multi_hand_landmarks:
        wrist = hand_res.multi_hand_landmarks[0].landmark[0]
        wx, wy = wrist.x*w, wrist.y*h
        if prev_wrist is not None:
            hand_buffer.append(np.sqrt((wx-prev_wrist[0])**2+(wy-prev_wrist[1])**2))
        prev_wrist = (wx, wy)
        if len(hand_buffer) >= 3:
            f_hand_score = max(0.0, 1.0-min(1.0, np.mean(hand_buffer)/15.0))
    else:
        prev_wrist = None

    # CNN
    cnn_conf = 0.0
    if cnn_model is not None and not calibrating:
        with torch.no_grad():
            inp      = cnn_tf(rgb).unsqueeze(0).to(device)
            out      = torch.softmax(cnn_model(inp), dim=1)
            cnn_conf = out[0][0].item()

    # DAI
    dai = 0.0
    if not calibrating:
        w1,w2,w3 = DAI_W
        dai_raw  = w1*f_eye_score + w2*f_head_score + w3*f_hand_score
        if cnn_model is not None:
            dai_raw = 0.7*dai_raw + 0.3*cnn_conf
        dai_buffer.append(dai_raw)
        dai = float(np.mean(dai_buffer))
        total_dai.append(dai)
        if dai > DAI_MILD: drowsy_frames += 1

    # Alert
    alert_level = 0
    if not calibrating:
        if dai >= DAI_CRIT:   alert_level = 3
        elif dai >= DAI_WARN: alert_level = 2
        elif dai >= DAI_MILD: alert_level = 1

    now = time.time()
    if alert_level > 0 and now > alert_cooldown:
        play_alert(alert_level)
        total_alerts[alert_level] += 1
        log_event(alert_level, dai, ear_val, pitch, f_hand_score, cnn_conf)
        alert_cooldown = now + {1:3.0, 2:2.0, 3:1.0}[alert_level]

    fps_buffer.append(1.0/(time.time()-t0+1e-6))
    fps = np.mean(fps_buffer)

    calib_rem = max(0, CALIB_SECS-(time.time()-calib_start)) if calib_start else CALIB_SECS
    frame = draw_overlay(frame, dai, ear_val, pitch, f_hand_score,
                         alert_level, fps, cnn_conf, calibrating, calib_rem)

    cv2.imshow("Drowsy Driver Detection  |  Press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print_summary()