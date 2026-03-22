"""
=============================================================
 detect.py — Real-Time Drowsy Driver Detection
 Uses: Trained CNN + MediaPipe (EAR + Head Tilt + Hands)
 Fuses everything into Driver Attention Index (DAI)
=============================================================
 USAGE:
   python detect.py
 REQUIREMENTS:
   Run train.py first to generate models/best_model.pth
=============================================================
"""

import cv2, time, numpy as np, torch, torch.nn as nn
import mediapipe as mp
from torchvision import models, transforms
from scipy.signal import savgol_filter
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
EAR_CONSEC   = 1.5        # seconds
HEAD_THRESH  = 20.0       # degrees
HAND_WINDOW  = 10         # frames
DAI_W        = (0.5, 0.3, 0.2)   # eye, head, hand weights
DAI_MILD     = 0.3
DAI_WARN     = 0.6
DAI_CRIT     = 0.8
IMG_SIZE     = 224

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
        print(f"[CNN] Loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[WARN] Model not found at {MODEL_PATH} — CNN score disabled. Run train.py first.")
        return None
    m.eval(); return m.to(device)

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
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)
hands_det = mp_hands.Hands(max_num_hands=2,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

# Eye landmark indices (MediaPipe FaceMesh)
LEFT_EYE  = [362,385,387,263,373,380]
RIGHT_EYE = [33,160,158,133,153,144]

def ear(landmarks, eye_pts, w, h):
    pts = [(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in eye_pts]
    A = np.linalg.norm(np.array(pts[1])-np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2])-np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0])-np.array(pts[3]))
    return (A+B)/(2.0*C) if C > 0 else 0.3

def head_pitch(landmarks, w, h):
    # Estimate forward pitch from nose tip vs forehead vs chin
    nose   = landmarks[1]
    chin   = landmarks[152]
    fore   = landmarks[10]
    dy_nc  = (chin.y - nose.y) * h
    dy_fn  = (nose.y - fore.y) * h
    if dy_fn < 1: return 0.0
    ratio  = dy_nc / dy_fn
    # ratio > 1.3 typically means head is pitching forward
    pitch  = max(0.0, (ratio - 1.0) * 40.0)
    return min(pitch, 60.0)

# =============================================================
# STATE
# =============================================================
ear_buffer    = deque(maxlen=30)
hand_buffer   = deque(maxlen=HAND_WINDOW)
dai_buffer    = deque(maxlen=20)
prev_wrist    = None
eye_close_t   = None
alert_cooldown= 0.0
fps_buffer    = deque(maxlen=30)

# =============================================================
# ALERT
# =============================================================
def play_alert(level):
    if not AUDIO: return
    freq = {1: 440, 2: 880, 3: 1200}.get(level, 440)
    dur  = {1: 200, 2: 400, 3: 600}.get(level, 200)
    try:
        arr = np.array([
            32767 * np.sin(2*np.pi*freq*t/44100)
            for t in range(int(44100*dur/1000))
        ], dtype=np.int16)
        stereo = np.column_stack([arr, arr])
        snd = pygame.sndarray.make_sound(stereo)
        snd.play()
    except: pass

# =============================================================
# DRAW OVERLAY
# =============================================================
def overlay(frame, dai, ear_val, pitch, hand_score, alert_level, fps):
    h, w = frame.shape[:2]
    # Dark panel
    panel = frame.copy()
    cv2.rectangle(panel, (0,0), (320, 180), (0,0,0), -1)
    cv2.addWeighted(panel, 0.5, frame, 0.5, 0, frame)

    color = (0,255,0) if alert_level == 0 else \
            (0,165,255) if alert_level == 1 else \
            (0,69,255)  if alert_level == 2 else (0,0,255)
    status = ["ALERT","MILD","WARNING","CRITICAL"][alert_level]

    cv2.putText(frame, f"DAI: {dai:.2f}",        (10,30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"EAR: {ear_val:.3f}",    (10,60),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cv2.putText(frame, f"Head: {pitch:.1f}deg",  (10,85),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cv2.putText(frame, f"Hand: {hand_score:.2f}",(10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cv2.putText(frame, f"FPS: {fps:.0f}",        (10,135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cv2.putText(frame, status,                   (10,165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # DAI bar
    bar_x, bar_y, bar_w, bar_h = 340, 20, 300, 25
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50,50,50), -1)
    fill = int(dai * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill, bar_y+bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (200,200,200), 1)
    cv2.putText(frame, "Driver Attention Index", (bar_x, bar_y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    if alert_level >= 3:
        cv2.rectangle(frame, (0,0), (w,h), (0,0,255), 8)

    return frame

# =============================================================
# MAIN LOOP
# =============================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("\n[DETECT] Starting — press Q to quit\n")

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- EAR ---
    ear_val     = 0.3
    f_eye_score = 0.0
    face_res    = face_mesh.process(rgb)
    pitch       = 0.0

    if face_res.multi_face_landmarks:
        lm = face_res.multi_face_landmarks[0].landmark
        l  = ear(lm, LEFT_EYE,  w, h)
        r  = ear(lm, RIGHT_EYE, w, h)
        ear_val = (l + r) / 2.0
        ear_buffer.append(ear_val)

        if ear_val < EAR_THRESH:
            if eye_close_t is None: eye_close_t = time.time()
            closed_dur = time.time() - eye_close_t
            f_eye_score = min(1.0, closed_dur / EAR_CONSEC)
        else:
            eye_close_t = None
            f_eye_score = 0.0

        # Head pitch
        raw_pitch = head_pitch(lm, w, h)
        pitch      = raw_pitch
        f_head_score = min(1.0, max(0.0, (pitch - 5.0) / HEAD_THRESH))
    else:
        f_head_score = 0.0

    # --- HAND STABILITY ---
    f_hand_score = 0.0
    hand_res = hands_det.process(rgb)
    if hand_res.multi_hand_landmarks:
        wrist = hand_res.multi_hand_landmarks[0].landmark[0]
        wx, wy = wrist.x * w, wrist.y * h
        if prev_wrist is not None:
            move = np.sqrt((wx-prev_wrist[0])**2 + (wy-prev_wrist[1])**2)
            hand_buffer.append(move)
        prev_wrist = (wx, wy)
        if len(hand_buffer) >= 3:
            avg_move = np.mean(hand_buffer)
            # Low movement = relaxed grip = higher fatigue score
            f_hand_score = max(0.0, 1.0 - min(1.0, avg_move / 15.0))
    else:
        # Hands not visible — moderate penalty
        f_hand_score = 0.5

    # --- CNN SCORE (optional boost) ---
    cnn_boost = 0.0
    if cnn_model is not None:
        with torch.no_grad():
            inp = cnn_tf(rgb).unsqueeze(0).to(device)
            out = torch.softmax(cnn_model(inp), dim=1)
            cnn_boost = out[0][0].item()  # probability of Drowsy class

    # --- DAI FUSION ---
    w1, w2, w3 = DAI_W
    dai_raw = w1*f_eye_score + w2*f_head_score + w3*f_hand_score
    # Blend with CNN if available
    if cnn_model is not None:
        dai_raw = 0.7*dai_raw + 0.3*cnn_boost
    dai_buffer.append(dai_raw)
    dai = float(np.mean(dai_buffer))

    # --- ALERT LEVEL ---
    alert_level = 0
    if dai >= DAI_CRIT:  alert_level = 3
    elif dai >= DAI_WARN: alert_level = 2
    elif dai >= DAI_MILD: alert_level = 1

    now = time.time()
    if alert_level > 0 and now > alert_cooldown:
        play_alert(alert_level)
        alert_cooldown = now + {1:3.0, 2:2.0, 3:1.0}[alert_level]

    # --- FPS ---
    fps_buffer.append(1.0/(time.time()-t0+1e-6))
    fps = np.mean(fps_buffer)

    # --- DRAW ---
    frame = overlay(frame, dai, ear_val, pitch, f_hand_score, alert_level, fps)
    cv2.imshow("Drowsy Driver Detection — press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[DETECT] Stopped.")