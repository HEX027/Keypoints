#!/usr/bin/env python3
"""
============================================================
  LAB #6 — KEYPOINTS LAB
  Theme: Watch Dogs / ctOS Hacker Interface
  Platform: NVIDIA Jetson Orin Nano (Docker)
  Model: MediaPipe Pose (lightweight, CPU-friendly)
  Anomaly: Both arms raised above the head
============================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math

# ─────────────────────────────────────────────
#  ctOS COLOR PALETTE  (Watch Dogs aesthetic)
# ─────────────────────────────────────────────
CTOS_CYAN        = (255, 230,   0)   # BGR — electric cyan
CTOS_GREEN       = (  0, 255, 120)   # BGR — matrix green
CTOS_ORANGE      = (  0, 165, 255)   # BGR — alert orange
CTOS_RED         = (  0,  40, 220)   # BGR — threat red
CTOS_WHITE       = (220, 220, 220)   # BGR — dim white
CTOS_DIM         = ( 40,  40,  40)   # BGR — dark panel
CTOS_ACCENT      = (200, 255,   0)   # BGR — yellow accent
BG_COLOR         = (  8,   8,   8)   # near-black background overlay

# ─────────────────────────────────────────────
#  MEDIAPIPE INITIALISATION
# ─────────────────────────────────────────────
mp_pose     = mp.solutions.pose
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,          # 0 = Lite — fastest on Jetson
    smooth_landmarks=True,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55,
)

# ─────────────────────────────────────────────
#  STATE VARIABLES
# ─────────────────────────────────────────────
anomaly_detected     = False
anomaly_start_time   = 0.0
anomaly_hold_secs    = 1.5          # must hold pose this long to trigger
scan_line_y          = 0            # animated scan line
frame_counter        = 0
fps_display          = 0.0
fps_timer            = time.time()
alert_flash_toggle   = True
alert_flash_timer    = time.time()
target_id            = "SUBJECT_4471"   # fake ctOS target ID


# ─────────────────────────────────────────────
#  HELPER: draw a bordered panel (glassmorphism-ish)
# ─────────────────────────────────────────────
def draw_panel(frame, x, y, w, h, color=CTOS_CYAN, alpha=0.25):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), CTOS_DIM, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # border
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
    # corner accents
    corner = 8
    cv2.line(frame, (x, y),           (x + corner, y),           color, 2)
    cv2.line(frame, (x, y),           (x, y + corner),           color, 2)
    cv2.line(frame, (x + w, y),       (x + w - corner, y),       color, 2)
    cv2.line(frame, (x + w, y),       (x + w, y + corner),       color, 2)
    cv2.line(frame, (x, y + h),       (x + corner, y + h),       color, 2)
    cv2.line(frame, (x, y + h),       (x, y + h - corner),       color, 2)
    cv2.line(frame, (x + w, y + h),   (x + w - corner, y + h),   color, 2)
    cv2.line(frame, (x + w, y + h),   (x + w, y + h - corner),   color, 2)


def draw_ctos_text(frame, text, x, y, color=CTOS_CYAN,
                   scale=0.45, thickness=1):
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)


def draw_dashed_line(frame, pt1, pt2, color, thickness=1, gap=8):
    dist  = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    steps = int(dist / gap)
    for i in range(steps):
        t0 = i / steps
        t1 = min((i + 0.5) / steps, 1.0)
        p0 = (int(pt1[0] + t0 * (pt2[0] - pt1[0])),
              int(pt1[1] + t0 * (pt2[1] - pt1[1])))
        p1 = (int(pt1[0] + t1 * (pt2[0] - pt1[0])),
              int(pt1[1] + t1 * (pt2[1] - pt1[1])))
        cv2.line(frame, p0, p1, color, thickness)


# ─────────────────────────────────────────────
#  ANOMALY DETECTION LOGIC
#  Anomaly = both wrists raised above both shoulders
# ─────────────────────────────────────────────
def check_anomaly(landmarks, img_h, img_w):
    lm = landmarks.landmark

    left_shoulder  = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_wrist     = lm[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist    = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

    # y in normalised coords — smaller y = higher on screen
    both_wrists_up = (left_wrist.y  < left_shoulder.y  and
                      right_wrist.y < right_shoulder.y)
    return both_wrists_up


# ─────────────────────────────────────────────
#  DRAW SKELETON  (ctOS style)
# ─────────────────────────────────────────────
def draw_ctos_skeleton(frame, landmarks, img_h, img_w, anomaly):
    lm     = landmarks.landmark
    color  = CTOS_RED if anomaly else CTOS_CYAN

    # Pairs to draw
    connections = [
        # torso
        (mp_pose.PoseLandmark.LEFT_SHOULDER,  mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.LEFT_SHOULDER,  mp_pose.PoseLandmark.LEFT_HIP),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
        (mp_pose.PoseLandmark.LEFT_HIP,       mp_pose.PoseLandmark.RIGHT_HIP),
        # arms
        (mp_pose.PoseLandmark.LEFT_SHOULDER,  mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_ELBOW,     mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_ELBOW,    mp_pose.PoseLandmark.RIGHT_WRIST),
        # legs
        (mp_pose.PoseLandmark.LEFT_HIP,       mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.LEFT_KNEE,      mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_HIP,      mp_pose.PoseLandmark.RIGHT_KNEE),
        (mp_pose.PoseLandmark.RIGHT_KNEE,     mp_pose.PoseLandmark.RIGHT_ANKLE),
    ]

    for a_idx, b_idx in connections:
        a = lm[a_idx]
        b = lm[b_idx]
        if a.visibility > 0.4 and b.visibility > 0.4:
            ax, ay = int(a.x * img_w), int(a.y * img_h)
            bx, by = int(b.x * img_w), int(b.y * img_h)
            cv2.line(frame, (ax, ay), (bx, by), color, 2)

    # Keypoint dots
    for idx, point in enumerate(lm):
        if point.visibility > 0.4:
            px, py = int(point.x * img_w), int(point.y * img_h)
            cv2.circle(frame, (px, py), 4, color, -1)
            cv2.circle(frame, (px, py), 6, color, 1)

    # Head circle
    nose = lm[mp_pose.PoseLandmark.NOSE]
    if nose.visibility > 0.4:
        nx, ny = int(nose.x * img_w), int(nose.y * img_h)
        cv2.circle(frame, (nx, ny - 18), 22, color, 2)
        # targeting reticle
        cv2.line(frame, (nx - 30, ny - 18), (nx - 10, ny - 18), color, 1)
        cv2.line(frame, (nx + 10, ny - 18), (nx + 30, ny - 18), color, 1)
        cv2.line(frame, (nx, ny - 48),      (nx, ny - 28),      color, 1)
        cv2.line(frame, (nx, ny - 8),       (nx, ny + 12),      color, 1)


# ─────────────────────────────────────────────
#  MAIN OVERLAY  (HUD)
# ─────────────────────────────────────────────
def draw_hud(frame, fps, anomaly, anomaly_timer, scan_y, flash):
    h, w = frame.shape[:2]
    now_str  = time.strftime("%H:%M:%S")
    date_str = time.strftime("%Y.%m.%d")

    # ── Dark vignette overlay ──
    vignette = np.zeros((h, w), dtype=np.uint8)
    for i in range(60):
        alpha_v = int(180 * (1 - i / 60))
        cv2.rectangle(vignette, (i, i), (w - i, h - i), alpha_v, 1)
    vignette_color = np.zeros_like(frame)
    vignette_color[:, :, 0] = vignette   # tint blue channel
    cv2.addWeighted(frame, 1.0, vignette_color, 0.15, 0, frame)

    # ── Animated scan line ──
    cv2.line(frame, (0, scan_y), (w, scan_y), CTOS_CYAN, 1)
    scan_overlay = frame.copy()
    cv2.rectangle(scan_overlay, (0, scan_y - 2), (w, scan_y + 2), CTOS_CYAN, -1)
    cv2.addWeighted(scan_overlay, 0.07, frame, 0.93, 0, frame)

    # ── Top-left panel: ctOS branding ──
    draw_panel(frame, 8, 8, 260, 60, CTOS_CYAN)
    draw_ctos_text(frame, "ctOS  SURVEILLANCE  v3.1", 16, 28, CTOS_CYAN, 0.48)
    draw_ctos_text(frame, f"DATE: {date_str}   TIME: {now_str}", 16, 52, CTOS_WHITE, 0.38)

    # ── Top-right panel: FPS / status ──
    draw_panel(frame, w - 200, 8, 192, 60, CTOS_GREEN)
    draw_ctos_text(frame, f"FPS : {fps:.1f}", w - 192, 28, CTOS_GREEN, 0.48)
    draw_ctos_text(frame, "CAM : ACTIVE", w - 192, 52, CTOS_GREEN, 0.38)

    # ── Bottom-left panel: target info ──
    draw_panel(frame, 8, h - 80, 280, 70, CTOS_CYAN)
    draw_ctos_text(frame, f"TARGET ID : {target_id}", 16, h - 58, CTOS_CYAN, 0.44)
    draw_ctos_text(frame, "PROFILE   : UNKNOWN CIVILIAN", 16, h - 38, CTOS_WHITE, 0.38)
    draw_ctos_text(frame, "THREAT LVL: MONITORING...", 16, h - 20, CTOS_WHITE, 0.36)

    # ── Bottom-right: anomaly status ──
    status_color = CTOS_RED if anomaly else CTOS_GREEN
    status_text  = "!! ANOMALY DETECTED !!" if anomaly else "NOMINAL — NO THREAT"
    draw_panel(frame, w - 280, h - 80, 272, 70, status_color)
    if anomaly and flash:
        draw_ctos_text(frame, status_text, w - 272, h - 55, status_color, 0.48, 2)
    elif not anomaly:
        draw_ctos_text(frame, status_text, w - 272, h - 55, status_color, 0.44)

    draw_ctos_text(frame, "POSE: ARMS RAISED = TRIGGER",
                   w - 272, h - 32, CTOS_WHITE, 0.35)
    if anomaly:
        hold_pct = min(int((time.time() - anomaly_timer) / anomaly_hold_secs * 100), 100)
        bar_w    = int(250 * hold_pct / 100)
        cv2.rectangle(frame, (w - 272, h - 16), (w - 272 + 250, h - 8), CTOS_DIM, -1)
        cv2.rectangle(frame, (w - 272, h - 16), (w - 272 + bar_w, h - 8), CTOS_RED, -1)
        draw_ctos_text(frame, f"CONFIRM {hold_pct}%", w - 272, h - 20, CTOS_ORANGE, 0.35)

    # ── Centre alert banner (anomaly confirmed) ──
    if anomaly and (time.time() - anomaly_timer) >= anomaly_hold_secs and flash:
        banner_h = 50
        overlay  = frame.copy()
        cv2.rectangle(overlay, (0, h // 2 - banner_h),
                      (w, h // 2 + banner_h), CTOS_RED, -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        draw_ctos_text(frame, ">>> THREAT CONFIRMED — HANDS UP DETECTED <<<",
                       w // 2 - 285, h // 2 + 8, CTOS_RED, 0.7, 2)

    # ── Decorative grid lines ──
    for gx in range(0, w, 80):
        cv2.line(frame, (gx, 0), (gx, h), (30, 30, 30), 1)
    for gy in range(0, h, 80):
        cv2.line(frame, (0, gy), (w, gy), (30, 30, 30), 1)

    # ── Corner crosshairs ──
    for cx, cy in [(20, 20), (w - 20, 20), (20, h - 20), (w - 20, h - 20)]:
        cv2.drawMarker(frame, (cx, cy), CTOS_CYAN,
                       cv2.MARKER_CROSS, 16, 1)


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def main():
    global anomaly_detected, anomaly_start_time
    global scan_line_y, frame_counter
    global fps_display, fps_timer
    global alert_flash_toggle, alert_flash_timer

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS,           30)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera device 0")
        return

    print("=" * 55)
    print("  ctOS SURVEILLANCE SYSTEM — KEYPOINTS LAB #6")
    print("  Raise BOTH hands above your shoulders to trigger")
    print("  Press  Q  to quit")
    print("=" * 55)

    window_name = "ctOS | Watch Dogs — Keypoints Lab"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    with pose_detector as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Dropped frame")
                continue

            frame_counter += 1
            h, w = frame.shape[:2]

            # ── FPS calc ──
            if frame_counter % 15 == 0:
                fps_display = 15.0 / (time.time() - fps_timer)
                fps_timer   = time.time()

            # ── Scan line animation ──
            scan_line_y = (scan_line_y + 3) % h

            # ── Flash toggle ──
            if time.time() - alert_flash_timer > 0.4:
                alert_flash_toggle = not alert_flash_toggle
                alert_flash_timer  = time.time()

            # ── Dark overlay for aesthetic ──
            dark = np.zeros_like(frame)
            cv2.addWeighted(dark, 0.18, frame, 0.82, 0, frame)

            # ── Pose inference ──
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            currently_anomalous = False

            if result.pose_landmarks:
                draw_ctos_skeleton(frame, result.pose_landmarks, h, w, anomaly_detected)
                currently_anomalous = check_anomaly(result.pose_landmarks, h, w)

                if currently_anomalous:
                    if not anomaly_detected:
                        anomaly_detected   = True
                        anomaly_start_time = time.time()
                else:
                    anomaly_detected   = False
                    anomaly_start_time = 0.0
            else:
                anomaly_detected   = False
                anomaly_start_time = 0.0

            # ── Draw HUD ──
            draw_hud(frame, fps_display, anomaly_detected,
                     anomaly_start_time, scan_line_y, alert_flash_toggle)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[ctOS] Session terminated by operator.")
                break

    cap.release()
    cv2.destroyAllWindows()
    pose_detector.close()


if __name__ == "__main__":
    main()
