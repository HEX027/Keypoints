# Lab #6 — Keypoints Lab 🎮
### Theme: Watch Dogs / ctOS Hacker Interface
**Platform:** NVIDIA Jetson Orin Nano | **Container:** Dusty-NV Jetson Containers | **Model:** MediaPipe Pose (Lite)

---

## 🔗 GitHub Repository

> **[[ INSERT YOUR GITHUB LINK HERE ]]**
> e.g., `https://github.com/YOUR_USERNAME/lab6-keypoints`

---

## 📋 Lab Overview

This lab demonstrates **keypoint detection / pose estimation** using **MediaPipe Pose** running inside a Docker container on the NVIDIA Jetson Orin Nano. The program monitors a live webcam feed and triggers an **anomaly alert** when both of the subject's wrists are raised above their shoulders — simulating a "hands up" threat detection scenario.

The OpenCV display is styled as a **ctOS hacker surveillance interface** inspired by the Watch Dogs video game franchise, featuring:
- Animated scan lines
- Corner-bracketed target reticle
- Live FPS and timestamp HUD panels
- Flashing red alert banner on anomaly confirmation
- Dark grid overlay and vignette effects

---

## 🐳 Docker Run Command

Run the following on your **Jetson Orin Nano terminal**:

```bash
docker run --runtime nvidia -it --rm \
  --device /dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/lab6 \
  --workdir /lab6 \
  dustynv/jetson-inference:r36.2.0 \
  /bin/bash
```

> **Notes:**
> - `--runtime nvidia` enables GPU/CUDA access.
> - `--device /dev/video0` passes the USB webcam into the container.
> - `-e DISPLAY` and `-v /tmp/.X11-unix` allow the `cv2.imshow` window to render on the Jetson desktop.
> - Mount your script directory with `-v $(pwd):/lab6`.
> - If your webcam is on a different index, change `/dev/video0` to `/dev/video1`, etc.

### Inside the container — install dependencies:

```bash
pip install mediapipe opencv-python numpy
```

### Run the script:

```bash
python3 watchdogs_keypoints.py
```

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Framework | MediaPipe |
| Model | Pose (Lite — `model_complexity=0`) |
| Input | Live webcam (BGR → RGB) |
| Keypoints | 33 body landmarks |
| Inference | CPU (ARM Cortex-A78AE) |
| FPS (approx.) | 15–25 FPS on Jetson Orin Nano |

---

## 🚨 Anomaly Definition

**Trigger condition:** Both wrists must be positioned **above** their respective shoulder landmarks in normalized image coordinates (`wrist.y < shoulder.y`).

This models a **"hands raised"** posture — a recognizable threat/surrender signal.

A **1.5-second hold** is required before the full red confirmation banner fires, reducing false positives from incidental arm movement.

---

## 🖼️ Results & Screenshots

### Normal State — Monitoring
> *Replace the placeholder below with your own screenshot.*

```
[ INSERT screenshot: normal_state.jpg ]
```
*Description: Subject standing normally. ctOS HUD shows green "NOMINAL — NO THREAT" panel.*

---

### Anomaly Detected — Hands Raised ⚠️
> *Take this photo with your phone aimed at the Jetson's monitor/display output.*

```
[ INSERT screenshot: anomaly_detected.jpg ]
```
*Description: Both arms raised. Red "!! ANOMALY DETECTED !!" panel is active. Centre banner reads ">>> THREAT CONFIRMED — HANDS UP DETECTED <<<". Skeleton overlay turns red.*

> 📸 **Note:** The anomaly screenshot was captured using a **phone camera** aimed at the Jetson Orin Nano's display output, as the lab environment does not support remote screenshot capture.

---

### Terminal Output
```
[ INSERT screenshot: terminal_output.jpg ]
```
*Description: Docker container startup, ctOS header banner, and any console messages.*

---

## 📂 File Structure

```
lab6-keypoints/
├── watchdogs_keypoints.py   # Main Watch Dogs themed script
├── README.md                # This file
└── screenshots/
    ├── normal_state.jpg
    ├── anomaly_detected.jpg
    └── terminal_output.jpg
```

---

## ⚙️ Key Code Highlights

### Anomaly Detection Function
```python
def check_anomaly(landmarks, img_h, img_w):
    lm = landmarks.landmark
    left_shoulder  = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_wrist     = lm[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist    = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
    both_wrists_up = (left_wrist.y  < left_shoulder.y and
                      right_wrist.y < right_shoulder.y)
    return both_wrists_up
```

### ctOS HUD Panel Drawing
The `draw_panel()` function renders a semi-transparent dark rectangle with corner bracket accents — a signature Watch Dogs UI element — using `cv2.addWeighted` for the alpha blend and explicit `cv2.line` calls for each bracket.

---

## 🎯 Learning Objectives Met

- [x] Ran a computer vision model inside a Docker container on Jetson Orin Nano
- [x] Used a keypoint/pose estimation model (MediaPipe Pose)
- [x] Defined and detected a custom body-pose anomaly
- [x] Rendered a themed real-time overlay using OpenCV drawing functions
- [x] Demonstrated the anomaly detection live (screenshot provided)

---

## 🔧 Troubleshooting

| Issue | Fix |
|---|---|
| `Cannot open camera device 0` | Check webcam index: try `cv2.VideoCapture(1)` |
| `cv2.imshow` window does not open | Run `xhost +local:docker` on the Jetson host before starting container |
| Low FPS | Reduce resolution: change `1280×720` to `640×480` in `cap.set()` calls |
| MediaPipe install fails | Use `pip install mediapipe --extra-index-url https://google-coral.github.io/py-repo/` |

---

## 👤 Author

| Field | Value |
|---|---|
| Name | `[ YOUR NAME ]` |
| Course | `[ COURSE NAME / NUMBER ]` |
| Date | `[ SUBMISSION DATE ]` |
| Instructor | `[ INSTRUCTOR NAME ]` |

---

*Lab #6 — Keypoints Lab — Watch Dogs Edition*
