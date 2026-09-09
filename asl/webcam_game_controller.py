#!/usr/bin/env python3
"""
Webcam Hand Gesture Game Controller
-----------------------------------
Detects 5 hand gestures using YOLO classification model:
  'I' -> Forward (Move Right)
  'L' -> Backward (Move Left)
  'X' -> Jump
  'Q' -> Forward + Jump
  'W' -> Backward + Jump

Broadcasts gesture commands over UDP (127.0.0.1:4242) to control Godot / custom games in real-time.
"""

import os
import sys
import time
import socket
from collections import deque, Counter
import cv2
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "runs", "hand_letters_5", "weights", "best.pt")

CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.45
SMOOTHING_FRAMES = 5
ROI_SIZE = 450

UDP_IP = "127.0.0.1"
UDP_PORT = 4242

# Gesture to Action Description Mapping
GESTURE_ACTIONS = {
    "I": ("FORWARD", (0, 255, 0)),         # Green
    "L": ("BACKWARD", (255, 165, 0)),      # Orange
    "X": ("JUMP", (255, 255, 0)),          # Yellow
    "Q": ("FORWARD + JUMP", (0, 255, 255)),# Cyan
    "W": ("BACKWARD + JUMP", (255, 0, 255)),# Magenta
    "NONE": ("IDLE", (128, 128, 128))      # Gray
}


def main():
    print("=" * 65)
    print("        WEBCAM HAND GESTURE GAME CONTROLLER")
    print("=" * 65)
    
    # --------------------------------------------------------
    # SETUP UDP SOCKET
    # --------------------------------------------------------
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[UDP] Broadcasting commands to {UDP_IP}:{UDP_PORT}")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------
    print(f"\n[Model] Loading YOLO model from:\n  {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model file not found at {MODEL_PATH}")
        sys.exit(1)

    try:
        model = YOLO(MODEL_PATH)
        print(f"[Model] Successfully loaded! Classes: {model.names}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)

    # --------------------------------------------------------
    # OPEN CAMERA
    # --------------------------------------------------------
    print(f"\n[Webcam] Opening camera index {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prediction_history = deque(maxlen=SMOOTHING_FRAMES)
    stable_letter = "NONE"
    stable_confidence = 0.0
    previous_time = time.time()

    print("\nStarting gesture loop... Press 'q' or 'ESC' on the camera window to exit.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read camera frame.")
                break

            # Mirror frame for intuitive interaction
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            # Center ROI coordinates
            center_x, center_y = width // 2, height // 2
            x1 = max(0, center_x - ROI_SIZE // 2)
            y1 = max(0, center_y - ROI_SIZE // 2)
            x2 = min(width, center_x + ROI_SIZE // 2)
            y2 = min(height, center_y + ROI_SIZE // 2)

            # Draw Hand ROI Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(
                frame,
                "SHOW HAND GESTURE HERE",
                (x1 + 10, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Crop Hand Area & Predict
            hand_crop = frame[y1:y2, x1:x2]
            current_letter = "NONE"
            current_conf = 0.0

            if hand_crop.size > 0:
                results = model.predict(hand_crop, imgsz=224, verbose=False)
                result = results[0]

                if result.probs is not None:
                    class_id = int(result.probs.top1)
                    confidence = float(result.probs.top1conf)
                    label = str(model.names[class_id]).upper()

                    if confidence >= CONFIDENCE_THRESHOLD:
                        current_letter = label
                        current_conf = confidence
                        prediction_history.append(label)
                    else:
                        prediction_history.append("NONE")

            # Majority Vote Filtering
            if len(prediction_history) > 0:
                counts = Counter(prediction_history)
                best_label, count = counts.most_common(1)[0]
                if count >= (SMOOTHING_FRAMES // 2 + 1):
                    stable_letter = best_label
                    stable_confidence = current_conf if best_label == current_letter else 0.80
                else:
                    stable_letter = "NONE"

            # Send UDP command to Game
            sock.sendto(stable_letter.encode("utf-8"), (UDP_IP, UDP_PORT))

            # --------------------------------------------------------
            # RENDER HUD & OVERLAYS
            # --------------------------------------------------------
            action_name, action_color = GESTURE_ACTIONS.get(stable_letter, ("IDLE", (128, 128, 128)))

            # Header Banner
            cv2.rectangle(frame, (0, 0), (width, 80), (20, 20, 20), -1)
            cv2.putText(
                frame,
                f"GESTURE: {stable_letter}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 255, 255) if stable_letter != "NONE" else (100, 100, 100),
                3
            )
            cv2.putText(
                frame,
                f"ACTION: {action_name}",
                (360, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                action_color,
                3
            )

            # Move Legend HUD Card (Top Right)
            hud_x = width - 310
            cv2.rectangle(frame, (hud_x - 10, 90), (width - 10, 260), (30, 30, 30), -1)
            cv2.rectangle(frame, (hud_x - 10, 90), (width - 10, 260), (100, 100, 100), 1)
            cv2.putText(frame, "MOVE CONTROLS:", (hud_x, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            
            moves = [
                ("I", "Forward"),
                ("L", "Backward"),
                ("X", "Jump"),
                ("Q", "Forward + Jump"),
                ("W", "Backward + Jump"),
            ]
            for idx, (g_letter, g_act) in enumerate(moves):
                y_pos = 140 + idx * 22
                is_active = (stable_letter == g_letter)
                color = (0, 255, 0) if is_active else (180, 180, 180)
                prefix = "> " if is_active else "  "
                cv2.putText(
                    frame,
                    f"{prefix}{g_letter} : {g_act}",
                    (hud_x, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2 if is_active else 1
                )

            # FPS
            curr_time = time.time()
            fps = 1.0 / max(curr_time - previous_time, 0.001)
            previous_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (width - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Hand Gesture Game Controller (5 Moves)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), ord("Q"), 27]: # 'q' or ESC
                break

    finally:
        sock.sendto(b"NONE", (UDP_IP, UDP_PORT))
        cap.release()
        cv2.destroyAllWindows()
        sock.close()
        print("\n[Controller] Closed successfully.")


if __name__ == "__main__":
    main()
