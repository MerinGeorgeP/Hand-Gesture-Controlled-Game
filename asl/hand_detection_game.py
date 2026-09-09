#!/usr/bin/env python3
"""
Standalone 2D Hand Gesture Detection Game
-----------------------------------------
Interactive 2D Platformer controlled entirely by 5 ASL Hand Gestures:
  'I' -> Move Forward (Right)
  'L' -> Move Backward (Left)
  'X' -> Jump
  'Q' -> Move Forward + Jump
  'W' -> Move Backward + Jump

Features live webcam overlay, player physics, platforms, collectible cherries,
score tracking, victory goal, and real-time YOLO gesture detection.
"""

import os
import sys
import time
import math
from collections import deque, Counter
import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "runs", "hand_letters_5", "weights", "best.pt")

GAME_WIDTH = 1280
GAME_HEIGHT = 720
CONFIDENCE_THRESHOLD = 0.45
SMOOTHING_FRAMES = 5

# Physics & Player Constants
GRAVITY = 0.8
PLAYER_SPEED = 7.0
JUMP_FORCE = -15.5
PLAYER_WIDTH = 40
PLAYER_HEIGHT = 60

# Colors (BGR)
COLOR_BG = (45, 30, 20)
COLOR_GROUND = (60, 140, 70)
COLOR_PLATFORM = (100, 180, 90)
COLOR_PLAYER = (240, 140, 40)
COLOR_COLLECTABLE = (50, 50, 240) # Red Cherries
COLOR_GOAL = (0, 215, 255)       # Gold Flag
COLOR_HUD_BG = (20, 20, 20)
COLOR_WHITE = (255, 255, 255)

GESTURE_MAP = {
    "I": ("FORWARD", (0, 255, 0)),
    "L": ("BACKWARD", (255, 165, 0)),
    "X": ("JUMP", (255, 255, 0)),
    "Q": ("FORWARD + JUMP", (0, 255, 255)),
    "W": ("BACKWARD + JUMP", (255, 0, 255)),
    "NONE": ("IDLE", (150, 150, 150))
}


class Player:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = False
        self.facing_right = True

    def get_rect(self):
        return [self.x, self.y, PLAYER_WIDTH, PLAYER_HEIGHT]


def check_collision(rect1, rect2):
    # rect format: [x, y, w, h]
    return (rect1[0] < rect2[0] + rect2[2] and
            rect1[0] + rect1[2] > rect2[0] and
            rect1[1] < rect2[1] + rect2[3] and
            rect1[1] + rect1[3] > rect2[1])


def main():
    print("=" * 65)
    print("           STANDALONE HAND DETECTION GAME")
    print("=" * 65)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: YOLO model missing at: {MODEL_PATH}")
        sys.exit(1)

    print(f"\n[Model] Loading YOLO from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("[Model] Loaded successfully!")

    # --------------------------------------------------------
    # OPEN WEBCAM
    # --------------------------------------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        sys.exit(1)

    prediction_history = deque(maxlen=SMOOTHING_FRAMES)
    stable_letter = "NONE"

    # --------------------------------------------------------
    # GAME LEVEL SETUP
    # --------------------------------------------------------
    player = Player(100, 500)
    score = 0

    # Platforms: [x, y, w, h]
    platforms = [
        [0, 640, 1280, 80],      # Main Floor
        [200, 520, 180, 20],     # Platform 1
        [450, 420, 200, 20],     # Platform 2
        [720, 320, 180, 20],     # Platform 3
        [980, 220, 220, 20],     # High Platform 4
        [500, 200, 160, 20],     # Bonus Platform
    ]

    # Collectables (Cherries): [x, y, collected]
    collectables = [
        [250, 480, False],
        [520, 380, False],
        [800, 280, False],
        [1060, 180, False],
        [560, 160, False],
    ]

    # Goal (Finish Banner): [x, y, w, h]
    goal = [1120, 140, 40, 80]
    game_won = False

    previous_time = time.time()
    print("\nGame starting! Control character using your hand gestures:")
    print("  'I' = Forward  |  'L' = Backward  |  'X' = Jump")
    print("  'Q' = Forward + Jump  |  'W' = Backward + Jump")
    print("Press 'R' to reset level, 'Q' or 'ESC' to exit.")

    while True:
        # Read Webcam Frame
        success, cam_frame = cap.read()
        if not success:
            break

        cam_frame = cv2.flip(cam_frame, 1)
        c_h, c_w = cam_frame.shape[:2]

        # Crop Hand ROI (Center of webcam)
        roi_size = min(c_h, c_w) - 60
        rx1 = max(0, c_w // 2 - roi_size // 2)
        ry1 = max(0, c_h // 2 - roi_size // 2)
        rx2 = rx1 + roi_size
        ry2 = ry1 + roi_size

        cv2.rectangle(cam_frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        hand_crop = cam_frame[ry1:ry2, rx1:rx2]

        current_letter = "NONE"
        if hand_crop.size > 0:
            results = model.predict(hand_crop, imgsz=224, verbose=False)
            res = results[0]
            if res.probs is not None:
                conf = float(res.probs.top1conf)
                if conf >= CONFIDENCE_THRESHOLD:
                    current_letter = str(model.names[int(res.probs.top1)]).upper()

        prediction_history.append(current_letter)
        counts = Counter(prediction_history)
        best_lbl, count = counts.most_common(1)[0]
        stable_letter = best_lbl if count >= (SMOOTHING_FRAMES // 2 + 1) else "NONE"

        # --------------------------------------------------------
        # UPDATE GAME PHYSICS
        # --------------------------------------------------------
        if not game_won:
            # Gesture Actions
            move_right = (stable_letter in ["I", "Q"])
            move_left = (stable_letter in ["L", "W"])
            do_jump = (stable_letter in ["X", "Q", "W"])

            # Horizontal velocity
            if move_right:
                player.vx = PLAYER_SPEED
                player.facing_right = True
            elif move_left:
                player.vx = -PLAYER_SPEED
                player.facing_right = False
            else:
                player.vx = 0.0

            # Jump
            if do_jump and player.on_ground:
                player.vy = JUMP_FORCE
                player.on_ground = False

            # Gravity & Vertical Velocity
            player.vy += GRAVITY

            # Move X & Collide
            player.x += player.vx
            p_rect = player.get_rect()
            for plat in platforms:
                if check_collision(p_rect, plat):
                    if player.vx > 0:
                        player.x = plat[0] - PLAYER_WIDTH
                    elif player.vx < 0:
                        player.x = plat[0] + plat[2]

            # Keep inside horizontal bounds
            player.x = max(0, min(GAME_WIDTH - PLAYER_WIDTH, player.x))

            # Move Y & Collide
            player.y += player.vy
            player.on_ground = False
            p_rect = player.get_rect()

            for plat in platforms:
                if check_collision(p_rect, plat):
                    if player.vy > 0: # Landing on top
                        player.y = plat[1] - PLAYER_HEIGHT
                        player.vy = 0.0
                        player.on_ground = True
                    elif player.vy < 0: # Hitting ceiling
                        player.y = plat[1] + plat[3]
                        player.vy = 0.0

            # Collectables Collision
            p_rect = player.get_rect()
            for c in collectables:
                if not c[2]: # not collected
                    c_rect = [c[0] - 12, c[1] - 12, 24, 24]
                    if check_collision(p_rect, c_rect):
                        c[2] = True
                        score += 100

            # Goal Collision
            if check_collision(p_rect, goal):
                game_won = True

        # --------------------------------------------------------
        # RENDER GAME CANVAS
        # --------------------------------------------------------
        canvas = np.zeros((GAME_HEIGHT, GAME_WIDTH, 3), dtype=np.uint8)
        canvas[:] = COLOR_BG

        # Draw Platforms
        for plat in platforms:
            x, y, w, h = plat
            cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PLATFORM, -1)
            cv2.rectangle(canvas, (x, y), (x + w, y + 4), (140, 230, 120), -1) # Platform top accent

        # Draw Collectables (Cherries)
        for c in collectables:
            if not c[2]:
                cx, cy = c[0], c[1]
                cv2.circle(canvas, (cx - 5, cy), 8, COLOR_COLLECTABLE, -1)
                cv2.circle(canvas, (cx + 5, cy), 8, COLOR_COLLECTABLE, -1)
                cv2.line(canvas, (cx - 5, cy - 6), (cx, cy - 14), (30, 160, 40), 2)
                cv2.line(canvas, (cx + 5, cy - 6), (cx, cy - 14), (30, 160, 40), 2)

        # Draw Goal Flag
        gx, gy, gw, gh = goal
        cv2.rectangle(canvas, (gx, gy), (gx + 6, gy + gh), (200, 200, 200), -1) # Pole
        flag_pts = np.array([[gx + 6, gy], [gx + 36, gy + 18], [gx + 6, gy + 36]], np.int32)
        cv2.fillPoly(canvas, [flag_pts], COLOR_GOAL)

        # Draw Player Character
        px, py = int(player.x), int(player.y)
        cv2.rectangle(canvas, (px, py), (px + PLAYER_WIDTH, py + PLAYER_HEIGHT), COLOR_PLAYER, -1)
        cv2.rectangle(canvas, (px, py), (px + PLAYER_WIDTH, py + PLAYER_HEIGHT), (255, 255, 255), 2)

        # Player Eyes & Expression
        eye_x = px + (28 if player.facing_right else 12)
        cv2.circle(canvas, (eye_x, py + 18), 4, (255, 255, 255), -1)
        cv2.circle(canvas, (eye_x, py + 18), 2, (0, 0, 0), -1)

        # --------------------------------------------------------
        # HUD & WEBCAM PIP OVERLAY
        # --------------------------------------------------------
        # Top HUD Bar
        cv2.rectangle(canvas, (0, 0), (GAME_WIDTH, 70), COLOR_HUD_BG, -1)
        cv2.putText(canvas, f"SCORE: {score}", (30, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

        act_name, act_col = GESTURE_MAP.get(stable_letter, ("IDLE", (150, 150, 150)))
        cv2.putText(canvas, f"HAND GESTURE: {stable_letter}", (300, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WHITE, 2)
        cv2.putText(canvas, f"ACTION: {act_name}", (650, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, act_col, 3)

        # Draw PIP Camera in Bottom-Left Corner
        pip_w, pip_h = 280, 180
        pip_frame = cv2.resize(cam_frame, (pip_w, pip_h))
        canvas[GAME_HEIGHT - pip_h - 20 : GAME_HEIGHT - 20, 20 : 20 + pip_w] = pip_frame
        cv2.rectangle(canvas, (20, GAME_HEIGHT - pip_h - 20), (20 + pip_w, GAME_HEIGHT - 20), (0, 255, 0), 2)
        cv2.putText(canvas, "LIVE WEBCAM CONTROLLER", (30, GAME_HEIGHT - pip_h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # Controls Hint Card (Bottom-Right)
        cv2.rectangle(canvas, (GAME_WIDTH - 300, GAME_HEIGHT - 170), (GAME_WIDTH - 20, GAME_HEIGHT - 20), (30, 30, 30), -1)
        cv2.rectangle(canvas, (GAME_WIDTH - 300, GAME_HEIGHT - 170), (GAME_WIDTH - 20, GAME_HEIGHT - 20), (100, 100, 100), 1)
        cv2.putText(canvas, "5-MOVE CHEAT SHEET:", (GAME_WIDTH - 285, GAME_HEIGHT - 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        moves_list = [
            ("I", "Forward"),
            ("L", "Backward"),
            ("X", "Jump"),
            ("Q", "Forward + Jump"),
            ("W", "Backward + Jump")
        ]
        for idx, (l_str, a_str) in enumerate(moves_list):
            active = (stable_letter == l_str)
            col = (0, 255, 0) if active else (180, 180, 180)
            cv2.putText(canvas, f"{'>' if active else ' '} {l_str} : {a_str}", (GAME_WIDTH - 285, GAME_HEIGHT - 120 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2 if active else 1)

        # Victory Screen Banner
        if game_won:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (GAME_WIDTH, GAME_HEIGHT), (0, 0, 0), -1)
            canvas = cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0)

            cv2.putText(canvas, "LEVEL COMPLETED!", (GAME_WIDTH // 2 - 280, GAME_HEIGHT // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)
            cv2.putText(canvas, f"FINAL SCORE: {score}", (GAME_WIDTH // 2 - 150, GAME_HEIGHT // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_WHITE, 3)
            cv2.putText(canvas, "Press 'R' to play again or 'Q' to exit", (GAME_WIDTH // 2 - 240, GAME_HEIGHT // 2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        # Show Game Window
        cv2.imshow("Hand Detection Platformer Game (5 Moves)", canvas)

        # Process Keyboard Events
        key = cv2.waitKey(1) & 0xFF
        if key in [ord("q"), ord("Q"), 27]: # ESC or Q
            break
        elif key in [ord("r"), ord("R")]: # Restart Level
            player = Player(100, 500)
            score = 0
            game_won = False
            for c in collectables:
                c[2] = False

    cap.release()
    cv2.destroyAllWindows()
    print("\nGame exited cleanly.")


if __name__ == "__main__":
    main()
