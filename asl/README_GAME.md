# Hand Detection Game (5 Move Control System)

An interactive hand detection game system that recognizes 5 American Sign Language (ASL) letter gestures in real time using a custom trained YOLO model (`best.pt`).

---

## Hand Gestures & Action Controls

| Hand Gesture | Letter | Action Description | Game Control |
| :--- | :---: | :--- | :--- |
| 🤙 / 🖐️ Pinky Finger | **`I`** | **Forward** | Move Character Right |
| 🪪 L-Shape | **`L`** | **Backward** | Move Character Left |
| ☝️ Hooked Index | **`X`** | **Jump** | Vertical Jump |
| 👈 Pointing Down | **`Q`** | **Forward & Jump** | Move Right + Jump simultaneously |
| 🤟 W-Shape | **`W`** | **Backward & Jump** | Move Left + Jump simultaneously |

---

## 🎮 How to Play

### Option 1: Standalone Python 2D Platformer Game (Quickest & Easiest)
Run the built-in 2D platformer game directly from your terminal:

```bash
asl/.venv_312/bin/python asl/hand_detection_game.py
```

- **Features**: Live picture-in-picture webcam HUD, player physics, jump mechanics, collectible cherries (+100 pts), goal flag, level restart (`R`), and win screen!

---

### Option 2: Play in Godot 2D Platformer Engine (`platform-game-(4.6)`)
Play using Godot 4.6 with UDP real-time gesture streaming:

1. **Step 1: Start the Webcam Controller**:
   ```bash
   asl/.venv_312/bin/python asl/webcam_game_controller.py
   ```
2. **Step 2: Launch Godot Platform Game**:
   - Open `platform-game-(4.6)` in Godot Engine and press **Play** (or run `godot platform-game-(4.6)/project.godot`).
   - The character will respond instantly to your hand gestures over UDP port `4242`!

---

## 📁 File Structure

- **`asl/webcam_game_controller.py`**: Webcam gesture tracking & UDP broadcaster (`127.0.0.1:4242`).
- **`asl/hand_detection_game.py`**: Full 2D Python platformer game with embedded webcam HUD.
- **`asl/runs/hand_letters_5/weights/best.pt`**: Pre-trained 99.9% accuracy YOLO classification model.
- **`platform-game-(4.6)/scenes/main_character.gd`**: Godot player script with UDP socket integration (`PacketPeerUDP`).
