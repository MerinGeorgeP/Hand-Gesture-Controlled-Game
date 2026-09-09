# Hand Gesture Controlled Game 🎮✋

A 2D game developed in Godot that allows players to control the character using **real-time hand gestures** instead of traditional keyboard controls.

The project combines **Computer Vision, Machine Learning, Python, UDP networking, and Godot game development** to create a hands-free gaming experience.

---


Traditional games usually rely on keyboards, controllers, or touchscreens for player interaction.

This project explores an alternative approach where the player can control a game character using predefined **hand gestures detected through a webcam**.

The webcam captures the player's hand, the gesture recognition system identifies the gesture, and the recognized command is sent to the Godot game through UDP communication.

### System Flow

```text
Webcam
   ↓
MediaPipe Hand Detection
   ↓
YOLO Gesture Classification
   ↓
Python Gesture Controller
   ↓
UDP Communication
   ↓
Godot Game
   ↓
Player Action
