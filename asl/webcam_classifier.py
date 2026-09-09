import cv2
import os
import time
from collections import deque, Counter
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# Use the NEW model after retraining.
MODEL_PATH = "runs/hand_letters_5/weights/best.pt"

CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.50

# Number of frames used for stable prediction
SMOOTHING_FRAMES = 7

# Size of hand area on screen
ROI_SIZE = 500


# ============================================================
# HEADER
# ============================================================

print("=" * 65)
print("                 HAND GESTURE CLASSIFIER")
print("=" * 65)


# ============================================================
# CHECK MODEL
# ============================================================

print("\nYOLO model:")
print(os.path.abspath(MODEL_PATH))

if not os.path.exists(MODEL_PATH):

    print("\nERROR: Model not found!")
    print("\nExpected:")
    print(os.path.abspath(MODEL_PATH))

    print("\nMake sure you have trained the model first.")

    input("\nPress Enter to exit...")
    exit()


# ============================================================
# LOAD YOLO
# ============================================================

print("\nLoading YOLO model...")

try:

    model = YOLO(MODEL_PATH)

except Exception as e:

    print("\nERROR loading YOLO:")
    print(e)

    input("\nPress Enter to exit...")
    exit()


print("YOLO loaded successfully!")

print("\nYOLO classes:")
print(model.names)


# ============================================================
# OPEN CAMERA
# ============================================================

print("\nOpening webcam...")

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    print("\nERROR: Could not open webcam.")

    input("\nPress Enter to exit...")
    exit()


# ============================================================
# CAMERA SETTINGS
# ============================================================



cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

print("[Webcam] Camera opened successfully!")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ============================================================
# PREDICTION HISTORY
# ============================================================

prediction_history = deque(
    maxlen=SMOOTHING_FRAMES
)

stable_letter = "NONE"
stable_confidence = 0.0


# ============================================================
# FPS
# ============================================================

previous_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    success, frame = cap.read()

    if not success:

        print("Failed to read webcam frame.")

        break


    # --------------------------------------------------------
    # MIRROR CAMERA
    # --------------------------------------------------------

    frame = cv2.flip(frame, 1)


    height, width = frame.shape[:2]


    # ========================================================
    # CREATE CENTER HAND BOX
    # ========================================================

    center_x = width // 2
    center_y = height // 2

    x1 = center_x - ROI_SIZE // 2
    y1 = center_y - ROI_SIZE // 2

    x2 = center_x + ROI_SIZE // 2
    y2 = center_y + ROI_SIZE // 2


    # Make sure coordinates are valid

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(width, x2)
    y2 = min(height, y2)


    # ========================================================
    # DRAW HAND BOX
    # ========================================================

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        3
    )


    cv2.putText(
        frame,
        "Place hand inside box",
        (x1, y1 - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # CROP ONLY HAND AREA
    # ========================================================

    hand_crop = frame[
        y1:y2,
        x1:x2
    ]


    # ========================================================
    # YOLO PREDICTION
    # ========================================================

    if hand_crop.size > 0:

        results = model.predict(
            hand_crop,
            imgsz=224,
            verbose=False
        )

        result = results[0]


        # ====================================================
        # CLASSIFICATION MODEL
        # ====================================================

        if result.probs is not None:

            # Best class
            class_id = int(
                result.probs.top1
            )

            confidence = float(
                result.probs.top1conf
            )

            label = str(
                model.names[class_id]
            )


            # =================================================
            # CONFIDENCE FILTER
            # =================================================

            if confidence >= CONFIDENCE_THRESHOLD:

                prediction_history.append(label)


                # =============================================
                # MAJORITY VOTE
                # =============================================

                if len(prediction_history) >= 3:

                    counts = Counter(
                        prediction_history
                    )

                    best_label, count = (
                        counts.most_common(1)[0]
                    )

                    stable_letter = best_label

                    stable_confidence = confidence


            else:

                prediction_history.clear()

        else:

            # This model should be a classification model.
            print(
                "WARNING: Model is not a classification model."
            )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    if stable_letter != "NONE":

        cv2.putText(
            frame,
            f"LETTER: {stable_letter}",
            (30, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 255, 0),
            4
        )

        cv2.putText(
            frame,
            f"Confidence: "
            f"{stable_confidence * 100:.1f}%",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "LETTER: NONE",
            (30, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (0, 0, 255),
            3
        )


    # ========================================================
    # SHOW TOP PREDICTIONS
    # ========================================================

    if hand_crop.size > 0 and result.probs is not None:

        top5 = result.probs.top5

        y_position = 170

        cv2.putText(
            frame,
            "Top predictions:",
            (30, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        y_position += 30

        for i, class_id in enumerate(top5):

            class_id = int(class_id)

            probability = float(
                result.probs.data[class_id]
            )

            class_name = str(
                model.names[class_id]
            )

            text = (
                f"{class_name}: "
                f"{probability * 100:.1f}%"
            )

            cv2.putText(
                frame,
                text,
                (30, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            y_position += 25


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    fps = 1 / max(
        current_time - previous_time,
        0.001
    )

    previous_time = current_time

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (width - 150, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    cv2.putText(
        frame,
        "Q = Quit",
        (30, height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Hand Gesture Classifier",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\nGesture recognition stopped.")