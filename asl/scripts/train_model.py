from ultralytics import YOLO
import os


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "prepared_dataset"
)

RUNS_DIR = os.path.join(
    BASE_DIR,
    "runs"
)


# ============================================================
# SETTINGS
# ============================================================

RUN_NAME = "hand_letters_5"


# ============================================================
# CHECK DATASET
# ============================================================

print("=" * 60)
print("                 YOLO TRAINING")
print("=" * 60)

print("\nDataset:")
print(DATASET_DIR)


if not os.path.exists(DATASET_DIR):

    print(
        "\nERROR: prepared_dataset not found!"
    )

    print(
        "Run prepare_dataset.py first."
    )

    exit()


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading YOLO classification model...")

model = YOLO(
    "yolo11n-cls.pt"
)

print("YOLO model loaded!")


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")
print("Classes should be:")
print("I, L, Q, W, X")

results = model.train(

    data=DATASET_DIR,

    epochs=50,

    imgsz=224,

    batch=16,

    workers=0,

    device="mps",

    pretrained=True,

    project=RUNS_DIR,

    name=RUN_NAME,

    patience=10,

    verbose=True
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("              TRAINING COMPLETE")
print("=" * 60)

print("\nYour NEW model is:")

print(
    os.path.join(
        RUNS_DIR,
        RUN_NAME,
        "weights",
        "best.pt"
    )
)