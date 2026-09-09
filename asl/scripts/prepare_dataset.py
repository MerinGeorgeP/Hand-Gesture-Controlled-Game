import os
import shutil
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SOURCE_DIR = os.path.join(
    BASE_DIR,
    "gd dataset"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "prepared_dataset"
)


# ============================================================
# CLASSES
# ============================================================

CLASSES = [
    "I",
    "L",
    "Q",
    "W",
    "X"
]


# ============================================================
# SPLIT
# ============================================================

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


# ============================================================
# START
# ============================================================

print("=" * 60)
print("              PREPARING DATASET")
print("=" * 60)

print("\nSource:")
print(SOURCE_DIR)

print("\nOutput:")
print(OUTPUT_DIR)


# ============================================================
# CHECK SOURCE
# ============================================================

if not os.path.exists(SOURCE_DIR):

    print("\nERROR: gd dataset folder not found!")

    exit()


# ============================================================
# DELETE OLD PREPARED DATASET
# ============================================================

if os.path.exists(OUTPUT_DIR):

    print("\nDeleting old prepared_dataset...")

    shutil.rmtree(OUTPUT_DIR)


# ============================================================
# CREATE FOLDERS
# ============================================================

for split in [
    "train",
    "val",
    "test"
]:

    for cls in CLASSES:

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                split,
                cls
            ),
            exist_ok=True
        )


# ============================================================
# PROCESS EACH CLASS
# ============================================================

for cls in CLASSES:

    class_path = os.path.join(
        SOURCE_DIR,
        cls
    )

    if not os.path.exists(class_path):

        print(
            f"\nERROR: Missing class folder: {cls}"
        )

        continue


    images = [

        img

        for img in os.listdir(class_path)

        if img.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp"
            )
        )

    ]


    print(
        f"{cls}: {len(images)} images"
    )


    if len(images) < 10:

        print(
            f"WARNING: {cls} has very few images!"
        )

        continue


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_imgs, temp_imgs = train_test_split(

        images,

        test_size=0.20,

        random_state=42,

        shuffle=True

    )


    # --------------------------------------------------------
    # VALIDATION / TEST
    # --------------------------------------------------------

    val_imgs, test_imgs = train_test_split(

        temp_imgs,

        test_size=0.50,

        random_state=42,

        shuffle=True

    )


    splits = {

        "train": train_imgs,

        "val": val_imgs,

        "test": test_imgs

    }


    # --------------------------------------------------------
    # COPY
    # --------------------------------------------------------

    for split, img_list in splits.items():

        for img in img_list:

            src = os.path.join(
                class_path,
                img
            )

            dst = os.path.join(

                OUTPUT_DIR,

                split,

                cls,

                img

            )

            shutil.copy2(
                src,
                dst
            )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print("\nClasses:")

print(
    ", ".join(CLASSES)
)

print("\nPrepared dataset:")

print(OUTPUT_DIR)