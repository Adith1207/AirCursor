import os
import shutil
import random

def split_dataset(source_dir="gestures", dest_dir="dataset", split_ratio=0.8):
    train_dir = os.path.join(dest_dir, "train")
    val_dir = os.path.join(dest_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Loop through gesture folders
    for gesture in os.listdir(source_dir):
        gesture_path = os.path.join(source_dir, gesture)
        if not os.path.isdir(gesture_path):
            continue

        images = [f for f in os.listdir(gesture_path) if f.endswith(".jpg")]
        random.shuffle(images)

        split_index = int(len(images) * split_ratio)
        train_imgs = images[:split_index]
        val_imgs = images[split_index:]

        os.makedirs(os.path.join(train_dir, gesture), exist_ok=True)
        os.makedirs(os.path.join(val_dir, gesture), exist_ok=True)

        # Copy training images
        for img in train_imgs:
            shutil.copy(os.path.join(gesture_path, img),
                        os.path.join(train_dir, gesture, img))
        # Copy validation images
        for img in val_imgs:
            shutil.copy(os.path.join(gesture_path, img),
                        os.path.join(val_dir, gesture, img))

    print(f"✅ Dataset split done! Train/Val folders created at '{dest_dir}'.")


if __name__ == "__main__":
    split_dataset()
