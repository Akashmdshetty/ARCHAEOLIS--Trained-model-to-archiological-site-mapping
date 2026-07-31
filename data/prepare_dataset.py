"""
prepare_dataset.py
------------------
Reads raw satellite JPGs from a source folder, validates, resizes to 224x224,
and saves to data/processed/ ready for BYOL training.
"""
import os
import shutil
from tqdm import tqdm
import yaml
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# ── Configuration ───────────────────────────────────────────────────────────
RAW_IMAGES_DIR  = r"C:\Users\aakas\Downloads\images"   # primary JPGs
PROCESSED_DIR   = "data/processed"
IMG_SIZE        = (224, 224)
# ─────────────────────────────────────────────────────────────────────────────


def prepare_from_folder(raw_dir: str = None, processed_dir: str = PROCESSED_DIR, img_size=IMG_SIZE):
    """
    Reads all .jpg/.png images from raw_dir, validates them, resizes to img_size,
    and saves to processed_dir.
    """
    if raw_dir is None or not os.path.isdir(raw_dir):
        if os.path.isdir(RAW_IMAGES_DIR):
            raw_dir = RAW_IMAGES_DIR
        elif os.path.exists("configs/config.yaml"):
            with open("configs/config.yaml", "r") as f:
                cfg = yaml.safe_load(f)
            raw_dir = cfg.get("dataset", {}).get("raw_images_dir", "data/raw_images")
        else:
            raw_dir = "data/raw_images"

    if not os.path.isdir(raw_dir):
        print(f"[ERROR] Source folder not found: {raw_dir}")
        return

    # Collect image paths
    valid_exts = ('.jpg', '.jpeg', '.png')
    all_files = [
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(valid_exts)
    ]

    if not all_files:
        print(f"[ERROR] No images found in {raw_dir}")
        return

    print(f"Found {len(all_files)} image(s) in {raw_dir}")

    # Recreate processed directory
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)

    valid_count = 0
    corrupted_count = 0

    for fname in tqdm(all_files, desc="Processing images"):
        src_path = os.path.join(raw_dir, fname)
        try:
            # Validate
            with Image.open(src_path) as img:
                img.verify()

            # Resize and save
            with Image.open(src_path) as img:
                img = img.convert('RGB')
                img = img.resize(img_size, Image.Resampling.LANCZOS)
                # Always save as .jpg for consistency
                out_name = os.path.splitext(fname)[0] + ".jpg"
                img.save(os.path.join(processed_dir, out_name), quality=95)
                valid_count += 1

        except Exception as e:
            corrupted_count += 1
            print(f"[WARN] Skipping corrupted image {fname}: {e}")

    print("\n" + "=" * 40)
    print("Dataset Preparation Complete")
    print("=" * 40)
    print(f"  Source folder      : {raw_dir}")
    print(f"  Total found        : {len(all_files)}")
    print(f"  Successfully saved : {valid_count}")
    print(f"  Corrupted/skipped  : {corrupted_count}")
    print(f"  Saved to           : {os.path.abspath(processed_dir)}")
    print(f"  Image size         : {img_size[0]}x{img_size[1]}")
    train_n = int(0.9 * valid_count)
    print(f"  Train/Val split    : ~{train_n} / ~{valid_count - train_n}")
    print("=" * 40)


if __name__ == "__main__":
    prepare_from_folder(RAW_IMAGES_DIR, PROCESSED_DIR, IMG_SIZE)
