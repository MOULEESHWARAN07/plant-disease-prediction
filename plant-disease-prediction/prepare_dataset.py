from pathlib import Path
import random
import shutil

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

# PlantVillage folder names -> simple project labels
CLASS_MAP = {
    "Tomato___healthy": "healthy",
    "Tomato___Early_blight": "early_blight",
    "Tomato___Late_blight": "late_blight",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

def find_source_folder(name):
    """Find a class folder either directly under data/raw or one level below."""
    direct = RAW_DIR / name
    if direct.exists():
        return direct

    matches = list(RAW_DIR.rglob(name))
    return matches[0] if matches else None

def copy_split(files, label):
    random.shuffle(files)
    n = len(files)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:],
    }

    for split, split_files in splits.items():
        target = OUT_DIR / split / label
        target.mkdir(parents=True, exist_ok=True)

        # Clear previous generated images
        for old_file in target.iterdir():
            if old_file.is_file() and old_file.suffix in IMAGE_EXTENSIONS:
                old_file.unlink()

        for i, src in enumerate(split_files):
            dst = target / f"{label}_{i:05d}{src.suffix.lower()}"
            shutil.copy2(src, dst)

    return {k: len(v) for k, v in splits.items()}

def main():
    random.seed(SEED)
    print("Preparing the 3-class tomato leaf dataset...\n")

    total = 0
    for source_name, label in CLASS_MAP.items():
        source = find_source_folder(source_name)

        if source is None:
            print(f"[MISSING] {source_name}")
            continue

        files = [
            p for p in source.rglob("*")
            if p.is_file() and p.suffix in IMAGE_EXTENSIONS
        ]

        counts = copy_split(files, label)
        total += len(files)
        print(f"[OK] {source_name}: {len(files)} images -> {counts}")

    if total == 0:
        print("\nNo images were found.")
        print("Put the PlantVillage class folders inside data/raw/ and run this script again.")
        return

    print(f"\nTotal images copied: {total}")
    print("Dataset is ready in data/processed/")

if __name__ == "__main__":
    main()
