"""Download HAM10000 into data/raw/ham10000/. Owner: Data Engineer.

Uses kagglehub, which downloads public Kaggle datasets (on Colab this works without
an API key; if it asks for one, follow https://github.com/Kaggle/kagglehub#authenticate).
"""
import shutil
from pathlib import Path

import kagglehub

TARGET = Path(__file__).resolve().parent / "raw" / "ham10000"


def main():
    """Download once and copy the metadata CSV and the two image folders into TARGET."""
    if (TARGET / "HAM10000_metadata.csv").exists():
        print("Already downloaded:", TARGET)
        return
    src = Path(kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000"))
    TARGET.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "HAM10000_metadata.csv", TARGET)
    for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        shutil.copytree(src / part, TARGET / part, dirs_exist_ok=True)
    print("Done:", TARGET)


if __name__ == "__main__":
    main()
