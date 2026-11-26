"""Single inference entry point (spec 3.7). Owner: Evaluation & Deployment Engineer.

Usage (after the final model is saved in milestone 4):
    python -m src.predict path/to/image.jpg
"""
import argparse
import time

import numpy as np
from tensorflow import keras

from src.config import ROOT, load_config
from src.data import load_image


def predict_image(path, cfg, model):
    """Same preprocessing as training (load_image), then return (class name, confidence)."""
    x = load_image(path, cfg["data"]["image_size"])[None, ...]   # add batch dimension
    proba = model.predict(x, verbose=0)[0]
    i = int(np.argmax(proba))
    return cfg["data"]["classes"][i], float(proba[i])


def main():
    """CLI: print the prediction and the inference time."""
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model = keras.models.load_model(ROOT / "models" / "final_model.keras")
    start = time.perf_counter()
    label, conf = predict_image(args.image, cfg, model)
    print(f"{label} ({conf:.1%})  in {1000 * (time.perf_counter() - start):.0f} ms")


if __name__ == "__main__":
    main()
