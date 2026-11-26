"""Data augmentation for the training images (spec 3.2). Owner: Model Architect.

Augmentation shows the model slightly changed copies of the training images, so it learns
the lesion and not the exact pixels. It is applied to the TRAINING set only: validation and
test images must stay real, or the scores would not reflect real use.
Each transformation must keep the label true:
- flips: a skin lesion has no "up" or "left", so a mirrored lesion has the same diagnosis;
- rotation: same reason, any angle is a valid photo of the lesion;
- zoom: the dermatoscope can be held a bit closer or further away;
- contrast: lighting differs between clinics and cameras.
"""
from tensorflow import keras


def build_augmentation(cfg):
    """Random augmentations. Used ONLY on the training set (spec 3.2 asks for at least two)."""
    a = cfg["data"]["augment"]
    layers = []
    if a.get("flip"):
        layers.append(keras.layers.RandomFlip("horizontal_and_vertical"))
    if a.get("rotation"):
        layers.append(keras.layers.RandomRotation(a["rotation"]))
    if a.get("zoom"):
        layers.append(keras.layers.RandomZoom(a["zoom"]))
    if a.get("contrast"):
        layers.append(keras.layers.RandomContrast(a["contrast"]))
    return keras.Sequential(layers, name="augmentation")
