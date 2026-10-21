"""Score images with a MEWC Keras classifier, preprocessing exactly as mewc-train does
(decode, bilinear resize to the model size, float 0-255). Run inside zaandahl/mewc-train:2.0.9.
The output axis follows LabelEncoder: class codes sorted as strings.
Usage: python vtl_score.py MODEL.keras CLASS_LIST_YAML PATHS_TXT OUT_PREFIX
Writes OUT_PREFIX.csv (path, top code and name, top probability) and OUT_PREFIX_probs.npy."""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "jax")
import keras
import numpy as np
import tensorflow as tf
import yaml

model_path, class_yaml, paths_txt, out_prefix = sys.argv[1:5]
cmap = {str(k): str(v).strip() for k, v in yaml.safe_load(open(class_yaml)).items()}
codes = sorted(cmap)
model = keras.saving.load_model(model_path, compile=False)
size = model.input_shape[1]
paths = [line.strip() for line in open(paths_txt) if line.strip()]


def load(p):
    img = tf.image.decode_png(tf.io.read_file(p), channels=3)
    return tf.image.resize(img, (size, size))


ds = tf.data.Dataset.from_tensor_slices(paths).map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(2)
probs = model.predict(ds, verbose=0)
assert probs.shape[1] == len(codes), f"model has {probs.shape[1]} outputs, class list has {len(codes)}"
np.save(out_prefix + "_probs.npy", probs.astype(np.float32))
top = probs.argmax(1)
with open(out_prefix + ".csv", "w") as f:
    f.write("path,top_code,top_name,top_prob\n")
    for p, t, pr in zip(paths, top, probs.max(1)):
        f.write(f"{p},{codes[t]},{cmap[codes[t]]},{pr:.4f}\n")
print(f"scored {len(paths)} images with {model_path} ({size}px, {len(codes)} classes)")
