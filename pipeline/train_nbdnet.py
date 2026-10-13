"""Train NBDNet on the tiles built by prepare_training_data.py.

MSE loss between predicted and clean (C_ref) 9-channel representations,
Adam optimizer -- matches the paper's training setup (lr=0.001). Batch
size and epochs are CLI args since our dataset/hardware differ from the
paper's Colab run.

Reads per-scene shards (pipeline/data/experiment/{train,val}/<scene>.npz)
via a streaming tf.data pipeline instead of loading the whole split into
RAM: one shard is held in memory at a time plus a shuffle buffer, so the
training set size is bounded by disk, not RAM -- the thing that actually
lets you add more scenes on a bigger box without hitting a wall.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import keras
import tensorflow as tf

from .model import build_nbdnet

DATA_DIR = "pipeline/data/experiment"
RESULTS_DIR = "pipeline/results"
TILE_SHAPE = (64, 64, 9)
SHUFFLE_BUFFER = 2000  # tiles (~2000 * 0.27MB =~ 540MB) -- within-epoch shuffle window, not the whole split


def shard_paths(split):
    paths = sorted(glob.glob(os.path.join(DATA_DIR, split, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"no shards in {DATA_DIR}/{split}/ -- run prepare_training_data.py first")
    return paths


def make_dataset(split, batch_size, shuffle):
    paths = shard_paths(split)

    def gen():
        for path in paths:
            d = np.load(path)
            inputs, targets = d["inputs"], d["targets"]
            idx = np.random.permutation(len(inputs)) if shuffle else np.arange(len(inputs))
            for i in idx:
                yield inputs[i], targets[i]

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=TILE_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=TILE_SHAPE, dtype=tf.float32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(SHUFFLE_BUFFER, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), len(paths)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--mixed-precision", action="store_true",
                    help="float16 compute on CUDA GPUs with tensor cores -- faster/bigger batches on a "
                         "bigger remote box; not meaningful on Apple Metal")
    args = p.parse_args()

    if args.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)
    tile_counts = meta["tile_counts"]

    train_ds, n_train_shards = make_dataset("train", args.batch_size, shuffle=True)
    val_ds, n_val_shards = make_dataset("val", args.batch_size, shuffle=False)
    n_train_tiles = sum(tile_counts[sid] for sid in meta["scenes"]["train"])
    n_val_tiles = sum(tile_counts[sid] for sid in meta["scenes"]["val"])
    print(f"train: {n_train_tiles} tiles from {n_train_shards} scene(s), "
          f"val: {n_val_tiles} tiles from {n_val_shards} scene(s)")

    model = build_nbdnet(TILE_SHAPE)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr), loss="mse")

    ckpt_path = os.path.join(RESULTS_DIR, "nbdnet_model.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ]

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - t0

    hist = {k: [float(v) for v in vs] for k, vs in history.history.items()}
    hist["elapsed_seconds"] = round(elapsed, 1)
    hist["epochs_ran"] = len(history.history["loss"])
    with open(os.path.join(RESULTS_DIR, "train_history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    model.save(ckpt_path)
    print(f"done in {elapsed:.1f}s, {hist['epochs_ran']} epochs, "
          f"final val_loss={hist['val_loss'][-1]:.5f} (best saved to {ckpt_path})")


if __name__ == "__main__":
    main()
