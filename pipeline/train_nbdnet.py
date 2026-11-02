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
from .pol_nbdnet import repr_to_cov_tf, wishart_distance_tf

DATA_DIR = "pipeline/data/experiment"
RESULTS_DIR = "pipeline/results"
TILE_SHAPE = (64, 64, 9)
# targets carry 2 extra channels (per-tile db_min/db_max, broadcast constants)
# beyond the 9 repr channels -- only consumed by the Wishart loss term
# (--wishart-weight > 0) to decode the physically-scaled covariance; plain
# MSE just slices off the first 9 and ignores them.
TARGET_SHAPE = (64, 64, 11)
SHUFFLE_BUFFER = 2000  # tiles (~2000 * 0.27MB =~ 540MB) -- within-epoch shuffle window, not the whole split


def shard_paths(split):
    paths = sorted(glob.glob(os.path.join(DATA_DIR, split, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"no shards in {DATA_DIR}/{split}/ -- run prepare_training_data.py first")
    return paths


def make_dataset(split, batch_size, shuffle, per_scene_norm):
    paths = shard_paths(split)

    def gen():
        for path in paths:
            scene_id = os.path.splitext(os.path.basename(path))[0]
            db_min, db_max = per_scene_norm[scene_id]
            d = np.load(path)
            inputs, targets = d["inputs"], d["targets"]
            norm_cols = np.empty(targets.shape[:-1] + (2,), dtype=np.float32)
            norm_cols[..., 0], norm_cols[..., 1] = db_min, db_max
            targets_ext = np.concatenate([targets, norm_cols], axis=-1)
            idx = np.random.permutation(len(inputs)) if shuffle else np.arange(len(inputs))
            for i in idx:
                yield inputs[i], targets_ext[i]

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=TILE_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=TARGET_SHAPE, dtype=tf.float32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(SHUFFLE_BUFFER, reshuffle_each_iteration=True)
    # .repeat() + explicit steps_per_epoch (passed by the caller, from
    # meta.json's known tile counts) instead of letting Keras infer epoch
    # boundaries from StopIteration -- a from_generator dataset has unknown
    # cardinality, so without this Keras logs a (harmless but alarming)
    # "input ran out of data; interrupting training" warning every epoch.
    ds = ds.repeat()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), len(paths)


def make_loss(wishart_weight, eps=1e-6):
    """MSE on the 9-channel repr, optionally plus wishart_weight * the
    Wishart (log-det) divergence between the reconstructed covariances
    (polarimetric roadmap Step 3) -- trains cross-channel structure, not
    just independent per-channel pixel values. wishart_weight=0 (default)
    keeps the original MSE-only behavior, so this is backward-compatible
    unless explicitly opted into via --wishart-weight."""
    def loss_fn(y_true, y_pred):
        targets = tf.cast(y_true[..., :9], tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mse = tf.reduce_mean(tf.square(targets - y_pred))
        if wishart_weight <= 0:
            return mse
        db_min, db_max = y_true[..., 9:10], y_true[..., 10:11]
        C_pred = repr_to_cov_tf(y_pred, db_min, db_max)
        C_true = repr_to_cov_tf(targets, db_min, db_max)
        wishart = tf.reduce_mean(wishart_distance_tf(C_pred, C_true, eps=eps))
        return mse + wishart_weight * wishart
    return loss_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--mixed-precision", action="store_true",
                    help="float16 compute on CUDA GPUs with tensor cores -- faster/bigger batches on a "
                         "bigger remote box; not meaningful on Apple Metal")
    p.add_argument("--wishart-weight", type=float, default=0.0,
                    help="polarimetric roadmap Step 3: weight of the Wishart-divergence loss term added "
                         "to MSE (0 = disabled, original MSE-only behavior). Doc recommends starting small "
                         "(e.g. 0.05-0.1) and ramping up while watching Step 1's polarimetric eval metrics.")
    args = p.parse_args()

    if args.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)
    tile_counts = meta["tile_counts"]

    per_scene_norm = meta["per_scene_norm"]
    train_ds, n_train_shards = make_dataset("train", args.batch_size, shuffle=True, per_scene_norm=per_scene_norm)
    val_ds, n_val_shards = make_dataset("val", args.batch_size, shuffle=False, per_scene_norm=per_scene_norm)
    n_train_tiles = sum(tile_counts[sid] for sid in meta["scenes"]["train"])
    n_val_tiles = sum(tile_counts[sid] for sid in meta["scenes"]["val"])
    steps_per_epoch = -(-n_train_tiles // args.batch_size)  # ceil div
    validation_steps = -(-n_val_tiles // args.batch_size)
    print(f"train: {n_train_tiles} tiles from {n_train_shards} scene(s) ({steps_per_epoch} steps/epoch), "
          f"val: {n_val_tiles} tiles from {n_val_shards} scene(s) ({validation_steps} steps)")

    # clipnorm is scoped to --mixed-precision only -- it's a safety net against
    # a late-training activation/gradient spike overflowing fp16's narrow range
    # (the loss-scale optimizer only guards gradient under/overflow, not a
    # forward-pass activation blowing past fp16 max and turning the loss
    # itself to NaN, which is what happened at epoch 3 of the first attempt).
    # Left off for fp32 to not disturb the already-reported baseline numbers.
    opt_kwargs = {"learning_rate": args.lr}
    if args.mixed_precision:
        opt_kwargs["clipnorm"] = 1.0
    model = build_nbdnet(TILE_SHAPE)
    model.compile(optimizer=keras.optimizers.Adam(**opt_kwargs), loss=make_loss(args.wishart_weight))

    ckpt_path = os.path.join(RESULTS_DIR, "nbdnet_model.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.TerminateOnNaN(),  # fail fast instead of burning the rest of the run on NaN
    ]

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=args.epochs,
        callbacks=callbacks,
        shuffle=False,  # already shuffled in make_dataset(); avoids a harmless but noisy fit() warning
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
