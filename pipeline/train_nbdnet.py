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
import datetime
import glob
import json
import os
import shutil
import time

import numpy as np
import keras
import tensorflow as tf

from .model import build_nbdnet
from .pol_nbdnet import enable_gpu_memory_growth, repr_to_cov_tf, wishart_distance_tf

enable_gpu_memory_growth()  # must run before any TF op touches the GPU -- see docstring

DATA_DIR = "pipeline/data/experiment"
RESULTS_DIR = "pipeline/results"
# Every run writes the canonical pipeline/results/{nbdnet_model.keras,train_history.json}
# too (evaluate_nbdnet.py/make_before_after.py/stitch_scene.py hardcode those paths and
# always read "the latest run") -- RUNS_DIR is a parallel, never-overwritten archive per
# run (model + history + the hyperparams used) so multiple training attempts can be
# compared later instead of each one clobbering the last.
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")
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
    """Per-shard tf.py_function + from_tensor_slices + interleave, instead
    of a from_generator yielding one tile at a time -- from_generator pays
    a python<->TF boundary crossing PER TILE (499720/epoch), which was the
    actual step-time bottleneck (measured 163-203ms/step), not GPU compute.
    This pays that crossing once per SHARD (one np.load() call), and lets
    TF slice/batch/interleave the resulting arrays in C++. cycle_length
    bounds how many shards are held in memory at once, keeping the
    bounded-by-disk-not-RAM property from the old generator (see module
    docstring) instead of loading the whole split at once."""
    paths = shard_paths(split)

    def load_shard(path_t):
        path = path_t.numpy().decode("utf-8")
        scene_id = os.path.splitext(os.path.basename(path))[0]
        db_min, db_max = per_scene_norm[scene_id]
        d = np.load(path)
        inputs, targets = d["inputs"], d["targets"]
        norm_cols = np.empty(targets.shape[:-1] + (2,), dtype=np.float32)
        norm_cols[..., 0], norm_cols[..., 1] = db_min, db_max
        targets_ext = np.concatenate([targets, norm_cols], axis=-1)
        return inputs.astype(np.float32), targets_ext.astype(np.float32)

    def shard_dataset(path_t):
        inputs, targets_ext = tf.py_function(load_shard, [path_t], [tf.float32, tf.float32])
        inputs.set_shape([None] + list(TILE_SHAPE))
        targets_ext.set_shape([None] + list(TARGET_SHAPE))
        shard_ds = tf.data.Dataset.from_tensor_slices((inputs, targets_ext))
        return shard_ds.shuffle(SHUFFLE_BUFFER) if shuffle else shard_ds

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        path_ds = path_ds.shuffle(len(paths), reshuffle_each_iteration=True)
    # cycle_length=4/num_parallel_calls=2 caps peak RAM to ~2 shards' full
    # arrays in flight (~2.5GB) at once -- AUTOTUNE let this run further
    # than expected and got OOM-killed by run.sh's MemoryMax cgroup cap on
    # a first attempt (report.md: 2026-11-25 new-box migration), so this is
    # a fixed, predictable bound instead of letting the tuner pick it.
    cycle_length = 4 if shuffle else 1  # val doesn't need parallel shard interleave
    ds = path_ds.interleave(
        shard_dataset,
        cycle_length=cycle_length,
        num_parallel_calls=2 if shuffle else 1,
        deterministic=not shuffle,
    )
    if shuffle:
        ds = ds.shuffle(SHUFFLE_BUFFER, reshuffle_each_iteration=True)
    # .repeat() + explicit steps_per_epoch (passed by the caller, from
    # meta.json's known tile counts) instead of letting Keras infer epoch
    # boundaries from StopIteration -- an interleaved dataset built from
    # py_function has unknown cardinality, so without this Keras logs a
    # (harmless but alarming) "input ran out of data; interrupting
    # training" warning every epoch.
    ds = ds.repeat()
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), len(paths)


def make_scene_dataset(path, batch_size, per_scene_norm):
    """Single val scene, no shuffle/repeat -- used by PerSceneValLoss below
    to break the combined val_loss apart per scene, since that average can
    hide one bad-fit scene driving an overfitting spike (report.md:
    2026-11-22 retrain investigation, bermsr+barrow_val1 pooled val_loss)."""
    scene_id = os.path.splitext(os.path.basename(path))[0]
    db_min, db_max = per_scene_norm[scene_id]
    d = np.load(path)
    inputs, targets = d["inputs"], d["targets"]
    norm_cols = np.empty(targets.shape[:-1] + (2,), dtype=np.float32)
    norm_cols[..., 0], norm_cols[..., 1] = db_min, db_max
    targets_ext = np.concatenate([targets, norm_cols], axis=-1)
    ds = tf.data.Dataset.from_tensor_slices((inputs, targets_ext)).batch(batch_size)
    return ds, scene_id


class PerSceneValLoss(keras.callbacks.Callback):
    """Logs per-val-scene loss every epoch alongside the pooled val_loss
    Keras already reports. Stored in train_history.json under
    val_loss_by_scene (scene_id -> list, one entry per epoch)."""
    def __init__(self, val_paths, batch_size, per_scene_norm):
        super().__init__()
        self.val_paths = val_paths
        self.batch_size = batch_size
        self.per_scene_norm = per_scene_norm
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        parts = []
        for path in self.val_paths:
            ds, scene_id = make_scene_dataset(path, self.batch_size, self.per_scene_norm)
            loss = self.model.evaluate(ds, verbose=0)
            self.history.setdefault(scene_id, []).append(float(loss))
            parts.append(f"{scene_id}={loss:.5f}")
        print("  per-scene val_loss: " + ", ".join(parts))


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
    p.add_argument("--run-name", default=None,
                    help="tag for this run's archived copy under pipeline/results/runs/<run-name>/ "
                         "(model + history + the hyperparams used) -- default is a timestamp + wishart "
                         "weight, e.g. 20261105_143200_w0.1. Doesn't affect the canonical "
                         "pipeline/results/nbdnet_model.keras that other scripts read.")
    args = p.parse_args()

    run_id = args.run_name or (
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_w{args.wishart_weight}"
    )
    run_dir = os.path.join(RUNS_DIR, run_id)

    if args.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    # written up front (not just at the end) so the hyperparams survive even if the
    # process dies outright (OOM-kill etc.) partway through, not just a clean exit
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

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

    # clipnorm is a safety net against a gradient spike turning the loss to
    # NaN. Originally scoped to --mixed-precision only (fp16 activation
    # overflow, epoch 3 of the first attempt) -- also needed for
    # --wishart-weight > 0: the Wishart term's gradient gets steep near
    # ill-conditioned covariance matrices, and even with repr_to_cov_tf's
    # [0,1]/|gamma|<=1 clamping (pol_nbdnet.py) on the *representation*, nothing
    # bounds how far raw network weights drift step to step -- a real run
    # reached "Invalid loss, terminating training" (loss: nan) at batch 1825
    # without this. Left off for plain-MSE fp32 to not disturb the
    # already-reported baseline numbers.
    opt_kwargs = {"learning_rate": args.lr}
    if args.mixed_precision or args.wishart_weight > 0:
        opt_kwargs["clipnorm"] = 1.0
    model = build_nbdnet(TILE_SHAPE)
    model.compile(optimizer=keras.optimizers.Adam(**opt_kwargs), loss=make_loss(args.wishart_weight))

    ckpt_path = os.path.join(RESULTS_DIR, "nbdnet_model.keras")
    per_scene_val_cb = PerSceneValLoss(shard_paths("val"), args.batch_size, per_scene_norm)
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.TerminateOnNaN(),  # fail fast instead of burning the rest of the run on NaN
        per_scene_val_cb,
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
    hist["val_loss_by_scene"] = per_scene_val_cb.history
    with open(os.path.join(RESULTS_DIR, "train_history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    model.save(ckpt_path)

    # archive this run's model + history alongside run_config.json (written earlier) --
    # the canonical RESULTS_DIR files above still get overwritten every run as before,
    # this is the non-overwritten copy for comparing runs later.
    with open(os.path.join(run_dir, "train_history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    shutil.copy2(ckpt_path, os.path.join(run_dir, "nbdnet_model.keras"))
    print(f"archived this run to {run_dir}/")

    print(f"done in {elapsed:.1f}s, {hist['epochs_ran']} epochs, "
          f"final val_loss={hist['val_loss'][-1]:.5f} (best saved to {ckpt_path})")


if __name__ == "__main__":
    main()
