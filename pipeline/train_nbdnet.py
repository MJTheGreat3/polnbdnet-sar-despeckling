"""Train NBDNet on the tiles built by prepare_training_data.py.

MSE loss between predicted and clean (C_ref) 9-channel representations,
Adam optimizer -- matches the paper's training setup (lr=0.001). Batch
size and epochs are CLI args since our dataset/hardware differ from the
paper's Colab run.
"""
import argparse
import json
import os
import time

import numpy as np
import keras

from .model import build_nbdnet

DATA_DIR = "pipeline/data/experiment"
RESULTS_DIR = "pipeline/results"


def load_split(name):
    d = np.load(os.path.join(DATA_DIR, f"{name}.npz"))
    return d["inputs"], d["targets"]


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

    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    print(f"train tiles: {x_train.shape}, val tiles: {x_val.shape}")

    model = build_nbdnet(x_train.shape[1:])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr), loss="mse")

    ckpt_path = os.path.join(RESULTS_DIR, "nbdnet_model.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ]

    t0 = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
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
