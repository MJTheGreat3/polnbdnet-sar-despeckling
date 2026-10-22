"""One-off diagnostic for the fp16-NaN issue (report.md Next Steps #4).

Loads the real training data, builds the model under mixed_float16, and
runs a handful of training steps while checking every layer's output for
non-finite values after each step -- prints the FIRST layer (by name) whose
output goes non-finite, which is the actual overflow site. Cheap (a few
batches, not a real training run) -- meant to be run once on the GPU box to
localize the bug precisely instead of guessing.
"""
import glob
import os

import keras
import numpy as np
import tensorflow as tf

from .model import build_nbdnet

DATA_DIR = "pipeline/data/experiment"
TILE_SHAPE = (64, 64, 9)
N_STEPS = 50


def main():
    keras.mixed_precision.set_global_policy("mixed_float16")
    model = build_nbdnet(TILE_SHAPE)
    model.compile(optimizer=keras.optimizers.Adam(1e-3, clipnorm=1.0), loss="mse")

    paths = sorted(glob.glob(os.path.join(DATA_DIR, "train", "*.npz")))[:1]
    d = np.load(paths[0])
    inputs, targets = d["inputs"], d["targets"]
    print(f"using {paths[0]}: {inputs.shape[0]} tiles")

    probe_names = [l.name for l in model.layers if l.output.dtype in ("float16", "float32") and l.weights]
    probe_outputs = [model.get_layer(n).output for n in probe_names]
    probe_model = keras.Model(inputs=model.input, outputs=probe_outputs)

    rng = np.random.default_rng(0)
    batch_size = 32
    found = False
    for step in range(N_STEPS):
        idx = rng.choice(inputs.shape[0], size=batch_size, replace=False)
        x, y = inputs[idx], targets[idx]

        outs = probe_model(x, training=True)
        for name, out in zip(probe_names, outs):
            out_np = out.numpy() if hasattr(out, "numpy") else out
            if not np.isfinite(out_np).all():
                print(f"step {step}: FIRST non-finite output at layer '{name}' "
                      f"(dtype={out.dtype}), {np.isnan(out_np).sum()} nan / {np.isinf(out_np).sum()} inf "
                      f"out of {out_np.size}")
                found = True
                break
        if found:
            break

        model.train_on_batch(x, y)
        loss = model.evaluate(x, y, verbose=0)
        print(f"step {step}: loss={loss}")
        if not np.isfinite(loss):
            print(f"step {step}: loss went non-finite AFTER this step's forward/backward pass "
                  f"(no single probed layer was non-finite on the step's own forward pass above -- "
                  f"likely the backward pass / weight update, not a forward-pass activation overflow)")
            found = True
            break

    if not found:
        print(f"ran {N_STEPS} steps clean, no non-finite activations or loss -- "
              f"the issue may need more steps/epochs to reproduce, or is specific to a later LR "
              f"(ReduceLROnPlateau) or a particular batch further into training")


if __name__ == "__main__":
    main()
