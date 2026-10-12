"""Step 11: split by scene, not by tile, to avoid train/test leakage."""
import random


def split_scenes(scene_ids, ratios=(0.7, 0.15, 0.15), seed=0):
    assert abs(sum(ratios) - 1.0) < 1e-6
    ids = list(scene_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def verify_no_leakage(split_dict):
    seen = {}
    for split_name, ids in split_dict.items():
        for sid in ids:
            if sid in seen:
                raise ValueError(f"scene {sid} appears in both {seen[sid]} and {split_name}")
            seen[sid] = split_name
    return True
