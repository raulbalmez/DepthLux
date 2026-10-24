"""Paired low/normal-light datasets: LOL-v1, LOL-v2-real, LOL-v2-synthetic.

Expected layout (<root> is the dataset folder itself):
  LOL-v1:   <root>/our485/{low,high}      <root>/eval15/{low,high}
  LOL-v2:   <root>/{Train,Test}/{Input,GT}   (or the original
            <root>/{Real_captured,Synthetic}/{Train,Test}/{Low,Normal})
"""
import os
import re
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

_DIRS = {  # dataset -> split -> (low, high); first existing layout wins
    "lolv1": {"train": ["our485/low", "lol_dataset/our485/low"],
              "test": ["eval15/low", "lol_dataset/eval15/low"]},
    "lolv2_real": {"train": ["Train/Input", "Real_captured/Train/Low"],
                   "test": ["Test/Input", "Real_captured/Test/Low"]},
    "lolv2_syn": {"train": ["Train/Input", "Synthetic/Train/Low"],
                  "test": ["Test/Input", "Synthetic/Test/Low"]},
}
_HIGH = {"low": "high", "Input": "GT", "Low": "Normal"}


def _pair(root, low_opts):
    for low_sub in low_opts:
        d = os.path.join(root, low_sub)
        if os.path.isdir(d):
            head, tail = os.path.split(low_sub)
            return d, os.path.join(root, head, _HIGH[tail])
    raise FileNotFoundError(f"none of {low_opts} under {root}")


def _key(f):
    # low00001 / normal00001 / r00816405t / 00816405t  ->  numeric key
    return re.sub(r"^[A-Za-z]+", "", os.path.splitext(f)[0])


def _list_pairs(root, dataset, split):
    ld, hd = _pair(root, _DIRS[dataset][split])
    ext = (".png", ".jpg", ".jpeg", ".bmp")
    low = {_key(f): f for f in os.listdir(ld) if f.lower().endswith(ext)}
    high = {_key(f): f for f in os.listdir(hd) if f.lower().endswith(ext)}
    keys = sorted(set(low) & set(high))
    assert keys and len(keys) == len(low) == len(high), "low/high mismatch"
    return ([os.path.join(ld, low[k]) for k in keys],
            [os.path.join(hd, high[k]) for k in keys])


def _read(path):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3,
                             expand_animations=False)
    return tf.image.convert_image_dtype(img, tf.float32)


def make_dataset(root, dataset, split, batch_size=1, patch_size=256,
                 shuffle=True):
    """Train: random crop + random flips. Test: full images, batch of 1."""
    low, high = _list_pairs(root, dataset, split)
    ds = tf.data.Dataset.from_tensor_slices((low, high))
    if split == "train":
        if shuffle:
            ds = ds.shuffle(len(low), reshuffle_each_iteration=True)
        ds = ds.repeat()

    def load(l, h):
        return _read(l), _read(h)

    def augment(l, h):
        both = tf.concat([l, h], axis=-1)
        both = tf.image.random_crop(both, [patch_size, patch_size, 6])
        both = tf.image.random_flip_left_right(both)
        both = tf.image.random_flip_up_down(both)
        return both[..., :3], both[..., 3:]

    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    if split == "train":
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE).batch(batch_size)
    else:
        ds = ds.batch(1)
    return ds.prefetch(AUTOTUNE), len(low)
