"""Amazon Beauty Reviews dataset loader for TokenMixer-Large.

The Amazon Beauty Reviews log
(jhan21___amazon-beauty-reviews-dataset, ~701k rows) is reframed as a
binary CTR-style ranking task:

    label = 1 if rating >= POSITIVE_RATING_THRESHOLD else 0

Features used:
    Sparse (4): user_id, asin, parent_asin, verified_purchase
        - Strings are hashed to fixed buckets via
          tf.strings.to_hash_bucket_fast (collisions accepted, no offline
          vocabulary needed).
    Dense  (5):
        helpful_vote (log1p),
        timestamp (min-max scaled to [0, 1]),
        text_length  = log1p(len(text))    — review effort signal,
        title_length = log1p(len(title))   — review headline signal,
        has_images   = 1 if `images` != '[]' else 0.

The output schema mirrors criteo_kaggle_dataset.py so the train loop only
needs to swap the loader and the per-dataset feature dimensions.
"""

from typing import List

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import tensorflow as tf


USER_BUCKETS = 400_000
ASIN_BUCKETS = 100_000
PARENT_BUCKETS = 60_000

AMAZON_BEAUTY_NUM_SPARSE_EMBS: List[int] = [
    USER_BUCKETS,
    ASIN_BUCKETS,
    PARENT_BUCKETS,
    2,
]
AMAZON_BEAUTY_NUM_CAT_FEATURES = 4
AMAZON_BEAUTY_NUM_DENSE_FEATURES = 5

POSITIVE_RATING_THRESHOLD = 4

USER_FEATURE_INDEX = 0
ASIN_FEATURE_INDEX = 1
PARENT_FEATURE_INDEX = 2
VERIFIED_FEATURE_INDEX = 3
HELPFUL_FEATURE_INDEX = AMAZON_BEAUTY_NUM_CAT_FEATURES + 0
TIMESTAMP_FEATURE_INDEX = AMAZON_BEAUTY_NUM_CAT_FEATURES + 1
TEXT_LEN_FEATURE_INDEX = AMAZON_BEAUTY_NUM_CAT_FEATURES + 2
TITLE_LEN_FEATURE_INDEX = AMAZON_BEAUTY_NUM_CAT_FEATURES + 3
HAS_IMAGES_FEATURE_INDEX = AMAZON_BEAUTY_NUM_CAT_FEATURES + 4

AMAZON_BEAUTY_FEATURE_GROUPS: List[List[int]] = [
    [USER_FEATURE_INDEX],
    [ASIN_FEATURE_INDEX, PARENT_FEATURE_INDEX],
    [VERIFIED_FEATURE_INDEX],
    [HELPFUL_FEATURE_INDEX, TEXT_LEN_FEATURE_INDEX, TITLE_LEN_FEATURE_INDEX, HAS_IMAGES_FEATURE_INDEX],
    [TIMESTAMP_FEATURE_INDEX],
]


def _load_arrow(arrow_file_path: str):
    with pa.memory_map(arrow_file_path) as src:
        reader = ipc.open_stream(src)
        return reader.read_all()


def _hash_strings(values, num_buckets: int) -> np.ndarray:
    if not values:
        return np.zeros(0, dtype=np.int32)
    safe = ["__missing__" if v is None else v for v in values]
    hashed = tf.strings.to_hash_bucket_fast(
        tf.constant(safe, dtype=tf.string), num_buckets
    )
    return hashed.numpy().astype(np.int32)


def _parse_timestamps(ts_strings) -> np.ndarray:
    if not ts_strings:
        return np.zeros(0, dtype=np.float64)
    seconds = np.empty(len(ts_strings), dtype=np.float64)
    for i, s in enumerate(ts_strings):
        if not s:
            seconds[i] = 0.0
            continue
        head = s.split(".")[0]
        try:
            seconds[i] = (
                np.datetime64(head)
                .astype("datetime64[s]")
                .astype(np.int64)
            )
        except ValueError:
            seconds[i] = 0.0
    return seconds


def _strlen_log(values) -> np.ndarray:
    if not values:
        return np.zeros(0, dtype=np.float32)
    lens = np.fromiter(
        (0 if v is None else len(v) for v in values),
        dtype=np.int64,
        count=len(values),
    )
    return np.log1p(lens.astype(np.float32))


def _has_images(values) -> np.ndarray:
    if not values:
        return np.zeros(0, dtype=np.float32)
    out = np.empty(len(values), dtype=np.float32)
    for i, v in enumerate(values):
        if v is None:
            out[i] = 0.0
            continue
        stripped = v.strip()
        out[i] = 0.0 if stripped in ("", "[]") else 1.0
    return out


def _table_to_arrays(table):
    rating = table.column("rating").to_numpy()
    user_id = table.column("user_id").to_pylist()
    asin = table.column("asin").to_pylist()
    parent_asin = table.column("parent_asin").to_pylist()
    verified = table.column("verified_purchase").to_numpy()
    helpful = table.column("helpful_vote").to_numpy()
    ts_strings = table.column("timestamp").to_pylist()
    text_strings = table.column("text").to_pylist()
    title_strings = table.column("title").to_pylist()
    images_strings = table.column("images").to_pylist()

    user_hash = _hash_strings(user_id, USER_BUCKETS)
    asin_hash = _hash_strings(asin, ASIN_BUCKETS)
    parent_hash = _hash_strings(parent_asin, PARENT_BUCKETS)
    verified_int = verified.astype(np.int32)

    sparse = np.column_stack(
        [user_hash, asin_hash, parent_hash, verified_int]
    ).astype(np.int32)

    helpful_log = np.log1p(
        np.clip(helpful.astype(np.float32), 0.0, None)
    )

    ts_seconds = _parse_timestamps(ts_strings)
    if ts_seconds.size > 0:
        ts_min = float(ts_seconds.min())
        ts_max = float(ts_seconds.max())
        denom = max(1.0, ts_max - ts_min)
        ts_norm = ((ts_seconds - ts_min) / denom).astype(np.float32)
    else:
        ts_norm = np.zeros(0, dtype=np.float32)

    text_log = _strlen_log(text_strings)
    title_log = _strlen_log(title_strings)
    has_imgs = _has_images(images_strings)

    dense = np.column_stack(
        [helpful_log, ts_norm, text_log, title_log, has_imgs]
    ).astype(np.float32)
    labels = (rating.astype(np.int32) >= POSITIVE_RATING_THRESHOLD).astype(np.int32)

    return labels, dense, sparse


_CACHE = {}


def get_dataset(
    arrow_file_path: str,
    split: str = "train",
    batch_size: int = 1024,
    shuffle: bool = True,
    train_size: int = 600_000,
    valid_size: int = None,
    shuffle_seed: int = 42,
    reshuffle_each_iteration: bool = True,
) -> tf.data.Dataset:
    print(f"Loading data from {arrow_file_path} for split: {split}...")
    if arrow_file_path not in _CACHE:
        table = _load_arrow(arrow_file_path)
        _CACHE[arrow_file_path] = _table_to_arrays(table)
    labels_all, dense_all, sparse_all = _CACHE[arrow_file_path]

    total_len = labels_all.shape[0]
    train_size = min(train_size, total_len)
    valid_end = (
        total_len if valid_size is None else min(train_size + valid_size, total_len)
    )

    if split == "train":
        row_slice = slice(0, train_size)
    elif split == "valid":
        row_slice = slice(train_size, valid_end)
    else:
        raise ValueError("split must be 'train' or 'valid'")

    labels = labels_all[row_slice]
    dense_features = dense_all[row_slice].copy()
    sparse_features = sparse_all[row_slice].copy()

    if shuffle and split == "train":
        print(
            "Shuffling data in memory "
            f"(seed={shuffle_seed}, reshuffle_each_iteration={reshuffle_each_iteration})..."
        )
        indices = np.random.default_rng(int(shuffle_seed)).permutation(len(labels))
        labels = labels[indices]
        dense_features = dense_features[indices]
        sparse_features = sparse_features[indices]

    print("Creating tf.data.Dataset pipeline...")
    dataset = tf.data.Dataset.from_tensor_slices(
        ((sparse_features, dense_features), labels)
    )

    if shuffle:
        buf = min(10000, max(1, int(len(labels))))
        dataset = dataset.shuffle(
            buffer_size=buf,
            seed=int(shuffle_seed),
            reshuffle_each_iteration=bool(reshuffle_each_iteration),
        )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    import os
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.normpath(
        os.path.join(here, "..", "dataset", "amazon-beauty-reviews-dataset-train.arrow")
    )
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    ds = get_dataset(path, split="train", batch_size=4, train_size=10000)
    for (sparse, dense), label in ds.take(2):
        print("sparse:", sparse.shape, sparse.numpy())
        print("dense :", dense.shape, dense.numpy())
        print("label :", label.shape, label.numpy())
    print("vocab:", AMAZON_BEAUTY_NUM_SPARSE_EMBS)
    print("groups:", AMAZON_BEAUTY_FEATURE_GROUPS)
