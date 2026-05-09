import tensorflow as tf
import numpy as np


def get_dataset(
    npz_file_path: str,
    split: str = "train",
    batch_size: int = 1024,
    shuffle: bool = True,
    train_size: int = 39291958,
    valid_size: int = None,
    shuffle_seed: int = 42,
    reshuffle_each_iteration: bool = True,
) -> tf.data.Dataset:
    print(f"Loading data from {npz_file_path} for split: {split}...")
    with np.load(npz_file_path) as data:
        total_len = data["y"].shape[0]
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

        labels = data["y"][row_slice].astype(np.int32)
        dense_features = data["X_int"][row_slice].astype(np.float32)
        sparse_features = data["X_cat"][row_slice].astype(np.int32)

    np.log1p(dense_features, out=dense_features)

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
