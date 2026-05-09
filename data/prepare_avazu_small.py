import argparse
import csv
import os

import numpy as np


def parse_nonnegative_float(value):
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Build a small real Avazu NPZ compatible with TokenMixer-Large."
    )
    parser.add_argument("--input", required=True, help="Path to Avazu train.csv")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--rows", type=int, default=10240, help="Number of real rows to keep")
    parser.add_argument("--num_sparse", type=int, default=26, help="Sparse feature count")
    parser.add_argument("--num_dense", type=int, default=13, help="Dense feature count")
    args = parser.parse_args()

    labels = []
    dense_rows = []
    sparse_rows = []
    cat_maps = [dict() for _ in range(args.num_sparse)]

    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {args.input}")

        label_col = "click" if "click" in reader.fieldnames else reader.fieldnames[0]
        feature_cols = [
            col for col in reader.fieldnames if col not in {label_col, "id"}
        ]

        for row in reader:
            labels.append(int(float(row[label_col])))

            dense_values = [
                parse_nonnegative_float(row.get(col, "0")) for col in feature_cols
            ][: args.num_dense]
            dense_values += [0.0] * (args.num_dense - len(dense_values))
            dense_rows.append(dense_values)

            sparse_values = feature_cols[: args.num_sparse]
            encoded_sparse = []
            for i, col in enumerate(sparse_values):
                value = row.get(col, "") or "__MISSING__"
                mapping = cat_maps[i]
                if value not in mapping:
                    mapping[value] = len(mapping)
                encoded_sparse.append(mapping[value])
            encoded_sparse += [0] * (args.num_sparse - len(encoded_sparse))
            sparse_rows.append(encoded_sparse)

            if len(labels) >= args.rows:
                break

    if not labels:
        raise ValueError(f"No valid Avazu rows found in {args.input}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(
        args.output,
        y=np.asarray(labels, dtype=np.int32),
        X_int=np.asarray(dense_rows, dtype=np.float32),
        X_cat=np.asarray(sparse_rows, dtype=np.int32),
    )

    vocab_sizes = [len(m) for m in cat_maps]
    print(f"Saved {len(labels)} rows to {args.output}")
    print(f"Sparse vocab sizes: {vocab_sizes}")


if __name__ == "__main__":
    main()
