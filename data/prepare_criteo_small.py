import argparse
import gzip
import os

import numpy as np


def open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def main():
    parser = argparse.ArgumentParser(
        description="Build a small real Criteo NPZ for TokenMixer-Large validation."
    )
    parser.add_argument("--input", required=True, help="Path to Criteo train.txt or train.txt.gz")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--rows", type=int, default=10240, help="Number of real rows to keep")
    args = parser.parse_args()

    labels = []
    dense_rows = []
    sparse_rows = []
    cat_maps = [dict() for _ in range(26)]

    with open_text(args.input) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 40:
                continue

            labels.append(int(parts[0]))
            dense_rows.append([float(v) if v else 0.0 for v in parts[1:14]])

            sparse = []
            for i, raw_value in enumerate(parts[14:]):
                value = raw_value or "__MISSING__"
                mapping = cat_maps[i]
                if value not in mapping:
                    mapping[value] = len(mapping)
                sparse.append(mapping[value])
            sparse_rows.append(sparse)

            if len(labels) >= args.rows:
                break

    if not labels:
        raise ValueError(f"No valid Criteo rows found in {args.input}")

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
