import os
import runpy
import sys


def main():
    has_backend = any(arg == "--backend" or arg.startswith("--backend=") for arg in sys.argv)
    if not has_backend:
        sys.argv.extend(["--backend", "cuda"])

    train_path = os.path.join(os.path.dirname(__file__), "train.py")
    runpy.run_path(train_path, run_name="__main__")


if __name__ == "__main__":
    main()
