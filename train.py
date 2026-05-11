import argparse
import os
import sys

# ----------------------------------------------------------------------------
# Early MUSA device selection — must happen BEFORE `import tensorflow as tf`.
#
# tf.config.set_visible_devices(...) only works when called before any TF
# device context has been initialized. tf.load_library / tf.load_op_library
# (which we need to pull in the MUSA plugin and its custom ops below) both
# touch the device, so by the time _configure_musa_physical_devices() calls
# set_visible_devices() it is too late — TF has already bound to whichever
# physical card the driver enumerated first (typically index 0). When index 0
# is busy on a shared host this manifests as an OOM at model build despite
# passing --musa_device_index N for some other N.
#
# We side-step the ordering problem by setting MUSA_VISIBLE_DEVICES in
# os.environ now, before any TF call. The MUSA driver then only ever shows TF
# one physical device, no in-Python re-routing is required, and the OOM-prone
# busy device is invisible to this process entirely.
#
# A pre-existing MUSA_VISIBLE_DEVICES in the environment always wins, so a
# container that already limits visibility (or a user who set the variable in
# their shell) is not overridden.
# ----------------------------------------------------------------------------
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--backend", default="cuda")
_pre_parser.add_argument("--musa_device_index", type=int, default=0)
_pre_parser.add_argument("--all_musa_devices", action="store_true")
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.backend == "musa" and not _pre_args.all_musa_devices:
    if "MUSA_VISIBLE_DEVICES" not in os.environ:
        os.environ["MUSA_VISIBLE_DEVICES"] = str(_pre_args.musa_device_index)
        print(
            f"[musa] set MUSA_VISIBLE_DEVICES={_pre_args.musa_device_index} "
            f"from --musa_device_index (before TF import).",
            file=sys.stderr,
        )
    else:
        print(
            f"[musa] MUSA_VISIBLE_DEVICES already set to "
            f"'{os.environ['MUSA_VISIBLE_DEVICES']}' in the environment; "
            f"--musa_device_index ignored.",
            file=sys.stderr,
        )

import csv
import time
import tensorflow as tf
import numpy as np
import random
import logging
from datetime import datetime
import shutil

parser = argparse.ArgumentParser(description="Train TokenMixer-Large with TensorFlow MUSA Extension")
parser.add_argument(
    "--backend",
    choices=["musa", "cuda"],
    default="cuda",
    help="Runtime backend. 'cuda' is the standard TensorFlow path; 'musa' enables the "
    "Moore Threads TensorFlow MUSA Extension (requires --lib_path).",
)
parser.add_argument(
    "--lib_path",
    type=str,
    default=None,
    help="Path to the TensorFlow MUSA library .so file. Required only for --backend musa.",
)
parser.add_argument(
    "--dataset",
    choices=["criteo", "amazon_beauty"],
    default="criteo",
    help="Which dataset to train on. Both are widely used recommendation benchmarks.",
)
parser.add_argument(
    "--data_path",
    type=str,
    default=None,
    help="Path to the dataset file. If omitted, a sensible default is picked per --dataset.",
)
parser.add_argument(
    "--save_checkpoints",
    action="store_true",
    default=False,
    help="Whether to save model checkpoints after each epoch",
)
parser.add_argument(
    "--enable_xla",
    action="store_true",
    default=False,
    help="Whether to enable XLA JIT compilation",
)
parser.add_argument(
    "--precision",
    choices=["fp32", "bf16", "mixed_bf16"],
    default="fp32",
    help="Numeric precision policy. 'bf16' uses bfloat16 for both variables and compute, "
    "'mixed_bf16' uses bfloat16 compute with float32 variables.",
)
parser.add_argument(
    "--disable_tf32",
    action="store_true",
    default=False,
    help="Disable TensorFloat-32 kernels when available (recommended for cross-backend alignment).",
)
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=4096, help="Training/eval batch size")
parser.add_argument(
    "--train_size",
    type=int,
    default=None,
    help="Number of leading rows to use for training. Default depends on --dataset.",
)
parser.add_argument(
    "--valid_size",
    type=int,
    default=None,
    help="Number of rows after train_size to use for validation",
)
parser.add_argument(
    "--max_train_batches",
    type=int,
    default=None,
    help="Optional cap on training batches per epoch for smoke validation",
)
parser.add_argument(
    "--max_valid_batches",
    type=int,
    default=None,
    help="Optional cap on validation batches for smoke validation",
)
parser.add_argument(
    "--derive_sparse_embs_from_data",
    action="store_true",
    default=False,
    help="Derive sparse embedding table sizes from the supplied NPZ for small local validation",
)
parser.add_argument("--peak_lr", type=float, default=0.001, help="Peak learning rate")
parser.add_argument("--init_lr", type=float, default=1e-8, help="Initial warmup learning rate")
parser.add_argument(
    "--grad_clip_norm",
    type=float,
    default=1.0,
    help="Global gradient clipping norm. Set <= 0 to disable.",
)
parser.add_argument(
    "--aux_loss_weight",
    type=float,
    default=0.1,
    help="Weight for supervised auxiliary losses from intermediate layers (Section 3.3.4).",
)
parser.add_argument("--num_layers", type=int, default=6, help="Number of TokenMixer-Large blocks")
parser.add_argument("--dim_emb", type=int, default=128, help="Token embedding dimension D")
parser.add_argument(
    "--num_heads",
    type=int,
    default=16,
    help="Number of mixing heads H. Must divide --dim_emb. Unlike RankMixer, T need NOT equal H "
    "in TokenMixer-Large — Mixing+Reverting handles dimension mismatches by design (Section 3.3.1).",
)
parser.add_argument("--num_experts", type=int, default=8, help="Total experts per S-P MoE (1 shared + N-1 routed)")
parser.add_argument("--top_k", type=int, default=4, help="Top-k routing (includes the shared expert)")
parser.add_argument(
    "--alpha",
    type=float,
    default=2.0,
    help="Gate-Value-Scaling alpha (Section 3.4.3). Paper: alpha = num_experts / top_k.",
)
parser.add_argument("--hidden_mult", type=float, default=4.0, help="SwiGLU hidden expansion factor n (Eq. 18)")
parser.add_argument(
    "--inter_residual_gap",
    type=int,
    default=2,
    help="Layer skip distance for inter-residuals (Section 3.3.4). Paper recommends 2 or 3.",
)
parser.add_argument(
    "--bias",
    action="store_true",
    default=False,
    help="Enable bias on linear kernels. Paper §A.4 removes all biases (Llama-style).",
)
parser.add_argument(
    "--dropout", type=float, default=0.5, help="Dropout in the projection head and tokenizer."
)
parser.add_argument(
    "--lr_schedule",
    choices=["warmup_constant", "warmup_cosine"],
    default="warmup_cosine",
    help="LR schedule. 'warmup_cosine' adds cosine decay after warmup (smoother late training).",
)
parser.add_argument(
    "--min_lr",
    type=float,
    default=1e-5,
    help="Minimum LR at the end of cosine decay (only used with warmup_cosine).",
)
parser.add_argument(
    "--feature_grouping",
    choices=["coarse", "semantic"],
    default="semantic",
    help="'coarse' = original 2 groups (all sparse, all dense); "
    "'semantic' = per-dataset semantic feature groups (recommended).",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Fix Python, NumPy, and TensorFlow RNGs; passed to dataset shuffling for repeatable order.",
)
parser.add_argument(
    "--deterministic_alignment",
    action="store_true",
    help="Stable batch order across epochs (no tf.data reshuffle each epoch); sets TF deterministic "
    "ops when available. Use with the same --seed when comparing MUSA vs CUDA.",
)
parser.add_argument(
    "--musa_device_index",
    type=int,
    default=0,
    help="MUSA only: physical device index to expose when not using --all_musa_devices "
    "(default matches single-GPU CUDA training).",
)
parser.add_argument(
    "--all_musa_devices",
    action="store_true",
    help="MUSA only: expose every MUSA accelerator to TensorFlow (default: hide all but one — "
    "training is still single-device unless you add a distribution strategy).",
)
parser.add_argument(
    "--disable_musa_mixed_adam",
    action="store_true",
    default=False,
    help="MUSA only: disable the mixed-precision Adam fast path that routes "
    "bf16/fp16 gradients through MusaResourceApplyAdamMixed with fp32 state. "
    "Set this only for A/B comparisons against the legacy Cast+ResourceApplyAdam "
    "path; it has no effect on --backend cuda or --precision fp32.",
)
args = parser.parse_args()


def _tf_version_tuple():
    parts = []
    for token in tf.__version__.split("."):
        digits = ""
        for ch in token:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            parts.append(int(digits))
        else:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


_TF_VERSION = _tf_version_tuple()
_LEGACY_TF = _TF_VERSION < (2, 7, 0)


def _configure_precision_policy(precision: str):
    """Resolve --precision to a Keras mixed-precision policy.

    On TF < 2.7 the global "bfloat16" policy has a known interaction with
    multi-input Keras Models that surfaces as ``make_shape`` receiving
    ``[TensorShape([...])]`` during the build call. We auto-degrade to
    ``mixed_bfloat16`` which keeps bf16 compute but uses fp32 master weights,
    avoiding that build path while remaining numerically usable for training.
    """
    if precision == "fp32":
        policy_name = "float32"
        model_dtype = tf.float32
    elif precision == "bf16":
        if _LEGACY_TF:
            policy_name = "mixed_bfloat16"
            model_dtype = tf.bfloat16
            print(
                f"[precision] TF {tf.__version__} (<2.7) has a multi-input Keras "
                f"build issue with global 'bfloat16' policy; degrading "
                f"--precision bf16 to 'mixed_bfloat16' (bf16 compute, fp32 "
                f"master weights). For pure bf16, upgrade TF to >= 2.7.",
                file=sys.stderr,
            )
        else:
            policy_name = "bfloat16"
            model_dtype = tf.bfloat16
    elif precision == "mixed_bf16":
        policy_name = "mixed_bfloat16"
        model_dtype = tf.bfloat16
    else:
        raise ValueError(f"Unknown precision: {precision}")
    tf.keras.mixed_precision.set_global_policy(policy_name)
    return policy_name, model_dtype


def _configure_musa_physical_devices(expose_all: bool, device_index: int) -> None:
    """Train loop is single-device; TF still enumerates every card unless we hide them.

    Even with MUSA_VISIBLE_DEVICES already set in the environment (which is
    the only fully reliable way to pin TF to a specific physical card on a
    shared host), we still run this for diagnostics: it prints exactly what
    TF sees post-import, so a mis-pinned process is obvious from the log
    instead of hidden behind a generic OOM later.
    """
    try:
        physical = tf.config.list_physical_devices("MUSA")
    except (ValueError, TypeError):
        physical = []

    musa_visible_env = os.environ.get("MUSA_VISIBLE_DEVICES", "<unset>")
    print(
        f"[musa] MUSA_VISIBLE_DEVICES={musa_visible_env}, "
        f"tf.list_physical_devices('MUSA') -> {len(physical)} device(s): "
        f"{[p.name for p in physical]}",
        file=sys.stderr,
    )

    if not physical:
        print(
            "[musa] WARNING: TensorFlow sees zero MUSA devices. Check that "
            "the plugin loaded successfully and that MUSA_VISIBLE_DEVICES is "
            "not filtered to a non-existent index.",
            file=sys.stderr,
        )
        return
    if expose_all:
        print(
            f"[musa] TensorFlow sees all {len(physical)} MUSA physical device(s). "
            "This script does not shard work across them; only the default device runs training."
        )
        return
    idx = max(0, min(device_index, len(physical) - 1))
    picked = physical[idx]
    try:
        tf.config.set_visible_devices([picked], "MUSA")
    except (RuntimeError, ValueError, TypeError) as exc:
        print(
            f"[musa] Could not set_visible_devices to index {idx}: {exc}. "
            "This is normally fine when MUSA_VISIBLE_DEVICES is already set "
            "in the environment (the driver has already filtered devices, so "
            "TF is bound to the right physical card regardless of this "
            "failure). If MUSA_VISIBLE_DEVICES is unset, TF will run on "
            "whichever device the driver enumerated first.",
            file=sys.stderr,
        )
        return
    print(
        f"[musa] TensorFlow restricted to one accelerator: index {idx} ({picked}). "
        "Matches single-GPU CUDA visibility; use --all_musa_devices for full enumeration."
    )


_MUSA_OP_MODULE = None
if args.backend == "musa":
    if args.lib_path is None:
        raise ValueError("--lib_path is required when --backend musa")
    tf.load_library(args.lib_path)
    # tf.load_library registers the MUSA device + kernels but does NOT make
    # plugin-defined custom ops (e.g. MusaResourceApplyAdamMixed) accessible
    # from Python: tf.raw_ops only contains ops baked into the TF wheel at
    # build time, so plugin ops never appear there even when they are fully
    # registered in the C++ op registry. tf.load_op_library on the same path
    # re-uses the dlopen cache (no double registration) and returns a Python
    # module whose attributes ARE the generated wrappers for those ops.
    try:
        _MUSA_OP_MODULE = tf.load_op_library(args.lib_path)
    except Exception as _exc:  # pragma: no cover - depends on plugin contents
        print(
            f"[musa] tf.load_op_library({args.lib_path}) failed: {_exc}. "
            f"Custom MUSA ops (e.g. MusaResourceApplyAdamMixed) will be "
            f"unavailable from Python and the mixed Adam fast path will "
            f"silently fall back to the stock Cast+Adam path.",
            file=sys.stderr,
        )
    _configure_musa_physical_devices(args.all_musa_devices, args.musa_device_index)

if args.backend == "cuda":
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

if args.enable_xla:
    tf.config.optimizer.set_jit(True)

if args.disable_tf32:
    set_tf32 = getattr(tf.config.experimental, "enable_tensor_float_32_execution", None)
    if callable(set_tf32):
        try:
            set_tf32(False)
        except (RuntimeError, ValueError, TypeError):
            pass

POLICY_NAME, MODEL_DTYPE = _configure_precision_policy(args.precision)
LABEL_DTYPE = MODEL_DTYPE


####################################################################################################
#                                           SET RANDOM SEEDS                                       #
####################################################################################################
def _activate_determinism_best_effort() -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    ena = getattr(tf.config.experimental, "enable_op_determinism", None)
    if callable(ena):
        try:
            ena()
        except TypeError:
            try:
                ena(True)  # type: ignore[misc]
            except (RuntimeError, ValueError, TypeError):
                pass
        except (RuntimeError, ValueError):
            pass


if args.deterministic_alignment:
    _activate_determinism_best_effort()

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)


####################################################################################################
#                                         CREATE LOGGER                                            #
####################################################################################################
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d-%H.%M.%S")
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
os.makedirs(f"logs/{formatted_time}", exist_ok=True)
file_handler = logging.FileHandler(
    f"logs/{formatted_time}/training.log", mode="a", encoding="utf-8"
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)
logger = logging.getLogger("tokenmixerlarge_training")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
LOGGER_PRINT_INTERVAL = 100

logger.info(f"seed={args.seed}, deterministic_alignment={args.deterministic_alignment}")
logger.info(f"dataset={args.dataset}")
logger.info(
    f"precision={args.precision}, keras_policy={POLICY_NAME}, "
    f"model_dtype={MODEL_DTYPE.name}, label_dtype={LABEL_DTYPE.name}, disable_tf32={args.disable_tf32}"
)
if args.backend == "musa":
    logger.info(
        f"musa_device_index={args.musa_device_index}, all_musa_devices={args.all_musa_devices}"
    )

summary_writer = tf.summary.create_file_writer(f"logs/{formatted_time}/tensorboard")
shutil.copy(
    src=__file__,
    dst=f"logs/{formatted_time}/",
)
shutil.copytree(
    src="data",
    dst=f"logs/{formatted_time}/data",
    ignore=shutil.ignore_patterns("__pycache__"),
    dirs_exist_ok=True,
)
shutil.copytree(
    src="model",
    dst=f"logs/{formatted_time}/model",
    ignore=shutil.ignore_patterns("__pycache__"),
    dirs_exist_ok=True,
)
checkpoint_dir = f"logs/{formatted_time}/checkpoints"
SAVE_CHECKPOINTS = args.save_checkpoints
if SAVE_CHECKPOINTS:
    os.makedirs(checkpoint_dir, exist_ok=True)

metrics_csv_path = f"logs/{formatted_time}/metrics.csv"
metrics_plot_path = f"logs/{formatted_time}/train_val_curves.png"
with open(metrics_csv_path, "w", newline="", encoding="utf-8") as fh:
    csv.writer(fh).writerow(
        [
            "epoch",
            "train_loss",
            "valid_loss",
            "valid_auc",
            "valid_accuracy",
            "valid_recall_pos",
            "valid_samples",
            "valid_pos_samples",
            "avg_train_iter_sec",
            "train_phase_sec",
            "valid_phase_sec",
            "epoch_total_sec",
        ]
    )

####################################################################################################
#                                         LOAD MODELS                                              #
####################################################################################################

if args.backend == "musa":
    logger.info("Successfully loaded tensorflow musa library from " + args.lib_path)
else:
    logger.info("Running with TensorFlow CUDA backend")
from model.tokenmixerlarge import TokenMixerLarge
from model.lr_schedule import LinearWarmup, WarmupCosine

####################################################################################################
#                                  DATASET SPECIFIC CONFIGURATION                                  #
####################################################################################################
CRITEO_NUM_SPARSE_EMBS = [
    1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145,
    5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4,
    7046547, 18, 15, 286181, 105, 142572,
]


def _default_data_path(dataset: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    if dataset == "criteo":
        return os.path.join(here, "dataset", "criteo_data.npz")
    if dataset == "amazon_beauty":
        return os.path.join(
            here, "dataset", "amazon-beauty-reviews-dataset-train.arrow"
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def _default_train_size(dataset: str) -> int:
    if dataset == "criteo":
        return 39_291_958
    if dataset == "amazon_beauty":
        return 600_000
    raise ValueError(f"Unknown dataset: {dataset}")


if args.data_path is None:
    args.data_path = _default_data_path(args.dataset)
if args.train_size is None:
    args.train_size = _default_train_size(args.dataset)

DATA_PATH = args.data_path

if args.dataset == "criteo":
    NUM_CAT_FEATURES = 26
    NUM_DENSE_FEATURES = 13
    NUM_SPARSE_EMBS = list(CRITEO_NUM_SPARSE_EMBS)
    from data.criteo_kaggle_dataset import get_dataset

    if args.derive_sparse_embs_from_data:
        with np.load(DATA_PATH) as data:
            derive_rows = min(
                data["X_cat"].shape[0],
                args.train_size + (args.valid_size or 0),
            )
            NUM_SPARSE_EMBS = (
                data["X_cat"][:derive_rows].max(axis=0).astype(np.int64) + 1
            ).tolist()

    SEMANTIC_FEATURE_GROUPS = None  # Criteo is anonymized; coarse split is the default.

elif args.dataset == "amazon_beauty":
    from data.amazon_beauty_dataset import (
        AMAZON_BEAUTY_FEATURE_GROUPS,
        AMAZON_BEAUTY_NUM_CAT_FEATURES,
        AMAZON_BEAUTY_NUM_DENSE_FEATURES,
        AMAZON_BEAUTY_NUM_SPARSE_EMBS,
        get_dataset,
    )

    NUM_CAT_FEATURES = AMAZON_BEAUTY_NUM_CAT_FEATURES
    NUM_DENSE_FEATURES = AMAZON_BEAUTY_NUM_DENSE_FEATURES
    NUM_SPARSE_EMBS = list(AMAZON_BEAUTY_NUM_SPARSE_EMBS)
    SEMANTIC_FEATURE_GROUPS = [list(g) for g in AMAZON_BEAUTY_FEATURE_GROUPS]

else:
    raise ValueError(f"Unknown dataset: {args.dataset}")

COARSE_FEATURE_GROUPS = [
    list(range(0, NUM_CAT_FEATURES)),
    list(range(NUM_CAT_FEATURES, NUM_CAT_FEATURES + NUM_DENSE_FEATURES)),
]

if args.feature_grouping == "semantic" and SEMANTIC_FEATURE_GROUPS is not None:
    FEATURE_GROUPS = SEMANTIC_FEATURE_GROUPS
else:
    FEATURE_GROUPS = COARSE_FEATURE_GROUPS

DIM_OUTPUT = 1
logger.info(
    f"data_path={DATA_PATH} num_cat={NUM_CAT_FEATURES} num_dense={NUM_DENSE_FEATURES} "
    f"num_sparse_embs={NUM_SPARSE_EMBS}"
)
logger.info(
    f"feature_grouping={args.feature_grouping} num_groups={len(FEATURE_GROUPS)} "
    f"groups={FEATURE_GROUPS}"
)

####################################################################################################
#                                   MODEL SPECIFIC CONFIGURATION                                   #
####################################################################################################
NUM_LAYERS = args.num_layers
DIM_EMB = args.dim_emb
NUM_HEADS = args.num_heads
NUM_EXPERTS = args.num_experts
TOP_K = args.top_k
HIDDEN_MULT = args.hidden_mult
ALPHA = args.alpha
INTER_RESIDUAL_GAP = args.inter_residual_gap
NUM_HIDDEN_HEAD = 2
DIM_HIDDEN_HEAD = 256
DROPOUT = args.dropout
BIAS = args.bias

logger.info(
    f"model: layers={NUM_LAYERS} dim={DIM_EMB} heads={NUM_HEADS} experts={NUM_EXPERTS} "
    f"top_k={TOP_K} hidden_mult={HIDDEN_MULT} alpha={ALPHA} bias={BIAS} "
    f"inter_residual_gap={INTER_RESIDUAL_GAP}"
)

####################################################################################################
#                                           CREATE MODEL                                           #
####################################################################################################
model = TokenMixerLarge(
    feature_groups=FEATURE_GROUPS,
    num_layers=NUM_LAYERS,
    num_sparse_embs=NUM_SPARSE_EMBS,
    dim_input_sparse=NUM_CAT_FEATURES,
    dim_input_dense=NUM_DENSE_FEATURES,
    dim_emb=DIM_EMB,
    num_heads=NUM_HEADS,
    num_experts=NUM_EXPERTS,
    top_k=TOP_K,
    num_hidden_head=NUM_HIDDEN_HEAD,
    dim_hidden_head=DIM_HIDDEN_HEAD,
    dim_output=DIM_OUTPUT,
    dropout=DROPOUT,
    bias=BIAS,
    hidden_mult=HIDDEN_MULT,
    alpha=ALPHA,
    inter_residual_gap=INTER_RESIDUAL_GAP,
)

####################################################################################################
#                                  TRAINING SPECIFIC CONFIGURATION                                 #
####################################################################################################
BATCH_SIZE = args.batch_size
TRAIN_EPOCHS = args.epochs
PEAK_LR = args.peak_lr
INIT_LR = args.init_lr
TOTAL_STEPS_PER_EPOCH = max(1, args.train_size // BATCH_SIZE)
WARMUP_STEPS = TOTAL_STEPS_PER_EPOCH
TOTAL_TRAIN_STEPS = max(WARMUP_STEPS + 1, TRAIN_EPOCHS * TOTAL_STEPS_PER_EPOCH)

if args.lr_schedule == "warmup_cosine":
    lr_schedule = WarmupCosine(
        initial_learning_rate=INIT_LR,
        peak_learning_rate=PEAK_LR,
        warmup_steps=WARMUP_STEPS,
        total_steps=TOTAL_TRAIN_STEPS,
        min_learning_rate=args.min_lr,
    )
else:
    lr_schedule = LinearWarmup(
        initial_learning_rate=INIT_LR,
        peak_learning_rate=PEAK_LR,
        warmup_steps=WARMUP_STEPS,
    )
logger.info(
    f"lr_schedule={args.lr_schedule} init_lr={INIT_LR} peak_lr={PEAK_LR} "
    f"warmup_steps={WARMUP_STEPS} total_steps={TOTAL_TRAIN_STEPS} min_lr={args.min_lr}"
)
embedding_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)


def _is_musa_mixed_adam_available() -> bool:
    """True if the MUSA plugin exposed musa_resource_apply_adam_mixed.

    The op is registered at plugin load time via REGISTER_OP inside the .so
    the user passed to --lib_path, but ``tf.raw_ops`` is a static module
    baked into the TF wheel and never sees plugin ops. We look up the op via
    the module that ``tf.load_op_library`` returned instead (loaded right
    after ``tf.load_library`` above; same dlopen handle, just exposes the
    auto-generated Python wrappers).
    """
    return (_MUSA_OP_MODULE is not None
            and hasattr(_MUSA_OP_MODULE, "musa_resource_apply_adam_mixed"))


def _make_musa_adam_class():
    """Build a Keras Adam subclass that dispatches to MusaResourceApplyAdamMixed.

    The mixed op accepts fp32 master state (``var``/``m``/``v``) and a bf16,
    fp16, or fp32 gradient. Promotion happens inside the kernel with RNE
    rounding, so the math is bit-equivalent to running stock Adam after an
    explicit fp32 cast — except no fp32 gradient tensor is materialized on
    device, saving one gradient-sized memory pass per parameter per step.

    Falls back to the parent implementation when AMSGrad is on (no mixed
    AMSGrad op yet), when the variable isn't on a MUSA device, when the
    variable isn't fp32 (proper mixed precision keeps master weights in
    fp32), or when the gradient dtype is unexpected.
    """

    class MusaAdam(tf.keras.optimizers.Adam):
        _LOWP_GRAD_DTYPES = (tf.float32, tf.bfloat16, tf.float16)

        def __init__(self, *opt_args, **opt_kwargs):
            super().__init__(*opt_args, **opt_kwargs)
            self._musa_amsgrad_warned = False
            self._musa_var_dtype_warned = False

        def _resource_apply_dense(self, grad, var, apply_state=None):
            if self.amsgrad:
                if not self._musa_amsgrad_warned:
                    logger.info(
                        "MusaAdam: AMSGrad enabled, falling back to stock Adam "
                        "apply path. Disable AMSGrad to use the MUSA mixed-"
                        "precision fast path."
                    )
                    self._musa_amsgrad_warned = True
                return super()._resource_apply_dense(grad, var, apply_state)

            var_device = var.device or ""
            if "MUSA" not in var_device:
                return super()._resource_apply_dense(grad, var, apply_state)

            if var.dtype.base_dtype != tf.float32:
                if not self._musa_var_dtype_warned:
                    logger.warning(
                        "MusaAdam: variable %s has dtype %s but the mixed Adam "
                        "fast path requires fp32 master weights. Falling back "
                        "to the stock (possibly bf16-state) Adam path, which "
                        "loses precision between iterations. Switch to "
                        "--precision mixed_bf16 (fp32 weights, bf16 compute) "
                        "to take the fast path.",
                        var.name,
                        var.dtype,
                    )
                    self._musa_var_dtype_warned = True
                return super()._resource_apply_dense(grad, var, apply_state)

            if grad.dtype not in self._LOWP_GRAD_DTYPES:
                return super()._resource_apply_dense(grad, var, apply_state)

            var_dtype = var.dtype.base_dtype
            coefficients = (
                (apply_state or {}).get((var_device, var_dtype))
                or self._fallback_apply_state(var_device, var_dtype)
            )
            m = self.get_slot(var, "m")
            v = self.get_slot(var, "v")

            def _f32(value):
                if isinstance(value, tf.Tensor) and value.dtype == tf.float32:
                    return value
                return tf.cast(value, tf.float32)

            return _MUSA_OP_MODULE.musa_resource_apply_adam_mixed(
                var=var.handle,
                m=m.handle,
                v=v.handle,
                beta1_power=_f32(coefficients["beta_1_power"]),
                beta2_power=_f32(coefficients["beta_2_power"]),
                lr=_f32(coefficients["lr_t"]),
                beta1=_f32(coefficients["beta_1_t"]),
                beta2=_f32(coefficients["beta_2_t"]),
                epsilon=_f32(coefficients["epsilon"]),
                grad=grad,
                use_locking=self._use_locking,
                use_nesterov=False,
            )

    return MusaAdam


_on_musa = (args.backend == "musa")
_is_low_precision_run = args.precision in ("bf16", "mixed_bf16")
_musa_mixed_adam_supported = _is_musa_mixed_adam_available()
_use_musa_mixed_adam = (
    _on_musa
    and _is_low_precision_run
    and _musa_mixed_adam_supported
    and not args.disable_musa_mixed_adam
)

if _use_musa_mixed_adam:
    MusaAdam = _make_musa_adam_class()
    other_optimizer = MusaAdam(learning_rate=lr_schedule)
    logger.info(
        "Using MusaAdam (fp32 state, bf16/fp16 grad consumed directly by "
        "MusaResourceApplyAdamMixed; no Cast(bf16->fp32) materialized)."
    )
else:
    other_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    if _on_musa and _is_low_precision_run and not _musa_mixed_adam_supported:
        logger.warning(
            "MusaResourceApplyAdamMixed not found in the loaded plugin "
            "(the .so at --lib_path does not expose "
            "musa_resource_apply_adam_mixed). The stock Adam path with an "
            "explicit Cast(bf16->fp32) will be used instead. Rebuild the "
            "plugin from a revision that includes the mixed-precision Adam "
            "op (verify with: nm -D <plugin.so> | grep -i AdamMixed)."
        )
    elif _on_musa and _is_low_precision_run and args.disable_musa_mixed_adam:
        logger.info(
            "MUSA mixed Adam is supported but disabled by "
            "--disable_musa_mixed_adam; using stock Adam."
        )
    else:
        logger.info(
            "Using stock tf.keras.optimizers.Adam (mixed Adam path doesn't "
            "apply for backend=%s precision=%s).",
            args.backend,
            args.precision,
        )


def binary_cross_entropy_with_logits(labels, logits):
    labels = tf.cast(labels, logits.dtype)
    per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
    )
    return tf.reduce_mean(per_example_loss)

####################################################################################################
#                                       CREATE DATALOADER                                          #
####################################################################################################

train_dataset = get_dataset(
    DATA_PATH,
    split="train",
    batch_size=BATCH_SIZE,
    shuffle=True,
    train_size=args.train_size,
    valid_size=args.valid_size,
    shuffle_seed=args.seed,
    reshuffle_each_iteration=not args.deterministic_alignment,
)

valid_dataset = get_dataset(
    DATA_PATH,
    split="valid",
    batch_size=BATCH_SIZE,
    shuffle=False,
    train_size=args.train_size,
    valid_size=args.valid_size,
    shuffle_seed=args.seed,
)

if MODEL_DTYPE != tf.float32:
    if _LEGACY_TF:
        logger.info(
            f"TF {tf.__version__} (<2.7): skipping explicit dataset bf16 cast; "
            f"Keras mixed-precision policy '{POLICY_NAME}' will auto-cast inputs "
            f"to compute_dtype inside layers."
        )
    else:
        logger.info(
            f"Casting dense features and labels to {MODEL_DTYPE.name} in tf.data pipeline"
        )

        def _cast_batch_to_target_dtype(inputs, labels):
            sparse_inputs, dense_inputs = inputs
            return (
                sparse_inputs,
                tf.cast(dense_inputs, MODEL_DTYPE),
            ), tf.cast(labels, LABEL_DTYPE)

        train_dataset = train_dataset.map(
            _cast_batch_to_target_dtype, num_parallel_calls=tf.data.AUTOTUNE
        )
        valid_dataset = valid_dataset.map(
            _cast_batch_to_target_dtype, num_parallel_calls=tf.data.AUTOTUNE
        )

####################################################################################################
#                                    BUILD MODEL & SEPARATE VARS                                   #
####################################################################################################
_BUILD_DENSE_DTYPE = (
    tf.float32 if tf.keras.mixed_precision.global_policy().variable_dtype == "float32"
    else MODEL_DTYPE
)
dummy_sparse = tf.zeros((1, NUM_CAT_FEATURES), dtype=tf.int32)
dummy_dense = tf.zeros((1, NUM_DENSE_FEATURES), dtype=_BUILD_DENSE_DTYPE)

if _LEGACY_TF:
    logger.info(
        f"TF {tf.__version__} (<2.7): building model via model.call(...) to bypass "
        f"Layer.__call__'s nested-tuple input bookkeeping (avoids "
        f"'make_shape got [TensorShape(...)]' in older TF)."
    )
    _ = model.call((dummy_sparse, dummy_dense), training=False)
    # TF 2.6 Keras Model._assert_weights_created() requires self.built==True when
    # the subclass defines build(); model.call(...) does not flip that flag, so
    # set it explicitly. Sublayers were already built via their own __call__
    # inside model.call(), so model.trainable_variables is now populated.
    model.built = True
else:
    _ = model((dummy_sparse, dummy_dense))

embedding_parameters = []
other_parameters = []

for var in model.trainable_variables:
    if hasattr(var, "path"):
        if "sparse_embedding" in var.path and "embeddings" in var.name:
            embedding_parameters.append(var)
        else:
            other_parameters.append(var)
    else:
        if "sparse_embedding" in var.name:
            embedding_parameters.append(var)
        else:
            other_parameters.append(var)

logger.info(f"Number of embedding parameters: {len(embedding_parameters)}")
logger.info(f"Number of other parameters: {len(other_parameters)}")
dtype_hist = {}
for var in model.trainable_variables:
    dtype_obj = var.dtype
    key = dtype_obj.name if hasattr(dtype_obj, "name") else str(dtype_obj)
    dtype_hist[key] = dtype_hist.get(key, 0) + 1
logger.info(f"Trainable variable dtype histogram: {dtype_hist}")


####################################################################################################
#                                          VALID FUNCTION                                          #
####################################################################################################
def validate(model, dataset, max_batches=None):
    num_samples = 0
    num_correct = 0
    pos_samples = 0
    pos_correct = 0
    loss_metric = tf.keras.metrics.Mean(dtype=tf.float32)
    auc_metric = tf.keras.metrics.AUC(name="validation_auc", dtype=tf.float32)

    for batch_idx, (inputs, labels) in enumerate(dataset):
        if max_batches is not None and batch_idx >= max_batches:
            break
        outputs = model(inputs, training=False)

        labels = tf.cast(labels, LABEL_DTYPE)
        outputs = tf.squeeze(outputs)
        batch_loss = binary_cross_entropy_with_logits(labels, outputs)
        loss_metric.update_state(tf.cast(batch_loss, tf.float32))
        auc_metric.update_state(
            tf.cast(labels, tf.float32),
            tf.cast(tf.sigmoid(outputs), tf.float32),
        )

        predictions = tf.cast(outputs >= tf.cast(0.0, outputs.dtype), labels.dtype)

        num_samples += int(tf.shape(labels)[0].numpy())
        pos_samples += int(tf.reduce_sum(tf.cast(labels, tf.int64)).numpy())

        correct_preds = tf.cast(tf.equal(predictions, labels), tf.int64)
        num_correct += int(tf.reduce_sum(correct_preds).numpy())

        pos_mask = tf.equal(labels, tf.cast(1, labels.dtype))
        pos_correct += int(
            tf.reduce_sum(tf.cast(tf.boolean_mask(predictions, pos_mask), tf.int64)).numpy()
        )

    accuracy = float(num_correct / num_samples) if num_samples > 0 else 0.0
    recall_pos = float(pos_correct / pos_samples) if pos_samples > 0 else 0.0
    validation_loss = float(loss_metric.result().numpy())
    validation_auc = float(auc_metric.result().numpy())
    return validation_loss, validation_auc, accuracy, num_samples, recall_pos, pos_samples


####################################################################################################
#                                         TRAINING STEP                                            #
####################################################################################################
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        outputs, aux_logits = model(inputs, training=True, return_aux=True)
        labels = tf.cast(labels, LABEL_DTYPE)
        prediction_loss = binary_cross_entropy_with_logits(labels, tf.squeeze(outputs))
        if aux_logits:
            aux_losses = [
                binary_cross_entropy_with_logits(labels, tf.squeeze(aux_logit))
                for aux_logit in aux_logits
            ]
            aux_loss = tf.add_n(aux_losses) / len(aux_losses)
        else:
            aux_loss = tf.cast(0.0, dtype=prediction_loss.dtype)
        loss = prediction_loss + tf.cast(args.aux_loss_weight, prediction_loss.dtype) * aux_loss

    grads = tape.gradient(loss, model.trainable_variables)
    grads_and_vars = [
        (grad, var)
        for grad, var in zip(grads, model.trainable_variables)
        if grad is not None
    ]
    if args.grad_clip_norm and args.grad_clip_norm > 0 and grads_and_vars:
        clipped_grads, _ = tf.clip_by_global_norm(
            [grad for grad, _ in grads_and_vars], args.grad_clip_norm
        )
        grads_and_vars = [
            (grad, var) for grad, (_, var) in zip(clipped_grads, grads_and_vars)
        ]
    emb_grads = []
    other_grads = []

    for grad, var in grads_and_vars:
        if hasattr(var, "path"):
            if "sparse_embedding" in var.path and "embeddings" in var.name:
                emb_grads.append((grad, var))
            else:
                other_grads.append((grad, var))
        else:
            if "sparse_embedding" in var.name:
                emb_grads.append((grad, var))
            else:
                other_grads.append((grad, var))
    embedding_optimizer.apply_gradients(emb_grads)
    other_optimizer.apply_gradients(other_grads)

    return loss


####################################################################################################
#                                           TRAINING LOOP                                          #
####################################################################################################
step = 0
embedding_lr_metric = tf.keras.metrics.Mean()
other_lr_metric = tf.keras.metrics.Mean()

for epoch in range(TRAIN_EPOCHS):
    logger.info(f"Starting Epoch {epoch+1}/{TRAIN_EPOCHS}")
    epoch_start_time = time.perf_counter()
    train_phase_start_time = epoch_start_time
    epoch_loss_metric = tf.keras.metrics.Mean(dtype=tf.float32)
    train_iter_time_total = 0.0
    train_iter_count = 0

    for batch_idx, (inputs, labels) in enumerate(train_dataset):
        if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
            break
        iter_start_time = time.perf_counter()
        # Do NOT pre-cast labels to LABEL_DTYPE here. The MUSA plugin's _Arg
        # kernel is only registered for {float, double, half, int32, int64,
        # bool, resource}; passing bfloat16 labels into the @tf.function would
        # require _Arg(bfloat16). The cast lives inside train_step (where it is
        # a regular Cast op that MUSA does support for all dtypes), so labels
        # cross the function boundary as their natural int32 dtype.

        loss = train_step(inputs, labels)
        train_iter_time_total += time.perf_counter() - iter_start_time
        train_iter_count += 1
        loss_fp32 = tf.cast(loss, tf.float32)
        epoch_loss_metric.update_state(loss_fp32)
        current_lr = lr_schedule(step)
        current_lr_fp32 = tf.cast(current_lr, tf.float32)

        if (batch_idx + 1) % LOGGER_PRINT_INTERVAL == 0:
            logger.info(
                f"Epoch [{epoch+1}/{TRAIN_EPOCHS}], "
                f"Batch [{batch_idx+1}/{TOTAL_STEPS_PER_EPOCH}], "
                f"Loss: {float(loss_fp32.numpy()):.4f}, "
                f"LR: {float(current_lr_fp32.numpy()):.6f}"
            )

        with summary_writer.as_default():
            tf.summary.scalar("training_loss", loss_fp32, step=step)
            tf.summary.scalar("optimizer_lr", current_lr_fp32, step=step)

        step += 1

    train_loss = float(epoch_loss_metric.result().numpy())
    train_phase_seconds = time.perf_counter() - train_phase_start_time
    avg_train_iter_seconds = (
        train_iter_time_total / train_iter_count if train_iter_count > 0 else 0.0
    )
    valid_phase_start_time = time.perf_counter()
    validation_loss, validation_auc, accuracy, num_samples, recall_pos, pos_samples = validate(
        model, valid_dataset, args.max_valid_batches
    )
    valid_phase_seconds = time.perf_counter() - valid_phase_start_time
    epoch_total_seconds = time.perf_counter() - epoch_start_time

    logger.info(
        f"Epoch {epoch+1}/{TRAIN_EPOCHS} Summary: "
        f"Train Loss: {train_loss:.4f}, "
        f"Validation Loss: {validation_loss:.4f}, "
        f"Validation AUC: {validation_auc:.4f}, "
        f"Validation Accuracy: {float(accuracy)*100:.2f}%, "
        f"Total Samples: {num_samples}, "
        f"Positive Recall: {float(recall_pos)*100:.2f}%, "
        f"Positive Samples: {pos_samples}, "
        f"Avg Train Iter Time: {avg_train_iter_seconds:.4f}s, "
        f"Train Phase Time: {train_phase_seconds:.2f}s, "
        f"Valid Phase Time: {valid_phase_seconds:.2f}s, "
        f"Epoch Time: {epoch_total_seconds:.2f}s"
    )

    with summary_writer.as_default():
        tf.summary.scalar("epoch_train_loss", train_loss, step=epoch + 1)
        tf.summary.scalar("validation_loss", validation_loss, step=epoch + 1)
        tf.summary.scalar("validation_auc", validation_auc, step=epoch + 1)
        tf.summary.scalar("validation_accuracy", float(accuracy), step=epoch + 1)
        tf.summary.scalar("validation_recall_pos", float(recall_pos), step=epoch + 1)
        tf.summary.scalar("avg_train_iter_sec", avg_train_iter_seconds, step=epoch + 1)
        tf.summary.scalar("train_phase_sec", train_phase_seconds, step=epoch + 1)
        tf.summary.scalar("valid_phase_sec", valid_phase_seconds, step=epoch + 1)
        tf.summary.scalar("epoch_total_sec", epoch_total_seconds, step=epoch + 1)

    with open(metrics_csv_path, "a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(
            [
                epoch + 1,
                f"{train_loss:.6f}",
                f"{validation_loss:.6f}",
                f"{validation_auc:.6f}",
                f"{float(accuracy):.6f}",
                f"{float(recall_pos):.6f}",
                int(num_samples),
                int(pos_samples),
                f"{avg_train_iter_seconds:.6f}",
                f"{train_phase_seconds:.6f}",
                f"{valid_phase_seconds:.6f}",
                f"{epoch_total_seconds:.6f}",
            ]
        )

    if SAVE_CHECKPOINTS:
        ckpt_path = os.path.join(checkpoint_dir, f"tokenmixerlarge_epoch_{epoch+1}")
        model.save_weights(ckpt_path)
        logger.info(f"Model checkpoint saved for epoch {epoch+1} at {ckpt_path}")


####################################################################################################
#                                        TRAIN/VAL CURVES                                          #
####################################################################################################
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs_arr, train_arr, valid_arr, auc_arr, acc_arr = [], [], [], [], []
    with open(metrics_csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            epochs_arr.append(int(row["epoch"]))
            train_arr.append(float(row["train_loss"]))
            valid_arr.append(float(row["valid_loss"]))
            auc_arr.append(float(row["valid_auc"]))
            acc_arr.append(float(row["valid_accuracy"]))

    if epochs_arr:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        axes[0].plot(epochs_arr, train_arr, marker="o", label="Train loss")
        axes[0].plot(epochs_arr, valid_arr, marker="o", label="Valid loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("BCE loss")
        axes[0].set_title(f"TokenMixer-Large / {args.dataset} loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(epochs_arr, auc_arr, marker="o", color="tab:green", label="Valid AUC")
        axes[1].plot(
            epochs_arr,
            acc_arr,
            marker="s",
            color="tab:orange",
            linestyle="--",
            label="Valid accuracy",
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_title(f"TokenMixer-Large / {args.dataset} valid metrics")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(metrics_plot_path, dpi=130)
        plt.close(fig)
        logger.info(f"Saved train/val curves to {metrics_plot_path}")
except ImportError:
    logger.warning("matplotlib not available; skipping plot generation.")
