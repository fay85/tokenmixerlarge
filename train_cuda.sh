# CUDA counterpart of train.sh, configured for the tightest reproducible
# accuracy comparison with the MUSA run.  Every flag that has any effect on
# numerics or RNG is held identical to the MUSA script.
#
# What's deliberately the same as train.sh:
#   - dataset, batch_size, train/valid_size, seed
#   - --precision mixed_bf16  (bf16 compute, fp32 master weights + Adam state)
#   - --deterministic_alignment, TF_DETERMINISTIC_OPS=1
#   - --disable_tf32  (CRITICAL: MUSA has no TF32; disabling it on CUDA
#     forces matmuls to use full fp32 Tensor-Core accumulation, matching
#     MUSA's accumulator behavior)
#
# What's CUDA-only:
#   - TF_CUDNN_DETERMINISTIC=1     -> cuDNN picks deterministic convolution /
#                                     reduction algorithms (counterpart of the
#                                     deterministic kernels MUSA selects when
#                                     TF_DETERMINISTIC_OPS=1).
#   - CUBLAS_WORKSPACE_CONFIG      -> cuBLAS GEMM determinism requires either
#                                     :4096:8 (default) or :16:8; without one
#                                     of these set, cuBLAS may pick different
#                                     kernels run-to-run.
#   - PYTHONHASHSEED=0             -> kills Python hash randomization which
#                                     can otherwise shift dict-iteration
#                                     order in sparse-feature handling.
#   - CUDA_VISIBLE_DEVICES         -> picks the GPU (the MUSA script uses
#                                     --musa_device_index 3 which sets
#                                     MUSA_VISIBLE_DEVICES pre-import).
#                                     Adjust the ordinal to whichever CUDA
#                                     device you actually want.
#
# What's deliberately NOT carried over from train.sh:
#   - --lib_path                   -> CUDA path doesn't load the MUSA plugin.
#   - --musa_device_index          -> use CUDA_VISIBLE_DEVICES instead.
#
# Numerical caveat: even with all of the above pinned, an exact bit-match
# between the two runs is impossible because (a) cuDNN and muDNN implement
# bf16 ops with different reduction trees and (b) the optimizer paths
# diverge slightly -- MUSA uses MusaResourceApplyAdamMixed (no Cast),
# CUDA uses the stock Adam path with an explicit Cast(bf16->fp32) inserted
# by Keras's mixed_bfloat16 policy.  Loss curves should align to ~3 sig figs;
# final eval metrics typically within +/- 0.1% point.

CUDA_VISIBLE_DEVICES=0 \
TF_DETERMINISTIC_OPS=1 \
TF_CUDNN_DETERMINISTIC=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTHONHASHSEED=0 \
python train.py \
    --backend cuda \
    --precision mixed_bf16 \
    --dataset criteo \
    --seed 42 \
    --deterministic_alignment \
    --disable_tf32 \
    --batch_size 2048 \
    --derive_sparse_embs_from_data \
    --train_size 5000000 \
    --valid_size 500000
