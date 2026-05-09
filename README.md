# TokenMixer‑Large (TensorFlow / Keras)

A clean, paper‑faithful TensorFlow / Keras 3 implementation of **TokenMixer‑Large**, the per‑token Sparse‑MoE recommendation backbone introduced by ByteDance in:

> Yuchen Jiang, Jie Zhu, Xintian Han, Hui Lu, Kunmin Bai, Mingyu Yang, Shikang Wu, *et al.*
> **TokenMixer‑Large: Scaling Up Large Ranking Models in Industrial Recommenders.**
> arXiv:2602.06563 (2026). [[arXiv]](https://arxiv.org/abs/2602.06563)

The repo ships two end‑to‑end training pipelines on **widely used recommendation benchmarks**:

| Dataset | Task | Loader | Source |
|---|---|---|---|
| Criteo Kaggle Display Ads (45.8 M rows × 26 cat + 13 dense) | binary CTR | `data/criteo_kaggle_dataset.py` | [Criteo Kaggle](https://www.kaggle.com/c/criteo-display-ad-challenge) |
| Amazon Beauty Reviews (701 k rows) | binary `rating ≥ 4` | `data/amazon_beauty_dataset.py` | [HuggingFace `jhan21/amazon-beauty-reviews-dataset`](https://huggingface.co/datasets/jhan21/amazon-beauty-reviews-dataset) |

It runs natively on standard NVIDIA TensorFlow (`--backend cuda`, the default) and optionally on Moore Threads GPUs via the [TensorFlow MUSA Extension](https://github.com/MooreThreads/tensorflow_musa_extension) (`--backend musa`).

---

## Architecture

The implementation reproduces the paper's core building blocks (Section 3):

- **Group‑wise tokenization with a global token** (Eq. 1‑4). Each semantic feature group gets its own MLP_i, plus a BERT‑style global token built from the *raw* concatenation of every group.
- **Mixing & Reverting block** (Section 3.3.1, Eq. 12 + 16). Two symmetric per‑token sub‑layers around the parameter‑free Mix / Revert transforms, each with **Pre‑Norm + intra‑sub‑layer residual** so signal paths stay dimensionally consistent at every depth.
- **Per‑token SwiGLU** (Section 3.3.2, Eq. 17‑18) — parameter‑isolated up/gate/down projections, one bank per token position.
- **Sparse‑Pertoken MoE** (Section 3.4) — *first enlarge then sparse* fine‑grained experts, plus a per‑token shared expert that is always active, plus Gate‑Value Scaling α = num_experts/top_k (Eq. 21), plus Down‑Matrix Small Init (variance scale 1e‑4 ⇔ stddev factor 0.01) for stable convergence at depth.
- **Inter‑residual + Auxiliary Loss** (Section 3.3.4) — extra skip residuals at intervals of 2 (configurable via `--inter_residual_gap`), excluded from the final layer, with auxiliary BCE losses computed from intermediate layer logits and joined to the main loss with `--aux_loss_weight`.
- **RMSNorm + no bias** (Section 3.3.3, Section A.4) — Llama‑style. Bias defaults to off; flip with `--bias` if you need it.

```
inputs (sparse_ids, dense_features)
    │
    ▼
Embedding ──► group MLPs + global MLP ──► tokens X ∈ R[B, T, D]
    │
    ▼
TokenMixerLargeBlock × num_layers
    │   ┌───────────────────────────────────────────────────┐
    │   │   H        = Mix(X)                               │
    │   │   H_next   = MoE_mix( Norm(H) )      + H          │  Eq. 12  (Pre‑Norm)
    │   │   X_revert = Revert(H_next)                       │
    │   │   X_next   = MoE_revert( Norm(X_revert) ) + X     │  Eq. 16  (Pre‑Norm)
    │   └───────────────────────────────────────────────────┘
    │   inter‑residual ──► aux head  (Section 3.3.4)
    ▼
mean‑pool over tokens → projection head → logit
```

---

## Repository layout

```
tokenmixerlarge/
├── README.md
├── LICENSE                  # Apache 2.0
├── requirements.txt
├── train.py                 # Single‑file training entry point
├── train_cuda.py            # Convenience wrapper: equivalent to `train.py --backend cuda`
├── model/
│   ├── tokenmixerlarge.py   # TokenMixerLarge model + Mix‑Revert block + S‑P MoE
│   ├── embedding.py         # Sparse / dense feature embedding tables
│   ├── lr_schedule.py       # LinearWarmup, WarmupCosine
│   └── mlp.py               # Projection‑head MLP
├── data/
│   ├── amazon_beauty_dataset.py  # Arrow ➜ tf.data
│   ├── criteo_kaggle_dataset.py  # NPZ ➜ tf.data
│   ├── prepare_avazu_small.py    # Build a small NPZ from raw Avazu CSV
│   └── prepare_criteo_small.py   # Build a small NPZ from raw Criteo TSV
└── dataset/                 # (gitignored) drop your local datasets here
```

---

## Installation

```bash
git clone <YOUR_FORK_URL> tokenmixerlarge
cd tokenmixerlarge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For the Moore Threads MUSA backend, follow the
[TensorFlow MUSA Extension](https://github.com/MooreThreads/tensorflow_musa_extension)
build instructions and pass `--backend musa --lib_path /path/to/libmusa_plugin.so`.

---

## Datasets

Both loaders look for files under `dataset/` by default:

```
dataset/criteo_data.npz                              # Criteo Kaggle (X_cat, X_int, y, counts)
dataset/amazon-beauty-reviews-dataset-train.arrow    # HuggingFace Arrow file
```

### Criteo

`criteo_data.npz` follows the `{X_cat[N,26]:int32, X_int[N,13]:int32, y[N]:int32, counts[26]:int32}` convention used by most public Criteo notebooks (e.g. the FB DLRM repo). If you only have the raw TSV, use the small‑subset helper:

```bash
python data/prepare_criteo_small.py \
    --input  /path/to/criteo/train.txt \
    --output dataset/criteo_data.npz \
    --rows 5_000_000
```

### Amazon Beauty Reviews

```bash
# Either: download the HuggingFace dataset and place its arrow file under dataset/
huggingface-cli download jhan21/amazon-beauty-reviews-dataset --local-dir dataset/

# Or: point --data_path to your existing copy.
```

The loader expects a column schema of
`{rating, title, text, images, asin, parent_asin, user_id, timestamp, helpful_vote, verified_purchase}`.

---

## Quick start

### Amazon Beauty Reviews (10 epochs, ≈ 3 min on a single RTX 4090)

```bash
python train.py \
    --dataset amazon_beauty \
    --feature_grouping semantic \
    --lr_schedule warmup_cosine \
    --epochs 10 --batch_size 4096 \
    --train_size 600000 --valid_size 100000 \
    --num_layers 6 --dim_emb 128 --num_heads 8 \
    --num_experts 4 --top_k 2 --alpha 2.0 \
    --inter_residual_gap 2 --aux_loss_weight 0.1 \
    --dropout 0.2 --peak_lr 1e-3 --min_lr 1e-5 \
    --grad_clip_norm 1.0 --seed 42
```

### Criteo Kaggle (5 epochs, ≈ 22 min on a single RTX 4090)

```bash
python train.py \
    --dataset criteo \
    --derive_sparse_embs_from_data \
    --feature_grouping coarse \
    --lr_schedule warmup_cosine \
    --epochs 5 --batch_size 8192 \
    --train_size 10000000 --valid_size 1000000 \
    --num_layers 6 --dim_emb 64 --num_heads 8 \
    --num_experts 4 --top_k 2 --alpha 2.0 \
    --inter_residual_gap 2 --aux_loss_weight 0.1 \
    --dropout 0.2 --peak_lr 1e-3 --min_lr 1e-5 \
    --grad_clip_norm 1.0 --seed 42
```

Each invocation produces a stamped run directory under `logs/<YYYY‑MM‑DD‑HH.MM.SS>/` containing:

| File | What |
|---|---|
| `training.log` | full per‑step / per‑epoch log |
| `metrics.csv` | per‑epoch `train_loss, valid_loss, valid_auc, valid_acc, …` |
| `train_val_curves.png` | auto‑generated matplotlib curves (loss + AUC + accuracy) |
| `tensorboard/` | `events.out.tfevents…` for `tensorboard --logdir …` |
| `train.py`, `model/`, `data/` | frozen copy of the code that produced the run |

---

## Reference results (single RTX 4090)

These come from runs in this repo with the configurations above. They are *not* paper numbers — the paper trains on proprietary Douyin tables an order of magnitude bigger and at 7 B / 15 B parameters. They are reasonable references when you wire in your own data.

### Amazon Beauty Reviews (binary `rating ≥ 4`, prior 67.85 % positive)

| Run | Best valid AUC | Best valid acc | Notes |
|---|---|---|---|
| Coarse 2‑group split, constant LR, 4 sparse + 2 dense | 0.6243 | 68.66 % | initial baseline |
| **5 semantic groups + 4 sparse + 5 dense + cosine LR + dropout 0.2** | **0.6765** | **69.87 %** | this repo's defaults |

### Criteo Kaggle (binary click, prior 25.77 % positive, 10 M train / 1 M valid)

| Epoch | train_loss | valid_loss | valid_auc | valid_acc |
|---:|---:|---:|---:|---:|
| 1 | 0.5577 | 0.4740 | 0.7725 | 77.97 % |
| 2 | 0.5149 | 0.4731 | 0.7787 | 78.02 % |
| 3 | 0.5093 | 0.4667 | 0.7822 | 78.37 % |
| 4 | 0.5053 | 0.4659 | 0.7842 | 78.42 % |
| **5** | **0.5025** | **0.4641** | **0.7852** | **78.51 %** |

Validation loss and AUC are still improving at epoch 5 — longer training and the full 39 M training partition will close the gap with the published Wukong / DCNv2 / AutoInt numbers.

---

## Configuration cheat sheet

The training loop exposes every relevant paper hyperparameter as a CLI flag.

| Flag | Default | Paper section / equation |
|---|---|---|
| `--num_layers` | 6 | §3.3 (depth) |
| `--dim_emb` | 128 | §3.2 (D) |
| `--num_heads` | 16 | §3.3.1 (H — decoupled from T thanks to Mix & Revert) |
| `--num_experts` | 8 | §3.4 (total experts = 1 shared + N‑1 routed) |
| `--top_k` | 4 | §3.4 (sparsity = (top_k‑1+1) / num_experts; default ⇒ 1:2) |
| `--alpha` | 2.0 | §3.4.3 Eq. 21 (default = num_experts / top_k) |
| `--hidden_mult` | 4.0 | §3.3.2 (n in `n·D`); per‑expert hidden = `n·D / num_experts` (first‑enlarge‑then‑sparse) |
| `--inter_residual_gap` | 2 | §3.3.4 ("intervals of 2 or 3 layers") |
| `--aux_loss_weight` | 0.1 | §3.3.4 (joint loss term) |
| `--bias` | off | §A.4 (Llama‑style biases removed) |
| `--lr_schedule` | `warmup_cosine` | smoother than constant peak after warmup |
| `--feature_grouping` | `semantic` | semantic groups when the dataset exposes them, falls back to `coarse` (sparse vs dense) for Criteo |

---

## Compliance with the paper

This implementation is paper‑faithful in the load‑bearing places:

- ✅ Mix & Revert with **intra‑mixing residual** (`H_next = MoE(Norm(H)) + H`, Eq. 12) and reverting residual to original X (Eq. 16).
- ✅ Two MoE layers per block exactly — `(Norm, Mixing, S‑P MoE, Reverting, Norm, S‑P MoE)` per Figure 1.
- ✅ Pre‑Norm + RMSNorm everywhere (§3.3.3 / §A.4).
- ✅ Shared expert always active, gate‑value scaling α (Eq. 21).
- ✅ Down‑Matrix Small Init at stddev factor 0.01 (variance scale 1e‑4, the best variant in Table 14).
- ✅ Inter‑residual at gap 2 with the last layer excluded (§3.3.4).
- ✅ "First enlarge, then sparse" per‑expert hidden = `n·D / num_experts` (§3.4.1).

Documented deviations:

- ⚠️ The Sparse‑Pertoken MoE forward is **dense compute, sparse output**. The paper's "Sparse Train, Sparse Infer" relies on the custom `MoEPermute` / `MoEGroupedFFN` operators (§3.5.1) that aren't reproducible in vanilla TF. Each routed expert is evaluated once per call (hoisted) and unused outputs are masked.
- ⚠️ Optimizer is SGD (sparse params) + Adam (dense params) with linear warmup; the paper uses Adagrad LR 0.05 / 0.01 (§4.1.1). Optimizer choice was left independent of the architecture refactor.

---

## Citation

```bibtex
@article{jiang2026tokenmixerlarge,
  title   = {TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders},
  author  = {Jiang, Yuchen and Zhu, Jie and Han, Xintian and Lu, Hui and Bai, Kunmin
             and Yang, Mingyu and Wu, Shikang and Zhang, Ruihao and Zhao, Wenlin
             and Bai, Shipeng and Zhou, Sijin and Yang, Huizhi and Liu, Tianyi
             and Liu, Wenda and Gong, Ziyan and Ding, Haoran and Chai, Zheng
             and Xie, Deping and Chen, Zhe and Zheng, Yuchao and Xu, Peng},
  journal = {arXiv preprint arXiv:2602.06563},
  year    = {2026}
}
```

The original TokenMixer/RankMixer is described in:

```bibtex
@inproceedings{zhu2025rankmixer,
  title     = {RankMixer: Scaling Up Ranking Models in Industrial Recommenders},
  author    = {Zhu, Jie and Fan, Zhifang and Zhu, Xiaoxie and Jiang, Yuchen
               and Wang, Hangyu and Han, Xintian and Ding, Haoran and Wang, Xinmin
               and Zhao, Wenlin and Gong, Zhen and others},
  booktitle = {Proceedings of the 34th ACM International Conference on
               Information and Knowledge Management},
  year      = {2025}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
