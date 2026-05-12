"""TokenMixer-Large in TensorFlow.

Faithful to the architecture described in:
    TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders
    (arXiv:2602.06563)

Key building blocks (Section 3 of the paper):
    - Tokenization (Eq. 1-4): per-group MLPs + global token from raw groups
    - TokenMixer-Large block: (Norm, Mix, S-P MoE, Revert, Norm, S-P MoE) (Eq. 12 + 16)
        * Pre-Norm with RMSNorm (Section 3.3.3, Section A.4)
        * Intra-mixing residual H_next = pSwiGLU(Norm(H)) + H (Eq. 12)
        * Reverting residual to original X: X_next = pSwiGLU(Norm(X_revert)) + X (Eq. 16)
    - Pertoken SwiGLU (Eq. 17-18): parameter-isolated per-token FCs
    - Sparse-Pertoken MoE (Section 3.4)
        * "First enlarge, then sparse": per-expert hidden = hidden_mult * D / num_experts
        * Shared expert always active (Eq. 20)
        * Gate Value Scaling alpha (Eq. 21)
        * Down-Matrix Small Init: stddev factor 0.01 (Section 3.4.4, Table 14)
    - Inter-residual + Auxiliary Loss (Section 3.3.4): skip-2 residuals with aux logits,
      never applied on the final layer
"""

from typing import List
import tensorflow as tf
from tensorflow.keras import layers

from model.mlp import MLP
from model.embedding import Embedding


class RMSNorm(layers.Layer):
    """RMSNorm (Section A.4, Eq. 22). No mean centering, no bias."""

    def __init__(self, dim, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.dim = dim

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale", shape=(self.dim,), initializer="ones", trainable=True
        )

    def call(self, x):
        norm = tf.reduce_mean(tf.pow(x, 2), axis=-1, keepdims=True)
        x = x * tf.math.rsqrt(norm + self.eps)
        return self.scale * x


class PerTokenDense(layers.Layer):
    """Per-token isolated linear: each token has its own (D_in -> D_out) kernel.

    Vectorized with a single einsum so the whole T-token bank runs in one GEMM.
    """

    def __init__(
        self,
        num_tokens,
        dim_in,
        dim_out,
        kernel_initializer="glorot_uniform",
        bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.use_bias = bias

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.num_tokens, self.dim_in, self.dim_out),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.num_tokens, self.dim_out),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None

    def call(self, x):
        x = tf.einsum("btd,tdh->bth", x, self.kernel)
        if self.bias is not None:
            x = x + self.bias
        return x


class PertokenSwiGLU(layers.Layer):
    """Per-token SwiGLU (Eq. 17-18).

    pSwiGLU(x) = FC_down(Swish(FC_gate(x)) * FC_up(x))

    The down-projection is initialized with a small variance so F(x) + x behaves
    like an approximate identity at the start of training (Section 3.4.4 / Table 14).
    The paper's best variant is stddev factor 0.01, which corresponds to
    VarianceScaling(scale=1e-4) since scale ~ stddev^2.
    """

    def __init__(
        self,
        dim,
        num_tokens,
        hidden_mult=4.0,
        down_init_scale=1e-4,
        bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        hidden_dim = max(1, int(round(dim * hidden_mult)))

        self.fc_up = PerTokenDense(num_tokens, dim, hidden_dim, bias=bias)
        self.fc_gate = PerTokenDense(num_tokens, dim, hidden_dim, bias=bias)
        self.fc_down = PerTokenDense(
            num_tokens,
            hidden_dim,
            dim,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=down_init_scale, mode="fan_avg", distribution="uniform"
            ),
            bias=bias,
        )

    def call(self, x):
        up = self.fc_up(x)
        gate_logits = self.fc_gate(x)
        gate = tf.nn.sigmoid(gate_logits) * gate_logits
        return self.fc_down(up * gate)


class SparsePertokenMoE(layers.Layer):
    """Sparse-Pertoken MoE (Section 3.4).

    "First enlarge, then sparse": each expert is a per-token SwiGLU with
    hidden = hidden_mult * D / num_experts so the total expert hidden capacity
    matches a single dense pSwiGLU.

    Routing (Eq. 21):
        S-P MoE(x) = alpha * sum_{i in TopK} g_i(x) * Expert_i(x) + SharedExpert(x)
    where g_i is a softmax over the top-(top_k - 1) routed experts.

    Note on training cost: TF/eager has no fused grouped-FFN kernel, so this
    implementation evaluates each routed expert once per call (hoisted) and
    masks the unused outputs. That is "dense compute, sparse output" — the
    paper's "Sparse Train, Sparse Infer" requires the custom MoEGroupedFFN
    operator described in Section 3.5.1, which is out of scope here.
    """

    def __init__(
        self,
        dim,
        num_tokens,
        num_experts=4,
        top_k=2,
        hidden_mult=4.0,
        alpha=2.0,
        bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_routed_experts = num_experts - 1
        self.top_k = min(top_k, num_experts)
        self.routed_top_k = min(max(self.top_k - 1, 0), self.num_routed_experts)
        self.alpha = alpha

        if self.num_routed_experts < 1:
            raise ValueError(
                "num_experts must include at least one routed expert and one shared expert"
            )

        per_expert_hidden_mult = float(hidden_mult) / float(num_experts)

        self.router = PerTokenDense(
            num_tokens, dim, self.num_routed_experts, bias=bias
        )
        self.experts = [
            PertokenSwiGLU(
                dim, num_tokens, per_expert_hidden_mult, bias=bias
            )
            for _ in range(self.num_routed_experts)
        ]
        self.shared_expert = PertokenSwiGLU(
            dim, num_tokens, per_expert_hidden_mult, bias=bias
        )

    def call(self, x):
        shared_out = self.shared_expert(x)

        if self.routed_top_k <= 0:
            return shared_out

        logits = self.router(x)
        topk_logits, topk_idx = tf.math.top_k(logits, k=self.routed_top_k)
        topk_probs = tf.nn.softmax(topk_logits, axis=-1)

        expert_outputs = tf.stack(
            [expert(x) for expert in self.experts], axis=2
        )  # [B, T, E_routed, D]

        selected = tf.gather(expert_outputs, topk_idx, batch_dims=2)  # [B, T, K, D]
        weighted = tf.expand_dims(topk_probs, axis=-1) * selected
        routed_out = self.alpha * tf.reduce_sum(weighted, axis=2)  # [B, T, D]

        return routed_out + shared_out


class TokenMixerLargeBlock(layers.Layer):
    """One TokenMixer-Large block (Figure 1 caption + Eq. 12, 16).

    Layout (Pre-Norm, no bias on linear kernels):

        H        = Mix(X)                                    (parameter-free)
        H_next   = MoE_mix(Norm(H)) + H                      (Eq. 12)
        X_revert = Revert(H_next)                            (parameter-free)
        X_next   = MoE_revert(Norm(X_revert)) + X            (Eq. 16)

    Mix splits each of T tokens into H heads (D/H wide) and concatenates
    the h-th head from all T tokens to form a new (T*D/H)-wide "head-token";
    Revert is the exact inverse, so the residual paths are dimensionally
    consistent at every layer (Section 3.3.1).
    """

    def __init__(
        self,
        dim,
        num_heads,
        num_tokens,
        num_experts=4,
        top_k=2,
        hidden_mult=4.0,
        alpha=2.0,
        bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.dim = dim
        self.num_tokens = num_tokens

        d = dim // num_heads
        mix_dim = num_tokens * d

        self.norm_mix = RMSNorm(mix_dim)
        self.mixing_moe = SparsePertokenMoE(
            mix_dim, num_heads, num_experts, top_k, hidden_mult, alpha, bias=bias
        )

        self.norm_revert = RMSNorm(dim)
        self.reverting_moe = SparsePertokenMoE(
            dim, num_tokens, num_experts, top_k, hidden_mult, alpha, bias=bias
        )

    def call(self, x):
        batch_size = tf.shape(x)[0]
        H = self.num_heads
        d = self.dim // H
        T = self.num_tokens

        x_split = tf.reshape(x, (batch_size, T, H, d))
        x_split = tf.transpose(x_split, perm=[0, 2, 1, 3])
        h_mat = tf.reshape(x_split, (batch_size, H, T * d))

        h_next = self.mixing_moe(self.norm_mix(h_mat)) + h_mat

        x_rev = tf.reshape(h_next, (batch_size, H, T, d))
        x_rev = tf.transpose(x_rev, perm=[0, 2, 1, 3])
        x_rev = tf.reshape(x_rev, (batch_size, T, self.dim))

        return self.reverting_moe(self.norm_revert(x_rev)) + x


class SemanticTokenizer(layers.Layer):
    """Group-wise tokenizer (Eq. 1-4).

    Each semantic group is concatenated and projected by its own MLP_i
    (Eq. 2). The global [CLS]-style token is built from the *raw* concatenation
    of all groups (Eq. 3), not from the post-projection tokens.

    Args:
        num_groups: number of semantic groups T-1 (the global token is added
            on top, giving T tokens in total).
    """

    def __init__(self, num_groups, model_dim, bias=False, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mlps = [
            tf.keras.Sequential(
                [
                    layers.Dense(model_dim, activation="relu", use_bias=bias),
                    layers.Dropout(dropout),
                    layers.Dense(model_dim, use_bias=bias),
                ]
            )
            for _ in range(num_groups)
        ]

        self.global_mlp = tf.keras.Sequential(
            [
                layers.Dense(model_dim, activation="relu", use_bias=bias),
                layers.Dropout(dropout),
                layers.Dense(model_dim, use_bias=bias),
            ]
        )

    def call(self, groups):
        tokens = []
        raw_concats = []
        for group_tensors, mlp in zip(groups, self.mlps):
            concat = tf.concat(group_tensors, axis=-1)
            raw_concats.append(concat)
            tokens.append(mlp(concat))

        stacked = tf.stack(tokens, axis=1)

        global_concat = tf.concat(raw_concats, axis=-1)
        global_token = self.global_mlp(global_concat)
        global_token = tf.expand_dims(global_token, axis=1)

        return tf.concat([global_token, stacked], axis=1)


class TokenMixerLarge(tf.keras.Model):
    """End-to-end TokenMixer-Large model (Figure 1).

    Stacks ``num_layers`` TokenMixerLargeBlock modules with inter-residual
    connections of ``inter_residual_gap`` (Section 3.3.4): every gap-th layer
    receives an extra residual from gap layers earlier. The final layer is
    excluded from inter-residuals and from auxiliary supervision because the
    paper observes that mixing low-level features back into the last layer
    hurts the abstraction needed for the prediction head.

    With ``return_aux=True`` the call returns (logits, [aux_logits]) where
    each aux logit comes from mean-pooling an inter-residual layer's output.
    The training loop combines them into a joint loss (Section 3.3.4).

    Note on T vs H: TokenMixer-Large explicitly decouples T (number of input
    tokens) from H (number of heads in the mix space) via Mix+Revert. The
    "T == H" constraint is a RankMixer artifact and does not apply here.
    """

    def __init__(
        self,
        feature_groups: List[List[int]],
        num_layers: int,
        num_sparse_embs: List[int],
        dim_input_sparse: int,
        dim_input_dense: int,
        dim_emb: int,
        num_heads: int,
        num_experts: int,
        top_k: int,
        num_hidden_head: int,
        dim_hidden_head: int,
        dim_output: int,
        dropout: float = 0.0,
        bias: bool = False,
        hidden_mult: float = 4.0,
        alpha: float = 2.0,
        inter_residual_gap: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if inter_residual_gap < 1:
            raise ValueError("inter_residual_gap must be >= 1")
        if not feature_groups:
            raise ValueError("feature_groups must contain at least one group")
        total_features = dim_input_sparse + dim_input_dense
        seen = set()
        for group in feature_groups:
            if not group:
                raise ValueError("each entry in feature_groups must be non-empty")
            for idx in group:
                if not 0 <= idx < total_features:
                    raise ValueError(
                        f"feature index {idx} is out of range "
                        f"[0, {total_features})"
                    )
                if idx in seen:
                    raise ValueError(f"feature index {idx} appears in multiple groups")
                seen.add(idx)
        self.embedding = Embedding(num_sparse_embs, dim_emb, dim_input_dense, bias)
        self.dim_emb = dim_emb
        self.dim_input_dense = dim_input_dense
        self.dim_input_sparse = dim_input_sparse
        self.num_layers = num_layers
        self.inter_residual_gap = inter_residual_gap
        self.feature_groups = [list(g) for g in feature_groups]
        self.tokenizer = SemanticTokenizer(
            len(self.feature_groups), dim_emb, bias, dropout
        )
        num_tokens = len(self.feature_groups) + 1
        self.blocks = [
            TokenMixerLargeBlock(
                dim_emb,
                num_heads,
                num_tokens,
                num_experts,
                top_k,
                hidden_mult,
                alpha,
                bias=bias,
            )
            for _ in range(num_layers)
        ]
        self.projection_head = MLP(
            dim_emb,
            num_hidden_head,
            dim_hidden_head,
            dim_output,
            dropout,
            bias,
        )
        self.aux_head = layers.Dense(dim_output, use_bias=bias)

    def build(self, input_shape):
        # All variables are created lazily on the first call(); this stub
        # keeps Keras from logging "build() was called on layer ... however
        # the layer does not have a build() method implemented" warnings.
        #
        # NOTE: Keras 2.13+ tightened tf.keras.Model.build() and it no longer
        # accepts a nested input_shape (a tuple / list of TensorShapes) the
        # way Keras 2.6 - 2.10 did.  Because our call() signature is
        #     call(self, inputs, ...)   with   inputs = (sparse, dense)
        # the framework passes input_shape as a 2-tuple of TensorShapes, which
        # the new super().build() tries to convert into a single TensorShape
        # and throws:
        #   TypeError: Error converting shape to a TensorShape: Dimension
        #   value must be integer or None or have an __index__ method, got
        #   value 'TensorShape([1, 26])' with type 'TensorShape'.
        # Skip super() entirely; setting self.built = True is the
        # documented pattern for lazily-built models.
        self.built = True

    def call(self, inputs, training=False, return_aux=False):
        sparse_inputs, dense_inputs = inputs
        x = self.embedding(sparse_inputs, dense_inputs)
        grouped = [
            [x[:, idx] for idx in group] for group in self.feature_groups
        ]
        x = self.tokenizer(grouped)

        last_layer = self.num_layers - 1
        gap = self.inter_residual_gap
        residual_history = []
        aux_logits = []

        for i, layer in enumerate(self.blocks):
            x = layer(x)

            apply_inter_residual = (
                i >= gap and (i % gap == 0) and i != last_layer
            )
            if apply_inter_residual:
                x = x + residual_history[i - gap]

            residual_history.append(x)

            if return_aux and apply_inter_residual:
                aux_pooled = tf.reduce_mean(x, axis=1)
                aux_logits.append(self.aux_head(aux_pooled))

        x = tf.reduce_mean(x, axis=1)
        x = self.projection_head(x)
        if return_aux:
            return x, aux_logits
        return x


if __name__ == "__main__":
    import numpy as np

    BATCH_SIZE = 2
    NUM_SPARSE_EMBS = [
        1460,
        583,
        10131227,
        2202608,
        305,
        24,
        12517,
        633,
        3,
        93145,
        5683,
        8351593,
        3194,
        27,
        14992,
        5461306,
        10,
        5652,
        2173,
        4,
        7046547,
        18,
        15,
        286181,
        105,
        142572,
    ]
    DIM_INPUT_SPARSE = 26
    DIM_INPUT_DENSE = 13

    feature_groups = [
        list(range(0, 13)),
        list(range(13, 26)),
        list(range(26, 39)),
    ]

    model = TokenMixerLarge(
        feature_groups=feature_groups,
        num_layers=6,
        num_sparse_embs=NUM_SPARSE_EMBS,
        dim_input_sparse=26,
        dim_input_dense=13,
        dim_emb=128,
        num_heads=4,
        num_experts=4,
        top_k=2,
        num_hidden_head=2,
        dim_hidden_head=256,
        dim_output=1,
        bias=False,
        hidden_mult=4.0,
        alpha=2.0,
        inter_residual_gap=2,
    )

    sparse_inputs = tf.constant(
        np.column_stack(
            [
                np.random.randint(0, high=NUM_SPARSE_EMBS[i], size=BATCH_SIZE)
                for i in range(DIM_INPUT_SPARSE)
            ]
        ).astype(np.int32)
    )
    dense_inputs = tf.constant(
        np.random.rand(BATCH_SIZE, DIM_INPUT_DENSE).astype(np.float32)
    )

    main_out, aux_outs = model(
        (sparse_inputs, dense_inputs), training=True, return_aux=True
    )
    print("Model main output shape:", main_out.shape)
    print(f"Number of auxiliary outputs: {len(aux_outs)}")
    for i, aux in enumerate(aux_outs):
        print(f"  aux[{i}] shape: {aux.shape}")
