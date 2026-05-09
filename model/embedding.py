import tensorflow as tf
from typing import List


class SparseEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_sparse_embs, dim_emb):
        super().__init__()
        self.embeddings = [
            tf.keras.layers.Embedding(input_dim=num_emb, output_dim=dim_emb)
            for num_emb in num_sparse_embs
        ]

    def call(self, sparse_inputs):
        sparse_outputs = [
            embedding(sparse_inputs[:, i])
            for i, embedding in enumerate(self.embeddings)
        ]
        return tf.stack(sparse_outputs, axis=1)


class Embedding(tf.keras.layers.Layer):
    def __init__(
        self,
        num_sparse_embs: List[int],
        dim_emb: int,
        dim_input_dense: int,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.dim_emb = dim_emb
        self.dim_input_dense = dim_input_dense

        self.sparse_embedding = SparseEmbedding(num_sparse_embs, dim_emb)
        self.dense_embedding = tf.keras.layers.Dense(
            units=dim_input_dense * dim_emb, use_bias=bias
        )

    def call(self, sparse_inputs: tf.Tensor, dense_inputs: tf.Tensor) -> tf.Tensor:
        sparse_outputs = self.sparse_embedding(sparse_inputs)

        dense_outputs = self.dense_embedding(dense_inputs)
        dense_outputs = tf.reshape(
            dense_outputs, [-1, self.dim_input_dense, self.dim_emb]
        )

        # concat along feature axis
        return tf.concat((sparse_outputs, dense_outputs), axis=1)
