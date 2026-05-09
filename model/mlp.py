import tensorflow as tf


class MLP(tf.keras.Sequential):
    def __init__(
        self,
        dim_in: int,
        num_hidden: int,
        dim_hidden: int,
        dim_out: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        layers = []

        for _ in range(num_hidden - 1):
            layers.append(tf.keras.layers.Dense(units=dim_hidden, use_bias=bias))
            layers.append(tf.keras.layers.BatchNormalization())
            layers.append(tf.keras.layers.ReLU())
            layers.append(tf.keras.layers.Dropout(dropout))

        layers.append(tf.keras.layers.Dense(units=dim_out, use_bias=bias))

        super().__init__(layers)
