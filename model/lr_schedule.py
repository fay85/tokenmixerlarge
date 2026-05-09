import math

import tensorflow as tf


class LinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup, then constant peak LR."""

    def __init__(self, initial_learning_rate, peak_learning_rate, warmup_steps):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.peak_learning_rate = peak_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        scale = tf.minimum(step / tf.maximum(warmup, 1.0), 1.0)
        return (
            self.initial_learning_rate
            + (self.peak_learning_rate - self.initial_learning_rate) * scale
        )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "peak_learning_rate": self.peak_learning_rate,
            "warmup_steps": self.warmup_steps,
        }


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay to ``min_learning_rate``.

    Stable late-training behaviour and avoids the stair-step instability that
    a constant peak LR can trigger right after warmup ends. Decay starts at
    ``warmup_steps`` and finishes at ``total_steps``.
    """

    def __init__(
        self,
        initial_learning_rate,
        peak_learning_rate,
        warmup_steps,
        total_steps,
        min_learning_rate=0.0,
    ):
        super().__init__()
        if total_steps <= warmup_steps:
            raise ValueError("total_steps must be greater than warmup_steps")
        self.initial_learning_rate = initial_learning_rate
        self.peak_learning_rate = peak_learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_learning_rate = min_learning_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total = tf.cast(self.total_steps, tf.float32)
        decay_steps = tf.maximum(total - warmup, 1.0)

        warmup_scale = tf.minimum(step / tf.maximum(warmup, 1.0), 1.0)
        warmup_lr = (
            self.initial_learning_rate
            + (self.peak_learning_rate - self.initial_learning_rate) * warmup_scale
        )

        progress = tf.minimum(tf.maximum(step - warmup, 0.0) / decay_steps, 1.0)
        cos_factor = 0.5 * (1.0 + tf.cos(tf.constant(math.pi, dtype=tf.float32) * progress))
        cosine_lr = self.min_learning_rate + (
            self.peak_learning_rate - self.min_learning_rate
        ) * cos_factor

        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "peak_learning_rate": self.peak_learning_rate,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_learning_rate": self.min_learning_rate,
        }
