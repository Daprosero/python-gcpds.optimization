"""Regularized physics-informed neural network (RPINN) for nonlinear constrained optimization."""


def set_seed(seed: int = 1) -> None:
    """Seed Python, NumPy and TensorFlow so a run is reproducible."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
