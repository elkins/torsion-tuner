import jax.numpy as jnp
from torsiontuner.train import train, Config

def test_train_smoke():
    """Smoke test to ensure the training loop runs for a few steps."""
    config = Config(n_steps=2) # Very few steps for speed
    model = train(config)
    assert model is not None
