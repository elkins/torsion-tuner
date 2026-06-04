from typing import Any

from torsiontuner.train import Config, train


def test_train_smoke() -> None:
    """Smoke test to ensure the training loop runs for a few steps."""
    config = Config(n_steps=2)  # Very few steps for speed
    model = train(config)
    assert model is not None


def test_train_default_config(monkeypatch: Any) -> None:
    """Test train() with default config (config=None)."""

    def mock_init(self: Any, **kwargs: Any) -> None:
        self.n_steps = 1
        self.learning_rate = 1e-3
        self.hidden_dim = 32
        self.n_layers = 3
        self.saxs_q_min = 0.0
        self.saxs_q_max = 0.5
        self.saxs_q_points = 50
        self.w_saxs = 1.0
        self.w_nmr = 1.0
        self.w_rama = 1.0
        self.w_reg = 0.1

    with monkeypatch.context() as m:
        m.setattr("torsiontuner.train.Config.__init__", mock_init)
        # We need to set other attrs too if we patch __init__
        for attr, val in [
            ("learning_rate", 1e-3),
            ("hidden_dim", 32),
            ("n_layers", 3),
            ("saxs_q_min", 0.0),
            ("saxs_q_max", 0.5),
            ("saxs_q_points", 50),
            ("w_saxs", 1.0),
            ("w_nmr", 1.0),
            ("w_rama", 1.0),
            ("w_reg", 0.1),
        ]:
            m.setattr(f"torsiontuner.train.Config.{attr}", val, raising=False)

        model = train(None)
        assert model is not None
