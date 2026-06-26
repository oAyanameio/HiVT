import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "training_presets.py"
SPEC = importlib.util.spec_from_file_location("training_presets_module", MODULE_PATH)
training_presets = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(training_presets)


def test_build_reliability_train_args_enables_reliability_flags():
    args = training_presets.build_reliability_train_args(embed_dim=64)

    assert args["embed_dim"] == 64
    assert args["use_reliability"] is True
    assert args["reliability_hidden_dim"] == 128
    assert args["reliability_rerank_alpha"] == 0.5
    assert args["reliability_loss_weight"] == 1.0
    assert args["scene_loss_weight"] == 0.5
    assert args["risk_fde_threshold"] == 2.0


def test_build_reliability_train_args_accepts_overrides():
    args = training_presets.build_reliability_train_args(
        embed_dim=128,
        reliability_hidden_dim=256,
        reliability_rerank_alpha=0.8,
        reliability_loss_weight=1.5,
        scene_loss_weight=0.7,
        risk_fde_threshold=1.2,
    )

    assert args["embed_dim"] == 128
    assert args["reliability_hidden_dim"] == 256
    assert args["reliability_rerank_alpha"] == 0.8
    assert args["reliability_loss_weight"] == 1.5
    assert args["scene_loss_weight"] == 0.7
    assert args["risk_fde_threshold"] == 1.2
