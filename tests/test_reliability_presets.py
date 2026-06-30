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
    assert args["reliability_rerank_alpha"] == 0.0
    assert args["reliability_loss_weight"] == 1.0
    assert args["scene_loss_weight"] == 0.2
    assert args["risk_fde_threshold"] == 2.0
    assert args["risk_miss_threshold"] == 4.0
    assert args["mode_target_policy"] == "fde_only"
    assert args["scene_target_policy"] == "target_best_mode_fail"
    assert args["risk_conflict_min_frames"] == 2
    assert args["risk_conflict_scope"] == "target_to_neighbors"
    assert args["risk_scene_rate_threshold"] == 0.5


def test_build_reliability_train_args_accepts_overrides():
    args = training_presets.build_reliability_train_args(
        embed_dim=128,
        reliability_hidden_dim=256,
        reliability_rerank_alpha=0.8,
        reliability_loss_weight=1.5,
        scene_loss_weight=0.7,
        risk_fde_threshold=1.2,
        risk_miss_threshold=5.0,
        mode_target_policy="miss_only",
        scene_target_policy="target_mode_rate",
        risk_conflict_min_frames=3,
        risk_conflict_scope="all_valid_pairs",
        risk_scene_rate_threshold=0.3,
    )

    assert args["embed_dim"] == 128
    assert args["reliability_hidden_dim"] == 256
    assert args["reliability_rerank_alpha"] == 0.8
    assert args["reliability_loss_weight"] == 1.5
    assert args["scene_loss_weight"] == 0.7
    assert args["risk_fde_threshold"] == 1.2
    assert args["risk_miss_threshold"] == 5.0
    assert args["mode_target_policy"] == "miss_only"
    assert args["scene_target_policy"] == "target_mode_rate"
    assert args["risk_conflict_min_frames"] == 3
    assert args["risk_conflict_scope"] == "all_valid_pairs"
    assert args["risk_scene_rate_threshold"] == 0.3


def test_build_reliability_train_args_accepts_freeze_backbone_flag():
    args = training_presets.build_reliability_train_args(
        embed_dim=64,
        freeze_backbone=True,
    )

    assert args["freeze_backbone"] is True


def test_build_reliability_train_args_accepts_threshold_objective_overrides():
    args = training_presets.build_reliability_train_args(
        embed_dim=64,
        mode_risk_threshold_weight_enabled=True,
        mode_risk_threshold_weight_radius=0.2,
        mode_risk_threshold_weight_peak=3.0,
        mode_risk_threshold_weight_base=1.25,
        mode_risk_rank_top_k=3,
        mode_risk_rank_near_threshold_only=True,
        mode_risk_rank_threshold_radius=0.15,
    )

    assert args["mode_risk_threshold_weight_enabled"] is True
    assert args["mode_risk_threshold_weight_radius"] == 0.2
    assert args["mode_risk_threshold_weight_peak"] == 3.0
    assert args["mode_risk_threshold_weight_base"] == 1.25
    assert args["mode_risk_rank_top_k"] == 3
    assert args["mode_risk_rank_near_threshold_only"] is True
    assert args["mode_risk_rank_threshold_radius"] == 0.15
