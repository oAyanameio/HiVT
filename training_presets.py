from typing import Any, Dict


def build_reliability_train_args(
    embed_dim: int,
    reliability_hidden_dim: int = 128,
    reliability_rerank_alpha: float = 0.5,
    reliability_loss_weight: float = 1.0,
    scene_loss_weight: float = 0.5,
    rank_loss_weight: float = 0.0,
    calib_loss_weight: float = 0.0,
    risk_fde_threshold: float = 2.0,
    risk_conflict_threshold: float = 2.0,
    risk_offroad_threshold: float = 2.0,
    **overrides: Any,
) -> Dict[str, Any]:
    args: Dict[str, Any] = {
        "embed_dim": embed_dim,
        "use_reliability": True,
        "reliability_hidden_dim": reliability_hidden_dim,
        "reliability_rerank_alpha": reliability_rerank_alpha,
        "reliability_loss_weight": reliability_loss_weight,
        "scene_loss_weight": scene_loss_weight,
        "rank_loss_weight": rank_loss_weight,
        "calib_loss_weight": calib_loss_weight,
        "risk_fde_threshold": risk_fde_threshold,
        "risk_conflict_threshold": risk_conflict_threshold,
        "risk_offroad_threshold": risk_offroad_threshold,
    }
    args.update(overrides)
    return args
