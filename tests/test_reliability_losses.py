import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "losses" / "reliability_losses.py"
SPEC = importlib.util.spec_from_file_location("reliability_losses_module", MODULE_PATH)
reliability_losses_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(reliability_losses_module)

RiskRankLoss = reliability_losses_module.RiskRankLoss


def test_risk_rank_loss_filters_to_top_k_and_near_threshold_pairs():
    loss_fn = RiskRankLoss(margin=0.1, error_margin=0.05)
    mode_risk = torch.tensor([[0.2, 0.1, 0.8, 0.05]])
    mode_error = torch.tensor([[1.95, 2.05, 3.0, 1.0]])
    valid_mask = torch.tensor([True])
    mode_logits = torch.tensor([[5.0, 4.0, 1.0, 0.5]])

    filtered = loss_fn(
        mode_risk=mode_risk,
        mode_error=mode_error,
        valid_mask=valid_mask,
        mode_logits=mode_logits,
        top_k=2,
        focus_threshold=2.0,
        focus_radius=0.15,
    )
    unfiltered = loss_fn(
        mode_risk=mode_risk,
        mode_error=mode_error,
        valid_mask=valid_mask,
    )

    assert torch.isclose(filtered, torch.tensor(0.2), atol=1e-6)
    assert torch.isclose(unfiltered, torch.tensor(0.0416666667), atol=1e-6)
    assert filtered > unfiltered


def test_risk_rank_loss_returns_zero_when_no_pairs_match_focus_filter():
    loss_fn = RiskRankLoss(margin=0.1, error_margin=0.05)
    mode_risk = torch.tensor([[0.2, 0.1, 0.8]])
    mode_error = torch.tensor([[1.0, 1.2, 3.0]])
    valid_mask = torch.tensor([True])
    mode_logits = torch.tensor([[4.0, 3.0, 2.0]])

    loss = loss_fn(
        mode_risk=mode_risk,
        mode_error=mode_error,
        valid_mask=valid_mask,
        mode_logits=mode_logits,
        top_k=2,
        focus_threshold=2.0,
        focus_radius=0.05,
    )

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
