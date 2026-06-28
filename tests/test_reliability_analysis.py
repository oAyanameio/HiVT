import importlib.util
import math
from pathlib import Path

import torch

from metrics.reliability_metrics import AUPRC
from metrics.reliability_metrics import AUROC
from metrics.reliability_metrics import BrierScore
from metrics.reliability_metrics import ECE


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_reliability_baselines.py"
SPEC = importlib.util.spec_from_file_location("reliability_analysis_module", MODULE_PATH)


def _load_module():
    module = importlib.util.module_from_spec(SPEC)
    assert SPEC.loader is not None
    SPEC.loader.exec_module(module)
    return module


def test_naive_risk_from_pi_gives_higher_score_to_lower_prob_mode():
    module = _load_module()
    pi = torch.tensor([[3.0, 1.0, -2.0]])
    risk = module.naive_risk_from_pi(pi)
    assert risk.shape == pi.shape
    assert risk[0, 0] < risk[0, 1] < risk[0, 2]


def test_spearman_rank_corr_returns_positive_one_for_identical_order():
    module = _load_module()
    x = torch.tensor([0.1, 0.2, 0.3, 0.4])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    corr = module.spearman_rank_corr(x, y)
    assert math.isclose(float(corr), 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_naive_risk_can_be_scored_with_existing_auroc_auprc():
    preds = torch.tensor([0.1, 0.8, 0.7, 0.2])
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0])
    auroc = AUROC(compute_on_step=False)
    auprc = AUPRC(compute_on_step=False)
    auroc.update(preds, targets)
    auprc.update(preds, targets)
    assert float(auroc.compute()) > 0.9
    assert float(auprc.compute()) > 0.9


def test_scene_metrics_accept_binary_scene_probs():
    preds = torch.tensor([0.2, 0.9, 0.8, 0.1])
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0])
    brier = BrierScore()
    ece = ECE()
    brier.update(preds, targets)
    ece.update(preds, targets)
    assert float(brier.compute()) < 0.1
    assert float(ece.compute()) < 0.2


def test_reranking_changes_top1_when_high_risk_mode_has_best_pi():
    pi = torch.tensor([[3.0, 2.0]])
    risk = torch.tensor([[1.0, 0.0]])
    reranked = torch.softmax(pi - risk * 2.0, dim=-1)
    assert int(pi.argmax(dim=-1)[0]) == 0
    assert int(reranked.argmax(dim=-1)[0]) == 1
