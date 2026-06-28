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
RERANK_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_reranking.py"
RERANK_SPEC = importlib.util.spec_from_file_location("reranking_eval_module", RERANK_MODULE_PATH)


def _load_module():
    module = importlib.util.module_from_spec(SPEC)
    assert SPEC.loader is not None
    SPEC.loader.exec_module(module)
    return module


def _load_rerank_module():
    module = importlib.util.module_from_spec(RERANK_SPEC)
    assert RERANK_SPEC.loader is not None
    RERANK_SPEC.loader.exec_module(module)
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


def test_probability_space_reranking_penalizes_high_risk_mode():
    module = _load_rerank_module()
    pi = torch.tensor([[3.0, 2.0]])
    risk = torch.tensor([[0.9, 0.1]])
    scores = module.rerank_scores(pi, risk, method="prob_product", alpha=1.0, top_k=None)
    assert scores.shape == pi.shape
    assert scores[0, 1] > scores[0, 0]


def test_top_k_reranking_only_changes_candidates_inside_top_k():
    module = _load_rerank_module()
    pi = torch.tensor([[5.0, 4.0, 0.0]])
    risk = torch.tensor([[0.9, 0.1, 0.0]])
    scores = module.rerank_scores(pi, risk, method="prob_product", alpha=1.0, top_k=2)
    assert scores[0, 2] < scores[0, 1]


def test_probability_space_reranking_alpha_zero_reduces_to_log_softmax():
    module = _load_rerank_module()
    pi = torch.tensor([[2.0, 1.0]])
    risk = torch.tensor([[0.9, 0.1]])
    scores = module.rerank_scores(pi, risk, method="prob_product", alpha=0.0, top_k=None)
    assert torch.allclose(scores, torch.log_softmax(pi, dim=-1), atol=1e-6)


def test_summarize_reranking_cases_separates_threshold_crossings():
    module = _load_rerank_module()
    original_fde = torch.tensor([1.0, 2.5, 2.8, 1.8])
    reranked_fde = torch.tensor([2.2, 1.9, 2.1, 1.2])
    summary = module.summarize_reranking_cases(
        original_fde=original_fde,
        reranked_fde=reranked_fde,
        miss_threshold=2.0,
    )

    assert summary["count"] == 4
    assert math.isclose(summary["original_mr"], 0.5, abs_tol=1e-6)
    assert math.isclose(summary["reranked_mr"], 0.5, abs_tol=1e-6)
    assert summary["hit_to_miss_count"] == 1
    assert summary["miss_to_hit_count"] == 1
    assert summary["still_miss_improved_count"] == 1
    assert summary["still_hit_improved_count"] == 1
    assert summary["still_miss_worsened_count"] == 0
    assert summary["still_hit_worsened_count"] == 0


def test_summarize_reranking_cases_tracks_unchanged_cases_separately():
    module = _load_rerank_module()
    original_fde = torch.tensor([1.0, 2.5, 1.5, 3.0])
    reranked_fde = torch.tensor([1.0, 2.5, 1.8, 3.2])
    summary = module.summarize_reranking_cases(
        original_fde=original_fde,
        reranked_fde=reranked_fde,
        miss_threshold=2.0,
    )

    assert summary["still_hit_unchanged_count"] == 1
    assert summary["still_miss_unchanged_count"] == 1
    assert summary["still_hit_worsened_count"] == 1
    assert summary["still_miss_worsened_count"] == 1
