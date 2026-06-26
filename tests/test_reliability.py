import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "models" / "reliability.py"
SPEC = importlib.util.spec_from_file_location("reliability_module", MODULE_PATH)
reliability_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(reliability_module)

ReliabilityModule = reliability_module.ReliabilityModule
apply_risk_reranking = reliability_module.apply_risk_reranking
compute_mode_risk_targets = reliability_module.compute_mode_risk_targets
compute_scene_risk_targets = reliability_module.compute_scene_risk_targets
compute_conflict_risk_targets = reliability_module.compute_conflict_risk_targets
compute_offroad_risk_targets = reliability_module.compute_offroad_risk_targets
build_reliability_targets = reliability_module.build_reliability_targets
reconstruct_lane_positions = reliability_module.reconstruct_lane_positions
summarize_reliability_targets = reliability_module.summarize_reliability_targets


def test_apply_risk_reranking_downweights_high_risk_modes():
    pi = torch.tensor([[0.0, 0.0, 0.0]])
    mode_risk = torch.tensor([[0.0, 1.0, 2.0]])

    reranked = apply_risk_reranking(pi=pi, mode_risk=mode_risk, alpha=1.0)

    assert reranked.shape == pi.shape
    assert torch.allclose(reranked.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert reranked[0, 0] > reranked[0, 1] > reranked[0, 2]


def test_compute_mode_and_scene_risk_targets_from_terminal_error():
    y_hat = torch.zeros(3, 3, 4, 2)
    y = torch.zeros(3, 4, 2)
    reg_mask = torch.tensor([
        [True, True, True, True],
        [True, True, True, True],
        [False, False, False, False],
    ])
    batch = torch.tensor([0, 0, 1])

    y_hat[1, 0, -1, 0] = 2.0
    y_hat[2, 1, -1, 1] = 1.5

    mode_targets, valid_mask, fde = compute_mode_risk_targets(
        y_hat=y_hat,
        y=y,
        reg_mask=reg_mask,
        fde_threshold=1.0,
    )
    scene_targets = compute_scene_risk_targets(mode_targets=mode_targets, batch=batch, valid_mask=valid_mask)

    assert valid_mask.tolist() == [True, True, False]
    assert mode_targets.tolist() == [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
    assert fde[0, 1].item() == 2.0
    assert fde[1, 2].item() == 1.5
    assert scene_targets.tolist() == [1.0, 0.0]


def test_compute_conflict_risk_targets_flags_close_future_pairs():
    y_hat = torch.zeros(2, 2, 3, 2)
    reg_mask = torch.tensor([
        [True, True, True],
        [True, True, True],
    ])
    batch = torch.tensor([0, 0])

    y_hat[0, 0] = torch.tensor([[0.0, 0.0], [0.2, 0.0], [0.3, 0.0]])
    y_hat[0, 1] = torch.tensor([[0.0, 1.0], [0.2, 0.1], [0.3, 0.1]])
    y_hat[1, 0] = torch.tensor([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]])
    y_hat[1, 1] = torch.tensor([[0.0, 6.0], [4.0, 6.0], [8.0, 6.0]])

    conflict_targets, min_pair_dist = compute_conflict_risk_targets(
        y_hat=y_hat,
        reg_mask=reg_mask,
        batch=batch,
        conflict_threshold=0.5,
    )

    assert conflict_targets.tolist() == [
        [1.0, 0.0],
        [1.0, 0.0],
    ]
    assert min_pair_dist[0, 0].item() < 0.5
    assert min_pair_dist[0, 1].item() > 1.0


def test_compute_offroad_risk_targets_flags_modes_far_from_lane_support():
    y_hat = torch.zeros(2, 2, 3, 2)
    reg_mask = torch.tensor([
        [True, True, True],
        [True, True, True],
    ])
    lane_positions = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
    ])
    lane_actor_index = torch.tensor([
        [0, 1, 2, 0, 1, 2],
        [0, 0, 0, 1, 1, 1],
    ])
    lane_actor_vectors = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [-10.0, 0.0],
        [-9.0, 0.0],
        [-8.0, 0.0],
    ])

    y_hat[0, 0] = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    y_hat[0, 1] = torch.tensor([[10.0, 0.0], [11.0, 0.0], [12.0, 0.0]])
    y_hat[1, 0] = torch.tensor([[6.0, 0.0], [7.0, 0.0], [8.0, 0.0]])
    y_hat[1, 1] = torch.tensor([[10.2, 0.0], [11.2, 0.0], [12.2, 0.0]])

    offroad_targets, lane_distance = compute_offroad_risk_targets(
        y_hat=y_hat,
        reg_mask=reg_mask,
        lane_positions=lane_positions,
        lane_actor_index=lane_actor_index,
        lane_actor_vectors=lane_actor_vectors,
        offroad_threshold=2.5,
    )

    assert offroad_targets.tolist() == [
        [0.0, 1.0],
        [1.0, 1.0],
    ]
    assert lane_distance[0, 0].item() < 0.1
    assert lane_distance[0, 1].item() > 2.5


def test_build_reliability_targets_combines_fde_conflict_and_offroad():
    y_hat = torch.zeros(2, 2, 3, 2)
    y = torch.zeros(2, 3, 2)
    reg_mask = torch.tensor([
        [True, True, True],
        [True, True, True],
    ])
    batch = torch.tensor([0, 0])
    lane_positions = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    lane_actor_index = torch.tensor([
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ])
    lane_actor_vectors = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [-8.0, 0.0],
        [-7.0, 0.0],
    ])

    y_hat[0, 0] = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    y_hat[0, 1] = torch.tensor([[0.0, 0.3], [0.1, 0.3], [0.2, 0.3]])
    y_hat[1, 0] = torch.tensor([[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]])
    y_hat[1, 1] = torch.tensor([[8.0, 0.0], [9.0, 0.0], [10.0, 0.0]])

    outputs = build_reliability_targets(
        y_hat=y_hat,
        y=y,
        reg_mask=reg_mask,
        batch=batch,
        lane_positions=lane_positions,
        lane_actor_index=lane_actor_index,
        lane_actor_vectors=lane_actor_vectors,
        fde_threshold=1.0,
        conflict_threshold=0.5,
        offroad_threshold=2.0,
    )

    assert outputs["mode_targets"].tolist() == [
        [1.0, 1.0],
        [1.0, 1.0],
    ]
    assert outputs["fde_targets"].tolist() == [
        [0.0, 1.0],
        [0.0, 1.0],
    ]
    assert outputs["conflict_targets"].tolist() == [
        [1.0, 0.0],
        [1.0, 0.0],
    ]
    assert outputs["offroad_targets"].tolist() == [
        [0.0, 0.0],
        [0.0, 1.0],
    ]
    assert outputs["scene_targets"].tolist() == [1.0]


def test_summarize_reliability_targets_reports_component_positive_rates():
    targets = {
        "mode_targets": torch.tensor([
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ]),
        "fde_targets": torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]),
        "conflict_targets": torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]),
        "offroad_targets": torch.tensor([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]),
        "scene_targets": torch.tensor([1.0, 0.0]),
        "valid_mask": torch.tensor([True, True, False]),
    }

    stats = summarize_reliability_targets(targets)

    assert stats["mode_positive_rate"].item() == 0.75
    assert stats["fde_positive_rate"].item() == 0.25
    assert stats["conflict_positive_rate"].item() == 0.5
    assert stats["offroad_positive_rate"].item() == 0.25
    assert stats["scene_positive_rate"].item() == 0.5


def test_reconstruct_lane_positions_from_lane_actor_vectors():
    current_positions = torch.tensor([
        [10.0, 5.0],
        [20.0, 5.0],
    ])
    lane_actor_index = torch.tensor([
        [0, 1, 2],
        [0, 0, 1],
    ])
    lane_actor_vectors = torch.tensor([
        [-1.0, 0.0],
        [0.0, 2.0],
        [3.0, -1.0],
    ])

    lane_positions = reconstruct_lane_positions(
        lane_actor_index=lane_actor_index,
        lane_actor_vectors=lane_actor_vectors,
        current_positions=current_positions,
        num_lanes=3,
    )

    assert lane_positions.tolist() == [
        [9.0, 5.0],
        [10.0, 7.0],
        [23.0, 4.0],
    ]


def test_reliability_module_returns_mode_and_scene_outputs():
    torch.manual_seed(7)
    num_modes = 3
    num_nodes = 4
    future_steps = 6
    embed_dim = 8
    batch = torch.tensor([0, 0, 1, 1])

    module = ReliabilityModule(
        embed_dim=embed_dim,
        future_steps=future_steps,
        num_modes=num_modes,
        hidden_dim=16,
        rerank_alpha=0.5,
    )
    local_embed = torch.randn(num_nodes, embed_dim)
    global_embed = torch.randn(num_modes, num_nodes, embed_dim)
    y_hat = torch.randn(num_modes, num_nodes, future_steps, 4)
    pi = torch.randn(num_nodes, num_modes)

    outputs = module(
        local_embed=local_embed,
        global_embed=global_embed,
        y_hat=y_hat,
        pi=pi,
        batch=batch,
    )

    assert outputs["mode_risk_logits"].shape == (num_nodes, num_modes)
    assert outputs["mode_risk"].shape == (num_nodes, num_modes)
    assert outputs["scene_risk_logits"].shape == (2,)
    assert outputs["scene_risk"].shape == (2,)
    assert outputs["reranked_pi"].shape == (num_nodes, num_modes)
    assert torch.all(outputs["mode_risk"] >= 0.0)
    assert torch.all(outputs["mode_risk"] <= 1.0)
    assert torch.all(outputs["scene_risk"] >= 0.0)
    assert torch.all(outputs["scene_risk"] <= 1.0)
    assert torch.allclose(outputs["reranked_pi"].sum(dim=-1), torch.ones(num_nodes), atol=1e-6)
