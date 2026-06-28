#!/usr/bin/env python
"""Audit InterAct-Risk target rates without backpropagation."""
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict
import sys
import warnings


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


warnings.filterwarnings(
    'ignore',
    message=r'pkg_resources is deprecated as an API.*',
    category=DeprecationWarning,
)
warnings.filterwarnings(
    'ignore',
    message=r"Deprecated call to `pkg_resources\\.declare_namespace\\('.*'\\)`\\.",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    'ignore',
    message=r'torch\\.distributed\\._sharded_tensor will be deprecated.*',
    category=DeprecationWarning,
)

import pytorch_lightning as pl
import torch

from datamodules import ArgoverseV1DataModule
from models import build_reliability_targets
from models import reconstruct_lane_positions
from models import summarize_reliability_targets
from models.hivt import HiVT


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _accumulate(totals: Dict[str, float], stats: Dict[str, torch.Tensor]) -> None:
    for key, value in stats.items():
        totals[key] = totals.get(key, 0.0) + _scalar(value)


def _print_table(totals: Dict[str, float], count: int) -> None:
    if count == 0:
        print("No batches were audited.")
        return
    print("metric,value")
    for key in sorted(totals):
        print("{},{}".format(key, totals[key] / count))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--max_batches', type=int, default=8)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--persistent_workers', type=bool, default=False)
    parser.add_argument('--gpus', type=int, default=0)
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    pl.seed_everything(2022)
    device = torch.device('cuda:0' if args.gpus > 0 and torch.cuda.is_available() else 'cpu')

    datamodule = ArgoverseV1DataModule.from_argparse_args(args)
    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.train_dataloader() if args.split == 'train' else datamodule.val_dataloader()

    if args.ckpt_path:
        model = HiVT.load_from_checkpoint(
            checkpoint_path=args.ckpt_path,
            map_location=device,
            strict=False,
            **vars(args),
        )
        model = model.to(device)
    else:
        model = HiVT(**vars(args)).to(device)
    model.eval()

    totals: Dict[str, float] = {}
    pred_totals: Dict[str, float] = {}
    audited = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            data = data.to(device)
            y_hat, _, reliability_outputs = model(data)
            reg_mask = ~data['padding_mask'][:, model.historical_steps:]
            graph_batch = getattr(data, 'batch', None)
            if graph_batch is None:
                graph_batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            current_positions = data['positions'][:, model.historical_steps - 1]
            lane_positions = reconstruct_lane_positions(
                lane_actor_index=data['lane_actor_index'],
                lane_actor_vectors=data['lane_actor_vectors'],
                current_positions=current_positions,
                num_lanes=data['lane_vectors'].size(0),
            )
            targets = build_reliability_targets(
                y_hat=y_hat.detach(),
                y=data.y,
                reg_mask=reg_mask,
                batch=graph_batch,
                lane_positions=lane_positions,
                lane_actor_index=data['lane_actor_index'],
                lane_actor_vectors=data['lane_actor_vectors'],
                positions=data['positions'],
                historical_steps=model.historical_steps,
                rotate_mat=data['rotate_mat'],
                agent_index=data['agent_index'],
                fde_threshold=model.risk_fde_threshold,
                miss_threshold=model.risk_miss_threshold,
                conflict_threshold=model.risk_conflict_threshold,
                conflict_min_frames=model.risk_conflict_min_frames,
                conflict_scope=model.risk_conflict_scope,
                offroad_threshold=model.risk_offroad_threshold,
                mode_target_policy=model.mode_target_policy,
                scene_target_policy=model.scene_target_policy,
                scene_rate_threshold=model.risk_scene_rate_threshold,
            )
            _accumulate(totals, summarize_reliability_targets(targets))
            if reliability_outputs is not None:
                pred_totals['mode_risk_pred_mean'] = (
                    pred_totals.get('mode_risk_pred_mean', 0.0) +
                    _scalar(reliability_outputs['mode_risk'].mean())
                )
                if reliability_outputs['scene_risk'].numel() > 0:
                    pred_totals['scene_risk_pred_mean'] = (
                        pred_totals.get('scene_risk_pred_mean', 0.0) +
                        _scalar(reliability_outputs['scene_risk'].mean())
                    )
            audited += 1

    totals.update(pred_totals)
    _print_table(totals, audited)


if __name__ == '__main__':
    main()
