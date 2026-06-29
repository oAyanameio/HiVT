#!/usr/bin/env python
"""Analyze scene-level calibration for reliability head outputs."""
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
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

from metrics import BrierScore
from metrics import ECE


def bucketize_binary_calibration(
    probs: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 10,
) -> List[Dict[str, float]]:
    probs = probs.reshape(-1).float()
    targets = targets.reshape(-1).float()
    rows = []
    bin_edges = torch.linspace(0.0, 1.0, steps=num_bins + 1)
    for idx in range(num_bins):
        left = float(bin_edges[idx].item())
        right = float(bin_edges[idx + 1].item())
        if idx == num_bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        count = int(mask.sum().item())
        if count == 0:
            rows.append({
                'bin_left': left,
                'bin_right': right,
                'count': 0,
                'avg_prob': 0.0,
                'avg_target': 0.0,
            })
            continue
        rows.append({
            'bin_left': left,
            'bin_right': right,
            'count': count,
            'avg_prob': float(probs[mask].mean().item()),
            'avg_target': float(targets[mask].mean().item()),
        })
    return rows


def summarize_binary_classification(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    probs = probs.reshape(-1).float()
    targets = targets.reshape(-1).float()
    preds = probs >= threshold
    positives = targets >= 0.5

    tp = float((preds & positives).sum().item())
    fp = float((preds & (~positives)).sum().item())
    tn = float(((~preds) & (~positives)).sum().item())
    fn = float(((~preds) & positives).sum().item())

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)

    return {
        'threshold': threshold,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'accuracy': accuracy,
    }


def main() -> None:
    from datamodules import ArgoverseV1DataModule
    from models import build_reliability_targets
    from models import reconstruct_lane_positions
    from models.hivt import HiVT

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--num_bins', type=int, default=10)
    parser.add_argument('--decision_threshold', type=float, default=0.5)
    parser.add_argument('--max_batches', type=int, default=32)
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

    model = HiVT.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        map_location=device,
        strict=False,
        **vars(args),
    ).to(device)
    model.eval()

    scene_brier = BrierScore().to(device)
    scene_ece = ECE(num_bins=args.num_bins).to(device)
    scene_prob_list: List[torch.Tensor] = []
    scene_target_list: List[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            data = data.to(device)
            y_hat, _, reliability_outputs = model(data)
            if reliability_outputs is None or reliability_outputs['scene_risk'].numel() == 0:
                continue

            reg_mask = ~data['padding_mask'][:, model.historical_steps:]
            batch = getattr(data, 'batch', None)
            if batch is None:
                batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
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
                batch=batch,
                lane_positions=lane_positions,
                lane_actor_index=data['lane_actor_index'],
                lane_actor_vectors=data['lane_actor_vectors'],
                positions=data['positions'],
                historical_steps=model.historical_steps,
                rotate_mat=data['rotate_mat'],
                agent_index=data['agent_index'],
                fde_threshold=model.risk_fde_threshold,
                conflict_threshold=model.risk_conflict_threshold,
                offroad_threshold=model.risk_offroad_threshold,
                miss_threshold=model.risk_miss_threshold,
                mode_target_policy=model.mode_target_policy,
                scene_target_policy=model.scene_target_policy,
                conflict_scope=model.risk_conflict_scope,
                conflict_min_frames=model.risk_conflict_min_frames,
                scene_rate_threshold=model.risk_scene_rate_threshold,
            )
            scene_targets = targets['scene_targets']
            if scene_targets.numel() == 0:
                continue
            scene_probs = reliability_outputs['scene_risk']

            scene_brier.update(scene_probs, scene_targets)
            scene_ece.update(scene_probs, scene_targets)
            scene_prob_list.append(scene_probs.detach().cpu())
            scene_target_list.append(scene_targets.detach().cpu())

    if not scene_prob_list:
        print('No scene predictions were collected.')
        return

    scene_probs_all = torch.cat(scene_prob_list, dim=0)
    scene_targets_all = torch.cat(scene_target_list, dim=0)
    calibration_rows = bucketize_binary_calibration(
        probs=scene_probs_all,
        targets=scene_targets_all,
        num_bins=args.num_bins,
    )
    decision_summary = summarize_binary_classification(
        probs=scene_probs_all,
        targets=scene_targets_all,
        threshold=args.decision_threshold,
    )

    print('metric,value')
    print('scene_count,{}'.format(int(scene_probs_all.numel())))
    print('scene_BrierScore,{:.6f}'.format(float(scene_brier.compute())))
    print('scene_ECE,{:.6f}'.format(float(scene_ece.compute())))
    print('scene_pred_mean,{:.6f}'.format(float(scene_probs_all.mean().item())))
    print('scene_target_rate,{:.6f}'.format(float(scene_targets_all.float().mean().item())))
    print('scene_decision_threshold,{:.6f}'.format(float(decision_summary['threshold'])))
    print('scene_precision_at_threshold,{:.6f}'.format(float(decision_summary['precision'])))
    print('scene_recall_at_threshold,{:.6f}'.format(float(decision_summary['recall'])))
    print('scene_specificity_at_threshold,{:.6f}'.format(float(decision_summary['specificity'])))
    print('scene_accuracy_at_threshold,{:.6f}'.format(float(decision_summary['accuracy'])))
    print('bin_left,bin_right,count,avg_prob,avg_target')
    for row in calibration_rows:
        print(
            '{:.6f},{:.6f},{},{:.6f},{:.6f}'.format(
                row['bin_left'],
                row['bin_right'],
                int(row['count']),
                row['avg_prob'],
                row['avg_target'],
            )
        )


if __name__ == '__main__':
    main()
