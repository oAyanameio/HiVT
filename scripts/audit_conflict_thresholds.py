#!/usr/bin/env python
"""Audit conflict target rates under different distance thresholds."""
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict
from typing import List
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
from utils import make_parser_arg_optional
from utils import merge_checkpoint_hparams
from utils import str2bool


RUNTIME_ARG_NAMES = {
    'root',
    'ckpt_path',
    'split',
    'conflict_thresholds',
    'max_batches',
    'train_batch_size',
    'val_batch_size',
    'shuffle',
    'num_workers',
    'pin_memory',
    'persistent_workers',
    'gpus',
}


def summarize_positive_rate(values: torch.Tensor) -> torch.Tensor:
    values = values.float().reshape(-1)
    if values.numel() == 0:
        return values.new_zeros(())
    return values.mean()


def parse_thresholds(raw: str) -> List[float]:
    thresholds = []
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        thresholds.append(float(part))
    if not thresholds:
        raise ValueError('At least one conflict threshold is required')
    return thresholds


def _accumulate_metric(
    totals: Dict[str, float],
    counts: Dict[str, int],
    key: str,
    value: torch.Tensor,
) -> None:
    totals[key] = totals.get(key, 0.0) + float(value.detach().cpu().item())
    counts[key] = counts.get(key, 0) + 1


def main() -> None:
    from datamodules import ArgoverseV1DataModule
    from models import compute_conflict_risk_targets
    from models import compute_mode_risk_targets
    from models.hivt import HiVT

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--conflict_thresholds', type=str, default='1.0,1.5,2.0,3.0')
    parser.add_argument('--max_batches', type=int, default=32)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--shuffle', type=str2bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', type=str2bool, default=False)
    parser.add_argument('--persistent_workers', type=str2bool, default=False)
    parser.add_argument('--gpus', type=int, default=0)
    parser = HiVT.add_model_specific_args(parser)
    make_parser_arg_optional(parser, 'embed_dim', default=None)
    args = parser.parse_args()

    thresholds = parse_thresholds(args.conflict_thresholds)
    pl.seed_everything(2022)
    device = torch.device('cuda:0' if args.gpus > 0 and torch.cuda.is_available() else 'cpu')

    datamodule = ArgoverseV1DataModule.from_argparse_args(args)
    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.train_dataloader() if args.split == 'train' else datamodule.val_dataloader()

    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    model_kwargs = merge_checkpoint_hparams(
        dict(vars(args)),
        checkpoint.get('hyper_parameters', {}),
        runtime_arg_names=RUNTIME_ARG_NAMES,
    )
    model = HiVT.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        map_location=device,
        strict=False,
        **model_kwargs,
    ).to(device)
    model.eval()

    threshold_totals: Dict[float, Dict[str, float]] = {thr: {} for thr in thresholds}
    threshold_counts: Dict[float, Dict[str, int]] = {thr: {} for thr in thresholds}

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            data = data.to(device)
            y_hat, _, _ = model(data)
            reg_mask = ~data['padding_mask'][:, model.historical_steps:]
            batch = getattr(data, 'batch', None)
            if batch is None:
                batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
            mode_targets, valid_mask, _ = compute_mode_risk_targets(
                y_hat=y_hat.detach(),
                y=data.y,
                reg_mask=reg_mask,
                fde_threshold=model.risk_fde_threshold,
            )
            _accumulate_metric(
                threshold_totals[thresholds[0]],
                threshold_counts[thresholds[0]],
                'mode_risk_target_rate',
                summarize_positive_rate(mode_targets[valid_mask]),
            )

            for threshold in thresholds:
                conflict_targets, min_pair_dist = compute_conflict_risk_targets(
                    y_hat=y_hat.detach(),
                    reg_mask=reg_mask,
                    batch=batch,
                    positions=data['positions'],
                    historical_steps=model.historical_steps,
                    rotate_mat=data['rotate_mat'],
                    agent_index=data['agent_index'],
                    conflict_threshold=threshold,
                    min_frames=model.risk_conflict_min_frames,
                    scope=model.risk_conflict_scope,
                )
                target_nodes = data['agent_index'].view(-1) if data['agent_index'] is not None else None
                if target_nodes is not None and target_nodes.numel() > 0:
                    target_values = conflict_targets[target_nodes]
                else:
                    target_values = conflict_targets.new_zeros((0,))
                finite_pair = min_pair_dist[torch.isfinite(min_pair_dist)]

                _accumulate_metric(
                    threshold_totals[threshold],
                    threshold_counts[threshold],
                    'conflict_risk_target_rate',
                    summarize_positive_rate(conflict_targets[valid_mask]),
                )
                _accumulate_metric(
                    threshold_totals[threshold],
                    threshold_counts[threshold],
                    'target_actor_conflict_rate',
                    summarize_positive_rate(target_values),
                )
                if finite_pair.numel() > 0:
                    _accumulate_metric(
                        threshold_totals[threshold],
                        threshold_counts[threshold],
                        'min_pair_dist_q10',
                        torch.quantile(finite_pair, 0.1),
                    )
                    _accumulate_metric(
                        threshold_totals[threshold],
                        threshold_counts[threshold],
                        'min_pair_dist_q50',
                        torch.quantile(finite_pair, 0.5),
                    )
                    _accumulate_metric(
                        threshold_totals[threshold],
                        threshold_counts[threshold],
                        'min_pair_dist_q90',
                        torch.quantile(finite_pair, 0.9),
                    )

    print('conflict_threshold,metric,value')
    for threshold in thresholds:
        totals = threshold_totals[threshold]
        counts = threshold_counts[threshold]
        if threshold != thresholds[0]:
            totals['mode_risk_target_rate'] = threshold_totals[thresholds[0]]['mode_risk_target_rate']
            counts['mode_risk_target_rate'] = threshold_counts[thresholds[0]]['mode_risk_target_rate']
        for key in sorted(totals.keys()):
            denom = max(counts.get(key, 0), 1)
            print('{:.2f},{},{}'.format(threshold, key, totals[key] / denom))


if __name__ == '__main__':
    main()
