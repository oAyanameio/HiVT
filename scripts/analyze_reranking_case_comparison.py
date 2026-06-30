#!/usr/bin/env python
from argparse import ArgumentParser
import csv
from pathlib import Path
import sys
from typing import Dict
from typing import List
from typing import Optional
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

from scripts.eval_reranking import rerank_scores
from scripts.eval_reranking import RUNTIME_ARG_NAMES
from scripts.eval_reranking import select_rerank_indices


def compute_quantiles(values: torch.Tensor, prefix: str) -> Dict[str, float]:
    values = values.reshape(-1).float().cpu()
    if values.numel() == 0:
        return {
            f'{prefix}_count': 0.0,
            f'{prefix}_q10': 0.0,
            f'{prefix}_q50': 0.0,
            f'{prefix}_q90': 0.0,
        }
    quantiles = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], dtype=values.dtype))
    return {
        f'{prefix}_count': float(values.numel()),
        f'{prefix}_q10': float(quantiles[0].item()),
        f'{prefix}_q50': float(quantiles[1].item()),
        f'{prefix}_q90': float(quantiles[2].item()),
    }


def summarize_case_comparison(
    original_fde: torch.Tensor,
    base_fde: torch.Tensor,
    compare_fde: torch.Tensor,
    risk_gap: torch.Tensor,
    original_risk: torch.Tensor,
    miss_threshold: float,
) -> Dict[str, float]:
    original_fde = original_fde.reshape(-1).float().cpu()
    base_fde = base_fde.reshape(-1).float().cpu()
    compare_fde = compare_fde.reshape(-1).float().cpu()
    risk_gap = risk_gap.reshape(-1).float().cpu()
    original_risk = original_risk.reshape(-1).float().cpu()

    original_miss = original_fde > miss_threshold
    base_miss = base_fde > miss_threshold
    compare_miss = compare_fde > miss_threshold

    base_switch = base_fde != original_fde
    compare_switch = compare_fde != original_fde
    blocked_switch = base_switch & (~compare_switch)

    base_hit_to_miss = (~original_miss) & base_miss
    base_miss_to_hit = original_miss & (~base_miss)
    compare_hit_to_miss = (~original_miss) & compare_miss
    compare_miss_to_hit = original_miss & (~compare_miss)

    blocked_harmful = blocked_switch & base_hit_to_miss
    blocked_helpful = blocked_switch & base_miss_to_hit
    blocked_neutral = blocked_switch & (~base_hit_to_miss) & (~base_miss_to_hit)

    summary: Dict[str, float] = {
        'case_count': float(original_fde.numel()),
        'original_mr': float(original_miss.float().mean().item()),
        'base_mr': float(base_miss.float().mean().item()),
        'compare_mr': float(compare_miss.float().mean().item()),
        'base_switch_count': float(base_switch.sum().item()),
        'compare_switch_count': float(compare_switch.sum().item()),
        'blocked_switch_count': float(blocked_switch.sum().item()),
        'base_hit_to_miss_count': float(base_hit_to_miss.sum().item()),
        'base_miss_to_hit_count': float(base_miss_to_hit.sum().item()),
        'compare_hit_to_miss_count': float(compare_hit_to_miss.sum().item()),
        'compare_miss_to_hit_count': float(compare_miss_to_hit.sum().item()),
        'blocked_harmful_count': float(blocked_harmful.sum().item()),
        'blocked_helpful_count': float(blocked_helpful.sum().item()),
        'blocked_neutral_count': float(blocked_neutral.sum().item()),
    }
    summary.update(compute_quantiles(risk_gap[base_switch], 'base_switch_gap'))
    summary.update(compute_quantiles(risk_gap[blocked_switch], 'blocked_switch_gap'))
    summary.update(compute_quantiles(risk_gap[blocked_harmful], 'blocked_harmful_gap'))
    summary.update(compute_quantiles(risk_gap[blocked_helpful], 'blocked_helpful_gap'))
    summary.update(compute_quantiles(risk_gap[compare_switch], 'kept_switch_gap'))
    summary.update(compute_quantiles(original_risk[base_switch], 'base_switch_orig_risk'))
    summary.update(compute_quantiles(original_fde[base_hit_to_miss], 'base_hit_to_miss_orig_fde'))
    summary.update(compute_quantiles((original_fde[base_hit_to_miss] - miss_threshold).abs(), 'base_hit_to_miss_abs_threshold_dist'))
    summary.update(compute_quantiles(original_fde[base_miss_to_hit], 'base_miss_to_hit_orig_fde'))
    summary.update(compute_quantiles((original_fde[base_miss_to_hit] - miss_threshold).abs(), 'base_miss_to_hit_abs_threshold_dist'))
    return summary


def write_case_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError('Expected at least one case row.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_case_row(
    case_index: int,
    batch_index: int,
    graph_index: int,
    original_index: int,
    base_index: int,
    compare_index: int,
    original_risk: float,
    base_risk: float,
    risk_gap: float,
    original_fde: float,
    base_fde: float,
    compare_fde: float,
    miss_threshold: float,
) -> Dict[str, object]:
    original_miss = original_fde > miss_threshold
    base_miss = base_fde > miss_threshold
    compare_miss = compare_fde > miss_threshold
    return {
        'case_index': case_index,
        'batch_index': batch_index,
        'graph_index': graph_index,
        'original_mode_index': original_index,
        'base_mode_index': base_index,
        'compare_mode_index': compare_index,
        'original_risk': original_risk,
        'base_risk': base_risk,
        'risk_gap': risk_gap,
        'original_fde': original_fde,
        'base_fde': base_fde,
        'compare_fde': compare_fde,
        'original_miss': int(original_miss),
        'base_miss': int(base_miss),
        'compare_miss': int(compare_miss),
        'base_switch': int(base_index != original_index),
        'compare_switch': int(compare_index != original_index),
        'blocked_by_compare_margin': int((base_index != original_index) and (compare_index == original_index)),
        'base_hit_to_miss': int((not original_miss) and base_miss),
        'base_miss_to_hit': int(original_miss and (not base_miss)),
        'compare_hit_to_miss': int((not original_miss) and compare_miss),
        'compare_miss_to_hit': int(original_miss and (not compare_miss)),
    }


def main() -> None:
    from datamodules import ArgoverseV1DataModule
    from models.hivt import HiVT

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--rerank_alpha', type=float, default=1.0)
    parser.add_argument('--rerank_method', type=str, default='prob_product')
    parser.add_argument('--rerank_top_k', type=int, default=3)
    parser.add_argument('--base_margin', type=float, default=0.0)
    parser.add_argument('--compare_margin', type=float, default=0.1)
    parser.add_argument('--rerank_guard', type=float, default=None)
    parser.add_argument('--miss_threshold', type=float, default=2.0)
    parser.add_argument('--max_batches', type=int, default=32)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--shuffle', type=str2bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', type=str2bool, default=False)
    parser.add_argument('--persistent_workers', type=str2bool, default=False)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument(
        '--output_csv',
        type=str,
        default=str(ROOT / 'docs' / 'reranking_margin_case_details.csv'),
    )
    parser = HiVT.add_model_specific_args(parser)
    make_parser_arg_optional(parser, 'embed_dim', default=None)
    args = parser.parse_args()

    pl.seed_everything(2022)
    device = torch.device('cuda:0' if args.gpus > 0 and torch.cuda.is_available() else 'cpu')
    datamodule = ArgoverseV1DataModule.from_argparse_args(args)
    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.val_dataloader()

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
    if model.reliability_module is None:
        raise RuntimeError('Checkpoint does not contain reliability module configuration.')

    case_rows: List[Dict[str, object]] = []
    original_fde_values = []
    base_fde_values = []
    compare_fde_values = []
    risk_gap_values = []
    original_risk_values = []
    next_case_index = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            data = data.to(device)
            y_hat, pi, reliability_outputs = model(data)
            y_hat_agent = y_hat[:, data['agent_index'], :, :2]
            y_agent = data.y[data['agent_index']]
            graph_indices = torch.arange(data.num_graphs, device=device)

            agent_pi = pi[data['agent_index']]
            agent_risk = reliability_outputs['mode_risk'][data['agent_index']]
            original_idx = agent_pi.argmax(dim=-1)
            base_scores = rerank_scores(
                pi=agent_pi,
                risk=agent_risk,
                method=args.rerank_method,
                alpha=args.rerank_alpha,
                top_k=args.rerank_top_k,
            )
            base_idx = base_scores.argmax(dim=-1)
            compare_idx = select_rerank_indices(
                pi=agent_pi,
                risk=agent_risk,
                method=args.rerank_method,
                alpha=args.rerank_alpha,
                top_k=args.rerank_top_k,
                margin=args.compare_margin,
                guard=args.rerank_guard,
            )

            original_best = y_hat_agent[original_idx, graph_indices]
            base_best = y_hat_agent[base_idx, graph_indices]
            compare_best = y_hat_agent[compare_idx, graph_indices]
            original_fde = torch.norm(original_best[:, -1] - y_agent[:, -1], p=2, dim=-1)
            base_fde = torch.norm(base_best[:, -1] - y_agent[:, -1], p=2, dim=-1)
            compare_fde = torch.norm(compare_best[:, -1] - y_agent[:, -1], p=2, dim=-1)

            batch_index = torch.arange(agent_pi.size(0), device=device)
            original_risk = agent_risk[batch_index, original_idx]
            base_risk = agent_risk[batch_index, base_idx]
            risk_gap = original_risk - base_risk

            original_fde_values.append(original_fde.cpu())
            base_fde_values.append(base_fde.cpu())
            compare_fde_values.append(compare_fde.cpu())
            risk_gap_values.append(risk_gap.cpu())
            original_risk_values.append(original_risk.cpu())

            for graph_offset in range(agent_pi.size(0)):
                case_rows.append(
                    build_case_row(
                        case_index=next_case_index,
                        batch_index=batch_idx,
                        graph_index=graph_offset,
                        original_index=int(original_idx[graph_offset].item()),
                        base_index=int(base_idx[graph_offset].item()),
                        compare_index=int(compare_idx[graph_offset].item()),
                        original_risk=float(original_risk[graph_offset].item()),
                        base_risk=float(base_risk[graph_offset].item()),
                        risk_gap=float(risk_gap[graph_offset].item()),
                        original_fde=float(original_fde[graph_offset].item()),
                        base_fde=float(base_fde[graph_offset].item()),
                        compare_fde=float(compare_fde[graph_offset].item()),
                        miss_threshold=args.miss_threshold,
                    )
                )
                next_case_index += 1

    output_path = Path(args.output_csv)
    write_case_rows_csv(output_path, case_rows)

    summary = summarize_case_comparison(
        original_fde=torch.cat(original_fde_values, dim=0),
        base_fde=torch.cat(base_fde_values, dim=0),
        compare_fde=torch.cat(compare_fde_values, dim=0),
        risk_gap=torch.cat(risk_gap_values, dim=0),
        original_risk=torch.cat(original_risk_values, dim=0),
        miss_threshold=args.miss_threshold,
    )

    print('metric,value')
    print(f'base_margin,{args.base_margin:.6f}')
    print(f'compare_margin,{args.compare_margin:.6f}')
    print(f'rerank_guard,{0.0 if args.rerank_guard is None else args.rerank_guard:.6f}')
    for key, value in summary.items():
        if float(value).is_integer():
            print(f'{key},{int(value)}')
        else:
            print(f'{key},{value:.6f}')
    print(f'output_csv,{output_path}')


if __name__ == '__main__':
    main()
