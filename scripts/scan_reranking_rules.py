#!/usr/bin/env python
from argparse import ArgumentParser
import csv
from pathlib import Path
import sys
from typing import Dict
from typing import Iterable
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

from scripts.eval_reranking import evaluate_reranking
from scripts.eval_reranking import RUNTIME_ARG_NAMES


def parse_float_grid(spec: str) -> List[float]:
    values: List[float] = []
    for part in spec.split(','):
        item = part.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError('Expected at least one float value in grid spec.')
    return values


def format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return 'none'
    return f'{value:.3f}'


def build_scan_row(
    margin: float,
    guard: Optional[float],
    metrics: Dict[str, float],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        'rerank_margin': margin,
        'rerank_guard': '' if guard is None else guard,
        'rerank_guard_label': format_optional_float(guard),
    }
    row.update(metrics)
    return row


def iter_scan_rows(
    margins: Iterable[float],
    guards: Iterable[Optional[float]],
    evaluator,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for margin in margins:
        for guard in guards:
            rows.append(build_scan_row(margin, guard, evaluator(margin, guard)))
    return rows


def write_scan_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError('Expected at least one row to write.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    from datamodules import ArgoverseV1DataModule
    from models.hivt import HiVT

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--rerank_alpha', type=float, default=1.0)
    parser.add_argument('--rerank_method', type=str, default='prob_product')
    parser.add_argument('--rerank_top_k', type=int, default=3)
    parser.add_argument('--margin_grid', type=str, default='0.0,0.05,0.1,0.15,0.2')
    parser.add_argument('--guard_grid', type=str, default='0.0,0.1,0.2,0.3')
    parser.add_argument('--include_no_guard', type=str2bool, default=False)
    parser.add_argument('--miss_threshold', type=float, default=2.0)
    parser.add_argument('--max_batches', type=int, default=32)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--shuffle', type=str2bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', type=str2bool, default=False)
    parser.add_argument('--persistent_workers', type=str2bool, default=False)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--output_csv', type=str, default=str(ROOT / 'docs' / 'reranking_rule_scan.csv'))
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
    model.reliability_module.rerank_alpha = args.rerank_alpha

    margins = parse_float_grid(args.margin_grid)
    guards: List[Optional[float]] = parse_float_grid(args.guard_grid)
    if args.include_no_guard:
        guards = [None] + guards

    def evaluator(margin: float, guard: Optional[float]) -> Dict[str, float]:
        return evaluate_reranking(
            model=model,
            loader=loader,
            device=device,
            rerank_method=args.rerank_method,
            rerank_alpha=args.rerank_alpha,
            rerank_top_k=args.rerank_top_k,
            rerank_margin=margin,
            rerank_guard=guard,
            miss_threshold=args.miss_threshold,
            max_batches=args.max_batches,
        )

    rows = iter_scan_rows(margins, guards, evaluator)
    output_path = Path(args.output_csv)
    write_scan_csv(output_path, rows)

    print('rerank_margin,rerank_guard,reranked_minMR,case_hit_to_miss_count,case_miss_to_hit_count,rerank_top1_change_rate')
    for row in rows:
        print(
            '{:.3f},{},{:.6f},{},{},{:.6f}'.format(
                float(row['rerank_margin']),
                row['rerank_guard_label'],
                float(row['reranked_minMR']),
                int(row['case_hit_to_miss_count']),
                int(row['case_miss_to_hit_count']),
                float(row['rerank_top1_change_rate']),
            )
        )
    print(f'output_csv,{output_path}')


if __name__ == '__main__':
    main()
