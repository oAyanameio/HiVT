#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Dict
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


RUNTIME_ARG_NAMES = {
    'root',
    'ckpt_path',
    'rerank_alpha',
    'rerank_method',
    'rerank_top_k',
    'miss_threshold',
    'max_batches',
    'train_batch_size',
    'val_batch_size',
    'shuffle',
    'num_workers',
    'pin_memory',
    'persistent_workers',
    'gpus',
}
def rerank_scores(
    pi: torch.Tensor,
    risk: torch.Tensor,
    method: str,
    alpha: float,
    top_k: Optional[int],
    eps: float = 1e-6,
) -> torch.Tensor:
    log_prob = torch.log_softmax(pi, dim=-1)
    clipped_risk = risk.clamp(min=0.0, max=1.0 - eps)

    if method == 'prob_product':
        reranked = log_prob + alpha * torch.log1p(-clipped_risk)
    elif method == 'logprob_minus_norm_risk':
        risk_mean = risk.mean(dim=-1, keepdim=True)
        risk_std = risk.std(dim=-1, keepdim=True, unbiased=False)
        norm_risk = (risk - risk_mean) / (risk_std + eps)
        reranked = log_prob - alpha * norm_risk
    else:
        raise ValueError(f'Unknown reranking method: {method}')

    if top_k is None or top_k >= pi.size(-1):
        return reranked

    masked = torch.full_like(reranked, float('-inf'))
    topk_idx = torch.topk(pi, k=top_k, dim=-1).indices
    topk_scores = reranked.gather(dim=-1, index=topk_idx)
    masked.scatter_(dim=-1, index=topk_idx, src=topk_scores)
    return masked


def summarize_reranking_cases(
    original_fde: torch.Tensor,
    reranked_fde: torch.Tensor,
    miss_threshold: float,
    delta_eps: float = 1e-9,
) -> Dict[str, float]:
    original_fde = original_fde.reshape(-1).float()
    reranked_fde = reranked_fde.reshape(-1).float()
    if original_fde.shape != reranked_fde.shape:
        raise ValueError('original_fde and reranked_fde must have the same shape')
    if original_fde.numel() == 0:
        return {
            'count': 0,
            'original_mr': 0.0,
            'reranked_mr': 0.0,
            'mean_fde_delta': 0.0,
            'hit_to_miss_count': 0,
            'miss_to_hit_count': 0,
            'still_miss_improved_count': 0,
            'still_miss_worsened_count': 0,
            'still_miss_unchanged_count': 0,
            'still_hit_improved_count': 0,
            'still_hit_worsened_count': 0,
            'still_hit_unchanged_count': 0,
        }

    original_miss = original_fde > miss_threshold
    reranked_miss = reranked_fde > miss_threshold
    fde_delta = reranked_fde - original_fde

    hit_to_miss = (~original_miss) & reranked_miss
    miss_to_hit = original_miss & (~reranked_miss)
    still_miss = original_miss & reranked_miss
    still_hit = (~original_miss) & (~reranked_miss)
    improved = fde_delta < -delta_eps
    worsened = fde_delta > delta_eps
    unchanged = ~(improved | worsened)

    return {
        'count': int(original_fde.numel()),
        'original_mr': float(original_miss.float().mean()),
        'reranked_mr': float(reranked_miss.float().mean()),
        'mean_fde_delta': float(fde_delta.mean()),
        'hit_to_miss_count': int(hit_to_miss.sum().item()),
        'miss_to_hit_count': int(miss_to_hit.sum().item()),
        'still_miss_improved_count': int((still_miss & improved).sum().item()),
        'still_miss_worsened_count': int((still_miss & worsened).sum().item()),
        'still_miss_unchanged_count': int((still_miss & unchanged).sum().item()),
        'still_hit_improved_count': int((still_hit & improved).sum().item()),
        'still_hit_worsened_count': int((still_hit & worsened).sum().item()),
        'still_hit_unchanged_count': int((still_hit & unchanged).sum().item()),
    }


def main() -> None:
    from datamodules import ArgoverseV1DataModule
    from metrics import ADE
    from metrics import FDE
    from metrics import MR
    from models.hivt import HiVT

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--rerank_alpha', type=float, default=0.5)
    parser.add_argument('--rerank_method', type=str, default='prob_product')
    parser.add_argument('--rerank_top_k', type=int, default=None)
    parser.add_argument('--miss_threshold', type=float, default=2.0)
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

    orig_ade = ADE().to(device)
    orig_fde = FDE().to(device)
    orig_mr = MR().to(device)
    rerank_ade = ADE().to(device)
    rerank_fde = FDE().to(device)
    rerank_mr = MR().to(device)
    changed = 0.0
    count = 0.0
    original_fde_values = []
    reranked_fde_values = []

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            data = data.to(device)
            y_hat, pi, reliability_outputs = model(data)
            y_hat_agent = y_hat[:, data['agent_index'], :, :2]
            y_agent = data.y[data['agent_index']]
            graph_indices = torch.arange(data.num_graphs, device=device)

            orig_idx = pi[data['agent_index']].argmax(dim=-1)
            agent_pi = pi[data['agent_index']]
            agent_risk = reliability_outputs['mode_risk'][data['agent_index']]
            reranked_scores = rerank_scores(
                pi=agent_pi,
                risk=agent_risk,
                method=args.rerank_method,
                alpha=args.rerank_alpha,
                top_k=args.rerank_top_k,
            )
            rerank_idx = reranked_scores.argmax(dim=-1)

            orig_best = y_hat_agent[orig_idx, graph_indices]
            rerank_best = y_hat_agent[rerank_idx, graph_indices]
            batch_orig_fde = torch.norm(orig_best[:, -1] - y_agent[:, -1], p=2, dim=-1)
            batch_rerank_fde = torch.norm(rerank_best[:, -1] - y_agent[:, -1], p=2, dim=-1)

            orig_ade.update(orig_best, y_agent)
            orig_fde.update(orig_best, y_agent)
            orig_mr.update(orig_best, y_agent)
            rerank_ade.update(rerank_best, y_agent)
            rerank_fde.update(rerank_best, y_agent)
            rerank_mr.update(rerank_best, y_agent)

            changed += float((orig_idx != rerank_idx).sum().item())
            count += float(orig_idx.numel())
            original_fde_values.append(batch_orig_fde.cpu())
            reranked_fde_values.append(batch_rerank_fde.cpu())

    case_summary = summarize_reranking_cases(
        original_fde=torch.cat(original_fde_values, dim=0),
        reranked_fde=torch.cat(reranked_fde_values, dim=0),
        miss_threshold=args.miss_threshold,
    )

    print('metric,value')
    print('original_minADE,{:.6f}'.format(float(orig_ade.compute())))
    print('original_minFDE,{:.6f}'.format(float(orig_fde.compute())))
    print('original_minMR,{:.6f}'.format(float(orig_mr.compute())))
    print('reranked_minADE,{:.6f}'.format(float(rerank_ade.compute())))
    print('reranked_minFDE,{:.6f}'.format(float(rerank_fde.compute())))
    print('reranked_minMR,{:.6f}'.format(float(rerank_mr.compute())))
    print('rerank_top1_change_rate,{:.6f}'.format(changed / max(count, 1.0)))
    print('case_count,{}'.format(case_summary['count']))
    print('case_original_mr,{:.6f}'.format(case_summary['original_mr']))
    print('case_reranked_mr,{:.6f}'.format(case_summary['reranked_mr']))
    print('case_mean_fde_delta,{:.6f}'.format(case_summary['mean_fde_delta']))
    print('case_hit_to_miss_count,{}'.format(case_summary['hit_to_miss_count']))
    print('case_miss_to_hit_count,{}'.format(case_summary['miss_to_hit_count']))
    print('case_still_miss_improved_count,{}'.format(case_summary['still_miss_improved_count']))
    print('case_still_miss_worsened_count,{}'.format(case_summary['still_miss_worsened_count']))
    print('case_still_miss_unchanged_count,{}'.format(case_summary['still_miss_unchanged_count']))
    print('case_still_hit_improved_count,{}'.format(case_summary['still_hit_improved_count']))
    print('case_still_hit_worsened_count,{}'.format(case_summary['still_hit_worsened_count']))
    print('case_still_hit_unchanged_count,{}'.format(case_summary['still_hit_unchanged_count']))


if __name__ == '__main__':
    main()
