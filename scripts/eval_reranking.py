#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
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
from metrics import ADE
from metrics import FDE
from metrics import MR
from models.hivt import HiVT


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--rerank_alpha', type=float, default=0.5)
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
    loader = datamodule.val_dataloader()

    model = HiVT.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        map_location=device,
        strict=False,
        **vars(args),
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
            rerank_idx = reliability_outputs['reranked_pi'][data['agent_index']].argmax(dim=-1)

            orig_best = y_hat_agent[orig_idx, graph_indices]
            rerank_best = y_hat_agent[rerank_idx, graph_indices]

            orig_ade.update(orig_best, y_agent)
            orig_fde.update(orig_best, y_agent)
            orig_mr.update(orig_best, y_agent)
            rerank_ade.update(rerank_best, y_agent)
            rerank_fde.update(rerank_best, y_agent)
            rerank_mr.update(rerank_best, y_agent)

            changed += float((orig_idx != rerank_idx).sum().item())
            count += float(orig_idx.numel())

    print('metric,value')
    print('original_minADE,{:.6f}'.format(float(orig_ade.compute())))
    print('original_minFDE,{:.6f}'.format(float(orig_fde.compute())))
    print('original_minMR,{:.6f}'.format(float(orig_mr.compute())))
    print('reranked_minADE,{:.6f}'.format(float(rerank_ade.compute())))
    print('reranked_minFDE,{:.6f}'.format(float(rerank_fde.compute())))
    print('reranked_minMR,{:.6f}'.format(float(rerank_mr.compute())))
    print('rerank_top1_change_rate,{:.6f}'.format(changed / max(count, 1.0)))


if __name__ == '__main__':
    main()
