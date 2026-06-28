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

from metrics import AUPRC
from metrics import AUROC
from metrics import BrierScore
from metrics import ECE


def naive_risk_from_pi(pi: torch.Tensor) -> torch.Tensor:
    return -torch.log_softmax(pi, dim=-1)


def spearman_rank_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()
    x_rank = x.argsort().argsort().float()
    y_rank = y.argsort().argsort().float()
    x_rank = x_rank - x_rank.mean()
    y_rank = y_rank - y_rank.mean()
    denom = x_rank.norm() * y_rank.norm()
    if float(denom) == 0.0:
        return x_rank.new_zeros(())
    return (x_rank * y_rank).sum() / denom


def main() -> None:
    from datamodules import ArgoverseV1DataModule
    from models import build_reliability_targets
    from models import reconstruct_lane_positions
    from models.hivt import HiVT

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--max_batches', type=int, default=16)
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

    naive_auroc = AUROC(compute_on_step=False).to(device)
    naive_auprc = AUPRC(compute_on_step=False).to(device)
    scene_brier = BrierScore().to(device)
    scene_ece = ECE().to(device)
    risk_list = []
    fde_list = []
    has_scene = False

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            data = data.to(device)
            y_hat, pi, reliability_outputs = model(data)
            if reliability_outputs is None:
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
            valid_mask = targets['valid_mask']
            if valid_mask.any():
                mode_targets = targets['mode_targets']
                naive_risk = naive_risk_from_pi(pi)
                naive_auroc.update(naive_risk[valid_mask], mode_targets[valid_mask])
                naive_auprc.update(naive_risk[valid_mask], mode_targets[valid_mask])
                risk_list.append(reliability_outputs['mode_risk'][valid_mask].reshape(-1).detach().cpu())
                fde_list.append(targets['fde'][valid_mask].reshape(-1).detach().cpu())
            if targets['scene_targets'].numel() > 0:
                has_scene = True
                scene_brier.update(reliability_outputs['scene_risk'], targets['scene_targets'])
                scene_ece.update(reliability_outputs['scene_risk'], targets['scene_targets'])

    risk_all = torch.cat(risk_list) if risk_list else torch.tensor([])
    fde_all = torch.cat(fde_list) if fde_list else torch.tensor([])
    spearman = spearman_rank_corr(risk_all, fde_all) if risk_all.numel() > 0 else torch.tensor(0.0)

    print('metric,value')
    if risk_list:
        print('naive_mode_AUROC,{:.6f}'.format(float(naive_auroc.compute())))
        print('naive_mode_AUPRC,{:.6f}'.format(float(naive_auprc.compute())))
        print('mode_risk_fde_spearman,{:.6f}'.format(float(spearman)))
    if has_scene:
        print('scene_BrierScore,{:.6f}'.format(float(scene_brier.compute())))
        print('scene_ECE,{:.6f}'.format(float(scene_ece.compute())))


if __name__ == '__main__':
    main()
