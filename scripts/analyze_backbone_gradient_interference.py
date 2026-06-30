#!/usr/bin/env python
"""Analyze gradient interference between prediction and reliability losses."""
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
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
import torch.nn.functional as F
from utils import make_parser_arg_optional
from utils import merge_checkpoint_hparams
from utils import str2bool


RUNTIME_ARG_NAMES = {
    'root',
    'ckpt_path',
    'split',
    'max_batches',
    'train_batch_size',
    'val_batch_size',
    'shuffle',
    'num_workers',
    'pin_memory',
    'persistent_workers',
    'gpus',
}


def flatten_gradients(grads: Sequence[Optional[torch.Tensor]]) -> torch.Tensor:
    flat_parts: List[torch.Tensor] = []
    for grad in grads:
        if grad is None:
            continue
        flat_parts.append(grad.reshape(-1).detach().float().cpu())
    if not flat_parts:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(flat_parts, dim=0)


def compare_gradient_vectors(reference: torch.Tensor, other: torch.Tensor) -> Dict[str, float]:
    reference = reference.reshape(-1).float()
    other = other.reshape(-1).float()
    ref_norm = float(reference.norm().item())
    other_norm = float(other.norm().item())
    if ref_norm == 0.0 or other_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(F.cosine_similarity(reference.unsqueeze(0), other.unsqueeze(0), dim=-1).item())
    return {
        'ref_norm': ref_norm,
        'other_norm': other_norm,
        'cosine': cosine,
        'norm_ratio': other_norm / max(ref_norm, 1e-8),
    }


def summarize_metric_rows(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {
            'count': 0.0,
            'mean_cosine': 0.0,
            'negative_fraction': 0.0,
            'mean_norm_ratio': 0.0,
            'mean_ref_norm': 0.0,
            'mean_other_norm': 0.0,
        }
    count = float(len(rows))
    return {
        'count': count,
        'mean_cosine': sum(row['cosine'] for row in rows) / count,
        'negative_fraction': sum(1.0 for row in rows if row['cosine'] < 0.0) / count,
        'mean_norm_ratio': sum(row['norm_ratio'] for row in rows) / count,
        'mean_ref_norm': sum(row['ref_norm'] for row in rows) / count,
        'mean_other_norm': sum(row['other_norm'] for row in rows) / count,
    }


def combine_weighted_losses(
    loss_terms: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    total: Optional[torch.Tensor] = None
    for key, weight in weights.items():
        loss = loss_terms[key] * weight
        total = loss if total is None else total + loss
    if total is None:
        return torch.zeros((), dtype=torch.float32)
    return total


def _named_backbone_modules(model) -> Dict[str, torch.nn.Module]:
    return {
        'local_encoder': model.local_encoder,
        'global_interactor': model.global_interactor,
        'decoder': model.decoder,
    }


def _module_parameters(module: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [param for param in module.parameters() if param.requires_grad]


def _compute_loss_terms(model, data) -> Dict[str, torch.Tensor]:
    from models import build_reliability_targets
    from models import compute_threshold_weights
    from models import reconstruct_lane_positions

    y_hat, pi, reliability_outputs = model(data)
    reg_mask = ~data['padding_mask'][:, model.historical_steps:]
    valid_steps = reg_mask.sum(dim=-1)
    cls_mask = valid_steps > 0

    l2_norm = (torch.norm(y_hat[:, :, :, :2] - data.y, p=2, dim=-1) * reg_mask).sum(dim=-1)
    best_mode = l2_norm.argmin(dim=0)
    y_hat_best = y_hat[best_mode, torch.arange(data.num_nodes, device=y_hat.device)]
    reg_loss = model.reg_loss(y_hat_best[reg_mask], data.y[reg_mask])
    soft_target = F.softmax(-l2_norm[:, cls_mask] / valid_steps[cls_mask], dim=0).t().detach()
    cls_loss = model.cls_loss(pi[cls_mask], soft_target)
    pred_loss = reg_loss + cls_loss

    mode_risk_loss = pred_loss.new_zeros(())
    scene_loss = pred_loss.new_zeros(())
    rank_loss = pred_loss.new_zeros(())
    calib_loss = pred_loss.new_zeros(())

    if model.reliability_module is not None and reliability_outputs is not None:
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=y_hat.device)
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
        mode_targets = targets['mode_targets']
        scene_targets = targets['scene_targets']
        if valid_mask.any():
            raw_mode_risk_loss = F.binary_cross_entropy_with_logits(
                reliability_outputs['mode_risk_logits'][valid_mask],
                mode_targets[valid_mask],
                reduction='none',
            )
            if model.mode_risk_threshold_weight_enabled:
                threshold_weights = compute_threshold_weights(
                    fde=targets['fde'],
                    threshold=model.risk_fde_threshold,
                    radius=model.mode_risk_threshold_weight_radius,
                    base_weight=model.mode_risk_threshold_weight_base,
                    peak_weight=model.mode_risk_threshold_weight_peak,
                    valid_mask=valid_mask,
                )[valid_mask]
                mode_risk_loss = (raw_mode_risk_loss * threshold_weights).sum() / threshold_weights.sum().clamp(min=1e-6)
            else:
                mode_risk_loss = raw_mode_risk_loss.mean()
            if model.rank_loss_weight > 0:
                rank_loss = model.rank_loss_fn(
                    mode_risk=reliability_outputs['mode_risk'],
                    mode_error=targets['fde'],
                    valid_mask=valid_mask,
                    mode_logits=pi.detach(),
                    top_k=model.mode_risk_rank_top_k if model.mode_risk_rank_top_k > 0 else None,
                    focus_threshold=model.risk_fde_threshold if model.mode_risk_rank_near_threshold_only else None,
                    focus_radius=model.mode_risk_rank_threshold_radius if model.mode_risk_rank_near_threshold_only else None,
                )
            if model.calib_loss_weight > 0:
                calib_loss = model.calib_loss_fn(
                    risk=reliability_outputs['mode_risk'][valid_mask],
                    target=mode_targets[valid_mask],
                )
        if scene_targets.numel() > 0:
            scene_loss = model.risk_loss(
                reliability_outputs['scene_risk_logits'],
                scene_targets,
            )

    loss_terms = {
        'pred_loss': pred_loss,
        'reg_loss': reg_loss,
        'cls_loss': cls_loss,
        'mode_risk_loss': mode_risk_loss,
        'scene_loss': scene_loss,
        'rank_loss': rank_loss,
        'calib_loss': calib_loss,
    }
    loss_terms['mode_component_loss'] = combine_weighted_losses(
        loss_terms,
        {
            'mode_risk_loss': float(model.reliability_loss_weight),
            'rank_loss': float(model.rank_loss_weight),
            'calib_loss': float(model.calib_loss_weight),
        },
    )
    loss_terms['scene_component_loss'] = combine_weighted_losses(
        loss_terms,
        {
            'scene_loss': float(model.scene_loss_weight),
        },
    )
    loss_terms['reliability_total_loss'] = loss_terms['mode_component_loss'] + loss_terms['scene_component_loss']
    return loss_terms


def _collect_gradients(loss: torch.Tensor, modules: Dict[str, torch.nn.Module]) -> Dict[str, torch.Tensor]:
    params: List[torch.nn.Parameter] = []
    for module in modules.values():
        params.extend(_module_parameters(module))
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    result: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, module in modules.items():
        module_params = _module_parameters(module)
        module_grads = grads[offset: offset + len(module_params)]
        offset += len(module_params)
        result[name] = flatten_gradients(module_grads)
    return result


def _analyze_batch(loss_terms: Dict[str, torch.Tensor], modules: Dict[str, torch.nn.Module]) -> Dict[str, Dict[str, Dict[str, float]]]:
    pred_grads = _collect_gradients(loss_terms['pred_loss'], modules)
    mode_grads = _collect_gradients(loss_terms['mode_component_loss'], modules)
    scene_grads = _collect_gradients(loss_terms['scene_component_loss'], modules)
    reliability_grads = _collect_gradients(loss_terms['reliability_total_loss'], modules)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for module_name in modules:
        results[module_name] = {
            'mode_vs_pred': compare_gradient_vectors(pred_grads[module_name], mode_grads[module_name]),
            'scene_vs_pred': compare_gradient_vectors(pred_grads[module_name], scene_grads[module_name]),
            'reliability_vs_pred': compare_gradient_vectors(pred_grads[module_name], reliability_grads[module_name]),
        }
    return results


def _accumulate_metrics(
    accumulator: Dict[str, Dict[str, List[Dict[str, float]]]],
    batch_metrics: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    for module_name, module_metrics in batch_metrics.items():
        module_acc = accumulator.setdefault(module_name, {})
        for relation, relation_metrics in module_metrics.items():
            module_acc.setdefault(relation, []).append(relation_metrics)


def _print_summary(
    metrics: Dict[str, Dict[str, List[Dict[str, float]]]],
    loss_means: Dict[str, float],
    batches: int,
) -> None:
    print('metric,value')
    print('analyzed_batches,{}'.format(batches))
    for key, value in sorted(loss_means.items()):
        print('loss_mean_{},{}'.format(key, value / max(batches, 1)))
    for module_name, module_metrics in metrics.items():
        for relation, rows in module_metrics.items():
            summary = summarize_metric_rows(rows)
            prefix = '{}_{}'.format(module_name, relation)
            for key, value in summary.items():
                print('{},{}'.format(prefix + '_' + key, value))


def main() -> None:
    from datamodules import ArgoverseV1DataModule
    from models.hivt import HiVT

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--max_batches', type=int, default=8)
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
    model.freeze_backbone = False
    for module in _named_backbone_modules(model).values():
        for param in module.parameters():
            param.requires_grad = True
    model.eval()

    accumulated_metrics: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    loss_means: Dict[str, float] = {}
    analyzed_batches = 0

    for batch_idx, data in enumerate(loader):
        if batch_idx >= args.max_batches:
            break
        data = data.to(device)
        model.zero_grad(set_to_none=True)
        loss_terms = _compute_loss_terms(model, data)
        batch_metrics = _analyze_batch(loss_terms, _named_backbone_modules(model))
        _accumulate_metrics(accumulated_metrics, batch_metrics)
        for loss_name, loss_value in loss_terms.items():
            loss_means[loss_name] = loss_means.get(loss_name, 0.0) + float(loss_value.detach().cpu().item())
        analyzed_batches += 1

    _print_summary(accumulated_metrics, loss_means, analyzed_batches)


if __name__ == '__main__':
    main()
