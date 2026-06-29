#!/usr/bin/env python
"""Analyze whether joint reliability training degrades trajectory regression."""
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def infer_regression_source(
    joint_summary: Dict[str, float],
    freeze_summary: Dict[str, float],
) -> str:
    joint_reg = joint_summary.get('val_reg_loss')
    freeze_reg = freeze_summary.get('val_reg_loss')
    joint_mr = joint_summary.get('val_minMR')
    freeze_mr = freeze_summary.get('val_minMR')
    joint_fde = joint_summary.get('val_minFDE')
    freeze_fde = freeze_summary.get('val_minFDE')
    joint_ade = joint_summary.get('val_minADE')
    freeze_ade = freeze_summary.get('val_minADE')

    required = [joint_reg, freeze_reg, joint_mr, freeze_mr, joint_fde, freeze_fde, joint_ade, freeze_ade]
    if any(value is None for value in required):
        return 'unresolved'

    reg_gap = joint_reg - freeze_reg
    mr_ratio = joint_mr / max(freeze_mr, 1e-8)
    fde_ratio = joint_fde / max(freeze_fde, 1e-8)
    ade_ratio = joint_ade / max(freeze_ade, 1e-8)

    # val_reg_loss is negative log-likelihood style; larger (less negative) means worse.
    if reg_gap > 0.03 and fde_ratio > 1.05 and ade_ratio > 1.05 and mr_ratio > 1.05:
        return 'trajectory_regression_shift'
    if abs(reg_gap) <= 0.03 and mr_ratio > 1.10:
        return 'selection_shift'
    return 'unresolved'


def _load_yaml(path: Path) -> Dict[str, object]:
    import yaml

    if not path.is_file():
        return {}
    with path.open('r', encoding='utf-8') as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _scalar_events_from_file(path: Path) -> Dict[str, List[Tuple[int, float, float]]]:
    import numpy as np

    if not hasattr(np, 'string_'):
        np.string_ = np.bytes_  # type: ignore[attr-defined]
    if not hasattr(np, 'unicode_'):
        np.unicode_ = np.str_  # type: ignore[attr-defined]
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(path))
    accumulator.Reload()
    result: Dict[str, List[Tuple[int, float, float]]] = {}
    for tag in accumulator.Tags().get('scalars', []):
        rows = []
        for item in accumulator.Scalars(tag):
            rows.append((int(item.step), float(item.value), float(item.wall_time)))
        result[tag] = rows
    return result


def _merge_scalar_events(event_dicts: List[Dict[str, List[Tuple[int, float, float]]]]) -> Dict[str, List[Tuple[int, float, float]]]:
    merged: Dict[str, List[Tuple[int, float, float]]] = {}
    for event_dict in event_dicts:
        for tag, rows in event_dict.items():
            merged.setdefault(tag, []).extend(rows)
    for tag, rows in merged.items():
        rows.sort(key=lambda item: (item[0], item[2]))
        deduped: List[Tuple[int, float, float]] = []
        last_step: Optional[int] = None
        for row in rows:
            step = row[0]
            if last_step is not None and step == last_step:
                deduped[-1] = row
            else:
                deduped.append(row)
                last_step = step
        merged[tag] = deduped
    return merged


def load_run_scalars(run_dir: Path) -> Dict[str, List[Tuple[int, float, float]]]:
    event_files = sorted(run_dir.glob('events.out.tfevents.*'))
    event_dicts = [_scalar_events_from_file(path) for path in event_files]
    return _merge_scalar_events(event_dicts)


def list_scalar_tags(run_dir: Path) -> List[str]:
    scalars = load_run_scalars(run_dir)
    return sorted(scalars.keys())


def latest_scalar_values(scalars: Dict[str, List[Tuple[int, float, float]]]) -> Dict[str, float]:
    latest: Dict[str, float] = {}
    for tag, rows in scalars.items():
        if rows:
            latest[tag] = rows[-1][1]
    return latest


def summarize_run(run_dir: Path) -> Dict[str, object]:
    hparams = _load_yaml(run_dir / 'hparams.yaml')
    scalars = load_run_scalars(run_dir)
    return {
        'run_dir': str(run_dir),
        'hparams': hparams,
        'tags': sorted(scalars.keys()),
        'latest': latest_scalar_values(scalars),
    }


def print_summary(prefix: str, summary: Dict[str, object]) -> None:
    latest = summary['latest']
    print('{}:run_dir={}'.format(prefix, summary['run_dir']))
    print('{}:available_tags={}'.format(prefix, ','.join(summary['tags'])))
    for key in (
        'val_minADE',
        'val_minFDE',
        'val_minMR',
        'val_reg_loss',
        'val_risk_loss',
        'val_scene_loss',
        'train_reg_loss_epoch',
        'train_risk_loss_epoch',
        'train_scene_loss_epoch',
    ):
        if key in latest:
            print('{}:{},{}'.format(prefix, key, latest[key]))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--baseline_run_dir', type=str, default=None)
    parser.add_argument('--joint_run_dir', type=str, required=True)
    parser.add_argument('--freeze_run_dir', type=str, required=True)
    args = parser.parse_args()

    baseline_summary = summarize_run(Path(args.baseline_run_dir)) if args.baseline_run_dir else None
    joint_summary = summarize_run(Path(args.joint_run_dir))
    freeze_summary = summarize_run(Path(args.freeze_run_dir))

    print_summary('joint', joint_summary)
    print_summary('freeze', freeze_summary)
    if baseline_summary is not None:
        print_summary('baseline', baseline_summary)

    joint_latest = joint_summary['latest']
    freeze_latest = freeze_summary['latest']

    regression_source = infer_regression_source(
        joint_summary=joint_latest,
        freeze_summary=freeze_latest,
    )
    print('regression_degradation_source={}'.format(regression_source))

    if baseline_summary is not None:
        baseline_latest = baseline_summary['latest']
        for key in ('val_minADE', 'val_minFDE', 'val_minMR'):
            if key in baseline_latest and key in joint_latest and key in freeze_latest:
                print(
                    'compare:{},{},{},{}'.format(
                        key,
                        baseline_latest[key],
                        joint_latest[key],
                        freeze_latest[key],
                    )
                )


if __name__ == '__main__':
    main()
