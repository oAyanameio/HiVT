# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser
from pathlib import Path
import warnings


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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datamodules import ArgoverseV1DataModule
from datasets import ShiftAugment
from models.hivt import HiVT

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--limit_train_batches', type=float, default=1.0)
    parser.add_argument('--limit_val_batches', type=float, default=1.0)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=5)
    parser.add_argument('--experiment_root', type=str, default='/home/lbh/HiVT/runs')
    parser.add_argument('--experiment_name', type=str, default='hivt')
    parser.add_argument('--experiment_version', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--init_ckpt_path', type=str, default=None)
    # 分布偏移增强（§11）；默认全 0 = 不启用
    parser.add_argument('--shift_history_dropout_p', type=float, default=0.0)
    parser.add_argument('--shift_neighbor_dropout_p', type=float, default=0.0)
    parser.add_argument('--shift_position_noise_std', type=float, default=0.0)
    parser.add_argument('--shift_heading_noise_std', type=float, default=0.0)
    parser.add_argument('--shift_map_jitter_std', type=float, default=0.0)
    parser.add_argument('--shift_lane_dropout_p', type=float, default=0.0)
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    train_transform = ShiftAugment(
        history_dropout_p=args.shift_history_dropout_p,
        neighbor_dropout_p=args.shift_neighbor_dropout_p,
        position_noise_std=args.shift_position_noise_std,
        heading_noise_std=args.shift_heading_noise_std,
        map_jitter_std=args.shift_map_jitter_std,
        lane_dropout_p=args.shift_lane_dropout_p,
    ) or None

    logger = TensorBoardLogger(
        save_dir=args.experiment_root,
        name=args.experiment_name,
        version=args.experiment_version,
        default_hp_metric=False,
    )
    checkpoint_dir = Path(logger.log_dir) / 'checkpoints'
    model_checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor=args.monitor,
        save_top_k=args.save_top_k,
        mode='min',
        filename='{epoch:02d}-{' + args.monitor + ':.4f}',
        save_last=True,
    )
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[model_checkpoint], logger=logger)
    if args.init_ckpt_path:
        model = HiVT.load_from_checkpoint(
            checkpoint_path=args.init_ckpt_path,
            strict=False,
            **vars(args),
        )
    else:
        model = HiVT(**vars(args))
    datamodule = ArgoverseV1DataModule.from_argparse_args(args, train_transform=train_transform)
    trainer.fit(model, datamodule, ckpt_path=args.ckpt_path)
