#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Beacon + STraTS self-supervised pretrain (ss+의 pretrain 단계).

- strats_pos_k.pt 를 사용해 BeaconPretrainDataset 구성
- args.pretrain = 1 로 설정하여 Strats가 forecast loss(MSE)를 학습
- best validation loss 기준으로 checkpoint 저장
"""

import argparse
import os

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from utils import Logger, set_all_seeds
from models import count_parameters
from modeling_strats import Strats
from beacon_pretrain_dataset import BeaconPretrainDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # dataset 관련
    parser.add_argument('--dataset', type=str, default='beacon',
                        choices=['beacon'])
    parser.add_argument('--run', type=str, default='1o10')          # seed 변화용

    # Beacon 전용 인자
    parser.add_argument(
        '--beacon_data_root',
        type=str,
        default='./data_split_beacon_strats/final_pt/in-motion',
        help='strats_pos_k.pt가 있는 폴더 경로'
    )
    parser.add_argument(
        '--pos_split',
        type=str,
        default='pos_4',
        help='사용할 pos split 이름 (pos_0 ~ pos_4)'
    )
    parser.add_argument(
        '--pretrain_val_frac',
        type=float,
        default=0.1,
        help='self-supervised pretrain에서 validation으로 사용할 비율'
    )

    # model 관련 (STraTS 하이퍼파라미터)
    parser.add_argument('--model_type', type=str, default='strats',
                        choices=['strats', 'istrats'])
    parser.add_argument('--max_obs', type=int, default=880)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attention_dropout', type=float, default=0.2)

    # training 관련
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_dir_prefix', type=str, default='pretrain_')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=32)

    # (Beacon Zone 개수: 24 → 아키텍처를 동일하게 맞춰주기 위해 명시)
    parser.add_argument('--num_classes', type=int, default=24)

    args = parser.parse_args()
    return args


def set_output_dir(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        args.output_dir = './outputs/' + args.output_dir_prefix
        args.output_dir += args.model_type
        args.output_dir += f"|pos:{args.pos_split}"
    os.makedirs(args.output_dir, exist_ok=True)


class BeaconPretrainEvaluator:
    """self-supervised pretrain용 evaluator (단순 MSE loss 평균)."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def evaluate(self, model, dataset, split: str):
        model.eval()
        indices = dataset.splits[split]
        bs = self.args.eval_batch_size

        total_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for start in range(0, len(indices), bs):
                ind = indices[start:start + bs]
                batch = dataset.get_batch(ind)
                batch = {k: v.to(self.args.device) for k, v in batch.items()}

                loss = model(**batch)
                bsz = batch['values'].size(0)
                total_loss += loss.item() * bsz
                total_count += bsz

        avg_loss = total_loss / max(total_count, 1)
        self.args.logger.write(f"\n[{split}] | avg MSE loss={avg_loss:.4f}")
        return avg_loss


if __name__ == "__main__":
    # 1) 설정
    args = parse_args()
    set_output_dir(args)
    args.logger = Logger(args.output_dir, 'log.txt')
    args.logger.write('\n' + str(args))
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.pretrain = 1  # 🔴 self-supervised 모드
    set_all_seeds(args.seed + int(args.run.split('o')[0]))
    model_path_best = os.path.join(args.output_dir, 'checkpoint_best.bin')

    # 2) Dataset / Model
    dataset = BeaconPretrainDataset(args)
    # BeaconPretrainDataset에서 args.V, args.D 세팅됨
    model = Strats(args)
    model.to(args.device)
    count_parameters(args.logger, model)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    evaluator = BeaconPretrainEvaluator(args)

    num_train = len(dataset.splits['train'])
    num_batches_per_epoch = int(np.ceil(num_train / args.train_batch_size))
    args.logger.write('\nNo. of training batches per epoch = '
                      + str(num_batches_per_epoch))

    best_val_loss = np.inf
    best_epoch = -1

    # 3) 학습 루프
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        cum_train_loss = 0.0
        num_batches_trained = 0

        pbar = tqdm(range(num_batches_per_epoch),
                    desc=f"[Pretrain] Epoch {epoch}/{args.max_epochs}",
                    ncols=120)
        for _ in pbar:
            batch = dataset.get_batch()  # train_cycler에서 뽑음
            batch = {k: v.to(args.device) for k, v in batch.items()}

            loss = model(**batch)

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                optimizer.step()
                optimizer.zero_grad()

            cum_train_loss += loss.item()
            num_batches_trained += 1
            avg_loss_running = cum_train_loss / max(num_batches_trained, 1)
            pbar.set_postfix(loss=f"{avg_loss_running:.4f}")

        avg_train_loss = cum_train_loss / max(num_batches_trained, 1)
        args.logger.write(f"\n[pretrain-train] epoch {epoch} | loss={avg_train_loss:.4f}")

        # --- validation ---
        val_loss = evaluator.evaluate(model, dataset, 'val')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            args.logger.write('\nSaving ckpt at ' + model_path_best)
            torch.save(model.state_dict(), model_path_best)

    args.logger.write(f"\n[Pretrain] Best epoch: {best_epoch} | best val loss={best_val_loss:.4f}")
