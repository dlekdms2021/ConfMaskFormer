#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from sklearn.metrics import classification_report, f1_score

from utils import Logger, set_all_seeds, save_confusion_matrix
from models import count_parameters
from beacon_dataset import BeaconDataset
from modeling_strats import Strats


# ----------------------- argument parser -----------------------


def parse_args() -> argparse.Namespace:
    """Beacon + STraTS (supervised, ss- 또는 ss+ fine-tune)."""
    parser = argparse.ArgumentParser()

    # dataset 관련
    parser.add_argument('--dataset', type=str, default='beacon',
                        choices=['beacon'])
    parser.add_argument('--train_frac', type=float, default=0.5)   # 형식상 남김
    parser.add_argument('--run', type=str, default='1o10')          # seed 변화용

    # 🔹 Beacon 전용 인자
    parser.add_argument(
        '--beacon_data_root',
        type=str,
        default='./data_split_beacon_strats/final_pt/in-motion',
        help='strats_pos_k.pt가 있는 폴더 경로'
    )
    parser.add_argument(
        '--pos_split',
        type=str,
        default='pos_1',
        help='사용할 pos split 이름 (pos_0 ~ pos_4)'
    )
    parser.add_argument(
        '--val_frac',
        type=float,
        default=0.0,   # val 안 쓸 거라 0.0으로
        help='(사용 안함) train 중 일부를 validation으로 사용할 비율'
    )

    # model 관련 (STraTS 하이퍼파라미터)
    parser.add_argument('--model_type', type=str, default='strats',
                        choices=['strats', 'istrats'])
    parser.add_argument('--load_ckpt_path', type=str, default=None)  # 🔴 pretrain ckpt 경로

    parser.add_argument('--max_obs', type=int, default=880)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attention_dropout', type=float, default=0.2)

    # training/eval 관련
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_dir_prefix', type=str, default='')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=32)

    return parser.parse_args()


def set_output_dir(args: argparse.Namespace) -> None:
    """output_dir 자동 설정."""
    if args.output_dir is None:
        args.output_dir = './outputs/' + args.output_dir_prefix
        args.output_dir += args.model_type
        args.output_dir += f"|pos:{args.pos_split}"
    os.makedirs(args.output_dir, exist_ok=True)


# ----------------------- Beacon 전용 evaluator -----------------------


class BeaconEvaluator:
    """다중 클래스 분류(Zone)용 evaluator. y_true / y_pred까지 뽑는다."""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def predict_and_score(self, model, dataset, split: str):
        """
        return:
          y_true: np.ndarray, shape (N,)
          y_pred: np.ndarray, shape (N,)
          avg_loss: float
          acc: float
        """
        model.eval()
        indices = dataset.splits[split]
        bs = self.args.eval_batch_size

        all_loss = 0.0
        all_correct = 0
        all_total = 0
        ys_true = []
        ys_pred = []

        with torch.no_grad():
            for start in range(0, len(indices), bs):
                ind = indices[start:start + bs]
                batch = dataset.get_batch(ind)
                labels = batch.pop('labels')     # LongTensor (B,)
                labels = labels.to(self.args.device)

                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                logits = model(**batch, labels=None)  # logits (B,C)
                loss = F.cross_entropy(logits, labels)

                preds = logits.argmax(dim=-1)

                all_loss += loss.item() * labels.size(0)
                all_correct += (preds == labels).sum().item()
                all_total += labels.size(0)

                ys_true.append(labels.cpu().numpy())
                ys_pred.append(preds.cpu().numpy())

        if all_total == 0:
            avg_loss = 0.0
            acc = 0.0
            y_true = np.array([], dtype=np.int64)
            y_pred = np.array([], dtype=np.int64)
        else:
            avg_loss = all_loss / all_total
            acc = all_correct / all_total
            y_true = np.concatenate(ys_true)
            y_pred = np.concatenate(ys_pred)

        self.args.logger.write(
            f"\n[{split}] | loss={avg_loss:.4f}, acc={acc:.4f}"
        )

        return y_true, y_pred, avg_loss, acc


# ------------------------------ main ------------------------------


if __name__ == "__main__":
    # 1) 설정/로그/디바이스
    args = parse_args()
    set_output_dir(args)
    args.logger = Logger(args.output_dir, 'log.txt')
    args.logger.write('\n' + str(args))
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.pretrain = 0  # 🔵 supervised (fine-tune) 모드
    set_all_seeds(args.seed + int(args.run.split('o')[0]))
    model_path_best = os.path.join(args.output_dir, 'checkpoint_best.bin')

    # 2) 데이터셋 로드
    dataset = BeaconDataset(args)
    args.V = dataset.V
    args.D = dataset.D
    args.num_classes = dataset.num_classes  # 24

    # 3) 모델 생성
    model = Strats(args)

    # 🔴 self-supervised pretrain ckpt 로드 (ss+ 세팅)
    if args.load_ckpt_path is not None and os.path.exists(args.load_ckpt_path):
        args.logger.write(f"\nLoading pretrained weights from {args.load_ckpt_path}")
        state_dict = torch.load(args.load_ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)

    model.to(args.device)
    count_parameters(args.logger, model)

    # 4) Optimizer & Evaluator 세팅
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    evaluator = BeaconEvaluator(args)

    # 5) epoch / batch 계산
    num_train = len(dataset.splits['train'])
    num_batches_per_epoch = int(np.ceil(num_train / args.train_batch_size))
    args.logger.write('\nNo. of training batches per epoch = '
                      + str(num_batches_per_epoch))

    # 🔹 best 결과 저장용 변수들
    best_epoch = -1
    best_test_acc = -np.inf
    best_test_macro_f1 = -np.inf
    best_report_str = ""
    best_cm = None

    # 6) 학습 루프
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        cum_train_loss = 0.0
        num_batches_trained = 0

        pbar = tqdm(range(num_batches_per_epoch),
                    desc=f"Epoch {epoch}/{args.max_epochs}",
                    ncols=120)
        for _ in pbar:
            batch = dataset.get_batch()                     # {'values','times','varis','obs_mask','demo','labels'}
            batch = {k: v.to(args.device) for k, v in batch.items()}

            loss = model(**batch)  # labels 포함 → loss 반환

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                optimizer.step()
                optimizer.zero_grad()

            cum_train_loss += loss.item()
            num_batches_trained += 1
            avg_loss_running = cum_train_loss / num_batches_trained
            pbar.set_postfix(loss=f"{avg_loss_running:.4f}")

        avg_train_loss = cum_train_loss / max(num_batches_trained, 1)
        args.logger.write(f"\n[train] epoch {epoch} | loss={avg_train_loss:.4f}")

        # --- 매 epoch 끝에 test 평가 ---
        y_true, y_pred, test_loss, test_acc = evaluator.predict_and_score(
            model, dataset, 'test'
        )
        if y_true.size == 0:
            args.logger.write(f"[WARN] test split is empty. Skipping metrics.")
            continue

        macro_f1 = f1_score(y_true, y_pred, average='macro')
        args.logger.write(
            f"[test] epoch {epoch} | loss={test_loss:.4f}, acc={test_acc:.4f}, macroF1={macro_f1:.4f}"
        )

        # 🔹 best 갱신 기준: MacroF1
        if macro_f1 > best_test_macro_f1:
            best_test_macro_f1 = macro_f1
            best_test_acc = test_acc
            best_epoch = epoch

            # classification report 문자열 저장
            report_str = classification_report(
                y_true,
                y_pred,
                digits=4,
                zero_division=0,
            )
            best_report_str = report_str

            # 혼동행렬 png + npy 저장
            cm = save_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                pos=args.pos_split,   # 예: "pos_0"
                result_dir=args.output_dir,
                num_classes=args.num_classes,
                filename=f"confmat_{args.pos_split}.png"
            )
            best_cm = cm
            np.save(
                os.path.join(args.output_dir, f"confmat_{args.pos_split}.npy"),
                cm
            )

            # 모델 checkpoint 저장
            args.logger.write('\nSaving ckpt at ' + model_path_best)
            torch.save(model.state_dict(), model_path_best)

    # 7) 최종 결과 로그 + txt 파일로 저장
    args.logger.write(f"\n[{args.pos_split}] Best epoch: {best_epoch}")
    args.logger.write(f"Best TEST Acc   : {best_test_acc:.4f}")
    args.logger.write(f"Best TEST MacroF1: {best_test_macro_f1:.4f}")
    args.logger.write("\n=== Classification Report (best) ===\n" + best_report_str)

    txt_path = os.path.join(args.output_dir, f"report_{args.pos_split}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"[{args.pos_split}] Best epoch: {best_epoch}\n")
        f.write(f"Best TEST Acc   : {best_test_acc:.4f}\n")
        f.write(f"Best TEST MacroF1: {best_test_macro_f1:.4f}\n\n")
        f.write("=== Classification Report (best) ===\n")
        f.write(best_report_str)
    print(f"[✓] Classification report saved to '{txt_path}'")

    if best_cm is not None:
        print(f"[✓] Best confusion matrix saved for {args.pos_split}")
