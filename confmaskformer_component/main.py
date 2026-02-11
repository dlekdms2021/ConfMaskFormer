# main.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, numpy as np, torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from data_loader import RSSIWindowNPZDataset, make_dataloaders_from_file
from transformer_model import TransformerClassifierImproved
from trainer import step_one_epoch, build_optimizer_scheduler, predict_all
from utils import save_confusion_matrix, set_seed

class Cfg:
    window_size     = 10
    embedding_dim   = 96
    n_heads         = 8
    n_layers        = 2
    dropout         = 0.3
    lr              = 1e-3
    weight_decay    = 1e-3
    batch_size      = 256
    epochs          = 100
    alpha_neighbor  = 0.20
    aux_mask_ratio  = 0.30
    aux_loss_weight = 0.40
    beacon_dropout_p= 0.30
    num_workers     = 0
    seed            = 42
    # ========== 구성요소 on/off 플래그 ==========
    use_confidence_gate  = True   # Confidence Gate 활성화
    use_auxiliary_loss   = True   # Auxiliary Reconstruction Loss 활성화
    use_beacon_dropout   = True   # Beacon Dropout 활성화
    use_combined_input   = True   # RSSI + Mask 결합 입력 활성화

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_root", type=str, default="../data/data_split/npz/in-motion", help="예: ./data_split/npz/in-motion")
    parser.add_argument("--epochs", type=int, default=Cfg.epochs)
    parser.add_argument("--batch_size", type=int, default=Cfg.batch_size)
    parser.add_argument("--embed", type=int, default=Cfg.embedding_dim)
    parser.add_argument("--heads", type=int, default=Cfg.n_heads)
    parser.add_argument("--layers", type=int, default=Cfg.n_layers)
    parser.add_argument("--dropout", type=float, default=Cfg.dropout)
    parser.add_argument("--lr", type=float, default=Cfg.lr)
    parser.add_argument("--weight_decay", type=float, default=Cfg.weight_decay)
    parser.add_argument("--alpha_neighbor", type=float, default=Cfg.alpha_neighbor)
    parser.add_argument("--aux_mask_ratio", type=float, default=Cfg.aux_mask_ratio)
    parser.add_argument("--aux_loss_weight", type=float, default=Cfg.aux_loss_weight)
    parser.add_argument("--beacon_dropout_p", type=float, default=Cfg.beacon_dropout_p)
    parser.add_argument("--val_ratio", type=float, default=0.0, help="train 내부 검증 분할(옵션)")
    parser.add_argument("--results_dir", type=str, default="./results")
    # ========== 구성요소 on/off 옵션 ==========
    parser.add_argument("--use_confidence_gate", type=lambda x: x.lower()=='true', default=Cfg.use_confidence_gate, help="Confidence Gate 활성화")
    parser.add_argument("--use_auxiliary_loss", type=lambda x: x.lower()=='true', default=Cfg.use_auxiliary_loss, help="Auxiliary Loss 활성화")
    parser.add_argument("--use_beacon_dropout", type=lambda x: x.lower()=='true', default=Cfg.use_beacon_dropout, help="Beacon Dropout 활성화")
    parser.add_argument("--use_combined_input", type=lambda x: x.lower()=='true', default=Cfg.use_combined_input, help="RSSI+Mask 결합 입력 활성화")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    set_seed(Cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = [f"pos_{i}" for i in range(5)]
    all_acc, all_f1 = [], []

    # ========== 구성요소 구성 로그 ==========
    config_str = f"[CONFIG] Confidence Gate: {args.use_confidence_gate} | Auxiliary Loss: {args.use_auxiliary_loss} | Beacon Dropout: {args.use_beacon_dropout} | Combined Input: {args.use_combined_input}"
    print(config_str, flush=True)
    with open(os.path.join(args.results_dir, "config_log.txt"), "w") as f:
        f.write(config_str + "\n")

    # 클래스 수 추정
    first_train = os.path.join(args.npz_root, "train_pos_0.npz")
    tmp_ds = RSSIWindowNPZDataset(first_train)
    n_classes = int(tmp_ds.y.max() + 1)

    cm_accum = None
    n_done_folds = 0

    for s in splits:
        print(f"\n=== Split {s} ===", flush=True)
        split_dir = os.path.join(args.results_dir, s)
        os.makedirs(split_dir, exist_ok=True)

        train_npz = os.path.join(args.npz_root, f"train_{s}.npz")
        test_npz  = os.path.join(args.npz_root, f"test_{s}.npz")
        print("train_npz:", train_npz, "exists:", os.path.isfile(train_npz), flush=True)
        print("test_npz :", test_npz,  "exists:", os.path.isfile(test_npz),  flush=True)
        if not (os.path.isfile(train_npz) and os.path.isfile(test_npz)):
            raise FileNotFoundError(f"Split files not found: {train_npz}, {test_npz}")

        # DataLoaders
        train_loader, val_loader = make_dataloaders_from_file(
            train_npz, batch_size=args.batch_size, val_ratio=args.val_ratio, num_workers=Cfg.num_workers
        )
        test_loader, _ = make_dataloaders_from_file(
            test_npz, batch_size=args.batch_size, val_ratio=0.0, num_workers=Cfg.num_workers
        )

        # 모델 구성 파라미터 동기화
        sample_x, _ = next(iter(train_loader))
        T, F = sample_x.shape[1], sample_x.shape[2]
        class ModelCfg:
            window_size = T
            embedding_dim = args.embed
            n_heads = args.heads
            n_layers = args.layers
            dropout = args.dropout

        model = TransformerClassifierImproved(
            n_beacons=F,
            n_classes=n_classes,
            cfg=ModelCfg,
            aux_mask_ratio=args.aux_mask_ratio,
            aux_loss_weight=args.aux_loss_weight,
            beacon_dropout_p=args.beacon_dropout_p,
            use_confidence_gate=args.use_confidence_gate,
            use_auxiliary_loss=args.use_auxiliary_loss,
            use_beacon_dropout=args.use_beacon_dropout,
            use_combined_input=args.use_combined_input
        ).to(device)

        optimizer, scheduler = build_optimizer_scheduler(
            model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
        )

        # ---- Best tracking (TEST 기준) ----
        best_test_acc  = -1.0
        best_test_f1   = -1.0
        best_epoch     = -1
        best_y_true    = None
        best_y_pred    = None
        best_state     = None
        best_report    = ""

        for epoch in range(1, args.epochs + 1):
            tr = step_one_epoch(
                model, train_loader, optimizer, scheduler, device, n_classes, train=True,
                alpha_neighbor=args.alpha_neighbor
            )

            if val_loader is not None:
                va = step_one_epoch(
                    model, val_loader, optimizer=None, scheduler=None, device=device,
                    n_classes=n_classes, train=False, alpha_neighbor=args.alpha_neighbor
                )
                print(f"[{s}][Ep {epoch:02d}] train: loss={tr['loss']:.4f} aux={tr['aux_loss']:.4f} acc={tr['acc']:.4f} | "
                      f"val: loss={va['loss']:.4f} aux={va['aux_loss']:.4f} acc={va['acc']:.4f}")
            else:
                print(f"[{s}][Ep {epoch:02d}] train: loss={tr['loss']:.4f} aux={tr['aux_loss']:.4f} acc={tr['acc']:.4f}")

            # TEST 평가 (best를 TEST 기준으로 갱신)
            y_true_tmp, y_pred_tmp = predict_all(model, test_loader, device)
            acc_tmp  = accuracy_score(y_true_tmp, y_pred_tmp)
            f1_tmp   = f1_score(y_true_tmp, y_pred_tmp, average="macro")
            print(f"[{s}][Ep {epoch:02d}] TEST: acc={acc_tmp:.4f}, macroF1={f1_tmp:.4f}")

            # 갱신 기준: acc 우선, 동률이면 macro-F1
            if (acc_tmp > best_test_acc) or (np.isclose(acc_tmp, best_test_acc) and f1_tmp > best_test_f1):
                best_test_acc = acc_tmp
                best_test_f1  = f1_tmp
                best_epoch    = epoch
                best_y_true   = y_true_tmp
                best_y_pred   = y_pred_tmp
                best_state    = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                best_report   = classification_report(best_y_true, best_y_pred, digits=4, zero_division=0)

        # ---- Best model 로드 및 저장 ----
        if best_state is not None:
            model.load_state_dict(best_state)
            save_path = os.path.join(split_dir, f"best_{s}.pt")
            # ✅ 저장 폴더 재보장 (중간에 폴더가 지워졌거나 마운트 문제 대비)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            try:
                torch.save(model.state_dict(), save_path)
                print(f"✔ [{s}] Saved best model → {save_path}")
            except OSError as e:
                raise OSError(f"Failed to save model to {save_path}. Check disk/mount/permissions.") from e
        else:
            print(f"⚠ [{s}] best_state is None — no improvement recorded for this split.")

        # ---- Fold best 결과 저장 ----
        # ✅ 가드: 만약 best 상태가 없다면(이상 케이스) 스킵
        if (best_y_true is None) or (best_y_pred is None):
            print(f"⚠ [{s}] Skip saving fold artifacts (no best predictions).")
            continue

        cm_best = confusion_matrix(best_y_true, best_y_pred, labels=list(range(n_classes)))
        save_confusion_matrix(
            y_true=None, y_pred=None, pos=s, result_dir=split_dir,
            num_classes=n_classes, matrix_override=cm_best, filename=f"confmat_{s}.png"
        )
        np.save(os.path.join(split_dir, f"y_true_{s}.npy"), best_y_true)
        np.save(os.path.join(split_dir, f"y_pred_{s}.npy"), best_y_pred)
        with open(os.path.join(split_dir, f"metrics_{s}.txt"), "w", encoding="utf-8") as f:
            f.write(f"[{s}] Best epoch: {best_epoch}\n")
            f.write(f"Best TEST Acc   : {best_test_acc:.4f}\n")
            f.write(f"Best TEST MacroF1: {best_test_f1:.4f}\n\n")
            f.write("=== Classification Report (best) ===\n")
            f.write(best_report)

            f.write(best_report)

        all_acc.append(best_test_acc)
        all_f1.append(best_test_f1)

        if cm_accum is None:
            cm_accum = cm_best.astype(np.float64)
        else:
            cm_accum += cm_best.astype(np.float64)
        n_done_folds += 1

        print(f"=== [{s}] Best Test ===")
        print(f"ACC = {best_test_acc:.4f}, Macro-F1 = {best_test_f1:.4f}")
        print(best_report)

    # ---- 전체 요약 및 평균 CM 저장 ----
    avg_acc = float(np.mean(all_acc)); std_acc = float(np.std(all_acc))
    avg_f1  = float(np.mean(all_f1));  std_f1  = float(np.std(all_f1))

    cm_avg = cm_accum / max(1, n_done_folds)
    cm_avg_int = np.rint(cm_avg).astype(int)
    save_confusion_matrix(
        y_true=None, y_pred=None, pos='avg', result_dir=args.results_dir,
        num_classes=n_classes, matrix_override=cm_avg_int, filename="confmat_avg.png"
    )
    np.savetxt(os.path.join(args.results_dir, "confmat_avg.csv"), cm_avg, delimiter=",")

    with open(os.path.join(args.results_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("K-Fold Best-by-Fold Summary\n")
        f.write("===========================\n")
        for i, s in enumerate(splits):
            if i < len(all_acc):
                f.write(f"{s}: acc={all_acc[i]:.4f}, macroF1={all_f1[i]:.4f}\n")
        f.write("\n")
        f.write(f"Avg ACC     : {avg_acc:.4f}  (± {std_acc:.4f})\n")
        f.write(f"Avg Macro-F1: {avg_f1:.4f}  (± {std_f1:.4f})\n")

    print("\n======================")
    print("K-Fold Best-by-Fold Result")
    print("======================")
    print(f"Avg ACC     : {avg_acc:.4f}  (± {std_acc:.4f})")
    print(f"Avg Macro-F1: {avg_f1:.4f}  (± {std_f1:.4f})")
    print(f"Saved: {os.path.join(args.results_dir, 'confmat_avg.png')} / confmat_avg.csv / summary.txt")

if __name__ == "__main__":
    main()
