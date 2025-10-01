# main.py
import os
import numpy as np
from config import Config
from train import train_and_evaluate
from utils import set_seed, save_confusion_matrix
from sklearn.metrics import classification_report
import pprint

def extract_macro_f1(report_dict):
    return report_dict.get('macro avg', {}).get('f1-score', 0.0)


def main():
    cfg = Config()
    set_seed(cfg.seed)

    base_path = ''../data/data_split/npz/in-motion'
    result_dir = './trans_results/'
    os.makedirs(result_dir, exist_ok=True)

    acc_list, f1_list, cm_list = [], [], []

    log_path = os.path.join(result_dir, "log.txt")
    with open(log_path, 'w') as log_file:
        # 🔧 하이퍼파라미터 저장
        log_file.write("[🔧 Hyperparameters]\n")
        log_file.write(pprint.pformat(vars(cfg)) + "\n\n")

        log_file.write("[🔎 Fold Performance]\n\n")

        for pos in range(5):
            print(f"\n🚀 Fold pos_{pos} 시작")
            train_path = os.path.join(base_path, f"train_pos_{pos}.npz")
            test_path = os.path.join(base_path, f"test_pos_{pos}.npz")

            result = train_and_evaluate(cfg, train_path, test_path, pos)

            acc = result['acc']
            macro_f1 = extract_macro_f1(result['report'])

            acc_list.append(acc)
            f1_list.append(macro_f1)

            # 📋 사람 읽기 쉬운 텍스트 포맷
            text_report = classification_report(result["labels"], result["preds"], digits=4, zero_division=0)

            # 🔥 로그 저장
            log_file.write(f"pos_{pos}:\n")
            log_file.write(f"- Accuracy: {acc:.4f}\n")
            log_file.write(f"- Macro F1: {macro_f1:.4f}\n")
            log_file.write(f"[📋 Classification Report]\n")
            log_file.write(text_report + "\n\n")

            # 혼동 행렬 저장
            cm = save_confusion_matrix(
                result["labels"], result["preds"], pos=pos,
                result_dir=result_dir, num_classes=cfg.n_classes
            )
            cm_list.append(cm)

        # 평균 계산 및 저장
        avg_acc = np.mean(acc_list)
        avg_f1 = np.mean(f1_list)
        avg_cm = np.mean(cm_list, axis=0).astype(int)

        save_confusion_matrix(
            y_true=None, y_pred=None, pos="avg",
            result_dir=result_dir, num_classes=cfg.n_classes,
            matrix_override=avg_cm
        )

        log_file.write("[📊 Average Performance]\n")
        log_file.write(f"AVG Accuracy: {avg_acc:.4f}\n")
        log_file.write(f"AVG Macro F1: {avg_f1:.4f}\n")

    print(f"\n🏁 All folds complete.\n⭐ Average Accuracy: {avg_acc:.4f}")


if __name__ == "__main__":
    main()
