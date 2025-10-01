import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils import shuffle
import traceback
import matplotlib
matplotlib.use('Agg')

# 📁 데이터 경로
data_dir = '../data/data_split/npz/in-motion' 
result_dir = './rf_results/in-motion'
os.makedirs(result_dir, exist_ok=True)

# ✅ 사용할 단일 하이퍼파라미터 조합
param_dict = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300}


print(f"\n🔧 사용 중인 하이퍼파라미터 조합: {param_dict}")

acc_list, f1_list = [], []
cm_total = np.zeros((24, 24), dtype=int)
success_pos = []
report_dict = {}

# 🔁 5개의 fold에 대해 반복
for pos in range(5):
    print(f"\n🚀 Fold pos_{pos}")
    try:
        train_path = os.path.join(data_dir, f'train_pos_{pos}.npz')
        test_path = os.path.join(data_dir, f'test_pos_{pos}.npz')
        train_data = np.load(train_path)['data']
        test_data = np.load(test_path)['data']

        X_train = train_data[:, :, :5].reshape(train_data.shape[0], -1)
        y_train = train_data[:, 0, 5].astype(int) - 1
        X_train, y_train = shuffle(X_train, y_train, random_state=42)

        X_test = test_data[:, :, :5].reshape(test_data.shape[0], -1)
        y_test = test_data[:, 0, 5].astype(int) - 1

        clf = RandomForestClassifier(**param_dict, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        acc_list.append((pos, acc))
        f1_list.append((pos, f1))
        success_pos.append(pos)

        report = classification_report(y_test, y_pred, digits=4)
        report_dict[pos] = report

        print(f"✔️ Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
        print("📋 Classification Report:\n", report)

        cm = confusion_matrix(y_test, y_pred, labels=list(range(24)))
        cm_total += cm

        # 🔹 혼동행렬 시각화
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - pos_{pos}")
        plt.colorbar()
        tick_marks = np.arange(1, 25)
        plt.xticks(tick_marks - 1, tick_marks)
        plt.yticks(tick_marks - 1, tick_marks)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        thresh = cm.max() / 2
        for i in range(24):
            for j in range(24):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f"confmat_pos_{pos}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"[❌] Error in Fold pos_{pos}: {e}")
        traceback.print_exc()

# ✅ 평균 성능 및 결과 저장
if success_pos:
    avg_acc = np.mean([v for _, v in acc_list])
    avg_f1 = np.mean([v for _, v in f1_list])
    cm_avg = cm_total // len(success_pos)

    print("\n📊 성능 요약:")
    for i, acc in acc_list:
        f1 = dict(f1_list)[i]
        print(f"pos_{i} → Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    print(f"\n✅ 평균 Accuracy: {avg_acc:.4f}")
    print(f"✅ 평균 Macro F1: {avg_f1:.4f}")

    # 🔸 로그 파일 저장
    with open(os.path.join(result_dir, "log.txt"), 'w') as f:
        f.write(f"[🔧 Hyperparameters]\n{param_dict}\n\n")
        f.write("[🔎 Fold Performance]\n")
        for i, acc in acc_list:
            f1_score_val = dict(f1_list)[i]
            f.write(f"\npos_{i}:\n")
            f.write(f"- Accuracy: {acc:.4f}\n")
            f.write(f"- Macro F1: {f1_score_val:.4f}\n")
            f.write("[📋 Classification Report]\n")
            f.write(report_dict[i] + "\n")
        f.write("\n[📊 Average Performance]\n")
        f.write(f"AVG Accuracy: {avg_acc:.4f}\n")
        f.write(f"AVG Macro F1: {avg_f1:.4f}\n")

    # 🔸 평균 혼동행렬 시각화 저장
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_avg, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title("Average Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(1, 25)
    plt.xticks(tick_marks - 1, tick_marks)
    plt.yticks(tick_marks - 1, tick_marks)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    thresh = cm_avg.max() / 2
    for i in range(24):
        for j in range(24):
            plt.text(j, i, str(cm_avg[i, j]), ha="center", va="center",
                     color="white" if cm_avg[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "confmat_avg.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Confmat avg saved to {os.path.join(result_dir, 'confmat_avg.png')}")
else:
    print("\n⚠️ 모든 Fold에서 오류가 발생했습니다.")
