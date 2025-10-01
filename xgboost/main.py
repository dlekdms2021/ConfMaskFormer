import os
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils import shuffle

# 📁 경로 설정
data_dir = '../data/data_split/npz/in-motion' 
result_dir = './xgb_results/in-motion'
os.makedirs(result_dir, exist_ok=True)

# ✅ 사용할 단일 하이퍼파라미터 조합
param = {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.7}

print(f"\n🔧 사용 중인 하이퍼파라미터 조합: {param}")

acc_list, f1_list = [], []
cm_total = np.zeros((24, 24), dtype=int)

log_path = os.path.join(result_dir, 'log.txt')
with open(log_path, 'w') as f_log:
    f_log.write(f"[🔧 Hyperparameters]\n{param}\n")

    for pos in range(5):
        print(f"  🚀 Fold pos_{pos} 실행 중...")
        f_log.write(f"\n\n[Fold pos_{pos}]\n")

        # 📥 데이터 로딩
        train_path = os.path.join(data_dir, f'train_pos_{pos}.npz')
        test_path = os.path.join(data_dir, f'test_pos_{pos}.npz')
        train_data = np.load(train_path)['data']
        test_data = np.load(test_path)['data']

        X_train = train_data[:, :, :5].reshape(train_data.shape[0], -1)
        y_train = train_data[:, 0, 5].astype(int) - 1
        X_train, y_train = shuffle(X_train, y_train, random_state=42)

        X_test = test_data[:, :, :5].reshape(test_data.shape[0], -1)
        y_test = test_data[:, 0, 5].astype(int) - 1

        # 모델 학습
        clf = xgb.XGBClassifier(
            **param,
            objective='multi:softprob',
            eval_metric='mlogloss',
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            random_state=42,
            verbosity=0
        )
        clf.fit(X_train, y_train)

        # 예측 및 평가
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        acc_list.append(acc)
        f1_list.append(f1)

        # 🔹 로그 기록
        f_log.write(f"Accuracy: {acc:.4f}, Macro F1: {f1:.4f}\n")

        # 🔹 Classification report 저장
        cls_report = classification_report(y_test, y_pred, labels=list(range(24)), digits=4)
        f_log.write("[📋 Classification Report]\n")
        f_log.write(cls_report + "\n")

        # 🔹 Confusion Matrix 시각화
        cm = confusion_matrix(y_test, y_pred, labels=list(range(24)))
        cm_total += cm

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
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        save_path = os.path.join(result_dir, f"confmat_pos_{pos}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[✓] Confusion matrix saved to '{save_path}'")

    # ✅ 평균 저장
    avg_acc = np.mean(acc_list)
    avg_f1 = np.mean(f1_list)

    f_log.write("\n\n[📊 Average Performance]\n")
    f_log.write(f"Avg Accuracy: {avg_acc:.4f}\n")
    f_log.write(f"Avg Macro F1: {avg_f1:.4f}\n")

    print(f"\n✅ 평균 Accuracy: {avg_acc:.4f}, 평균 Macro F1: {avg_f1:.4f}")

# ✅ 최종 Confusion Matrix 시각화
cm_avg = cm_total // 5

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
for i in range(cm_avg.shape[0]):
    for j in range(cm_avg.shape[1]):
        plt.text(j, i, format(cm_avg[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm_avg[i, j] > thresh else "black")

plt.tight_layout()
save_path = os.path.join(result_dir, "confmat_avg.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[✓] Average confusion matrix saved to '{save_path}'")
