# utils.py
# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# -----------------------------
# Seed
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 성능/재현 균형: 아래 두 줄은 환경/필요에 맞게 선택
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# -----------------------------
# Confusion Matrix (save)
# -----------------------------
def save_confusion_matrix(
    y_true=None, y_pred=None, pos: str | int = 0,
    result_dir: str = './results',
    num_classes: int = 24,
    matrix_override: np.ndarray = None,
    filename: str = None
):
    if matrix_override is not None:
        cm = matrix_override
    else:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    plt.figure(figsize=(10, 8))
    cmap = plt.cm.Oranges if pos == 'avg' else plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f"Confusion Matrix - {pos}" if pos != 'avg' else "Average Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(1, num_classes + 1)
    plt.xticks(tick_marks - 1, tick_marks)
    plt.yticks(tick_marks - 1, tick_marks)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    os.makedirs(result_dir, exist_ok=True)
    if filename is None:
        filename = f"confmat_{pos}.png"
    save_path = os.path.join(result_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[✓] Confusion matrix saved to '{save_path}'")
    return cm
