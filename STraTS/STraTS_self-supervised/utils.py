"""This file contain common utility functions."""
from datetime import datetime
import string
import os
import random
import json
from pytz import timezone
from tqdm import tqdm
tqdm.pandas()
from transformers import set_seed
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Optimizer
from typing import Any, Union
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def get_curr_time() -> str:
    """Get current date and time in PST as str."""
    return datetime.now().astimezone(
            timezone('US/Pacific')).strftime("%d/%m/%Y %H:%M:%S")


class Logger: 
    """Class to write message to both output_dir/filename.txt and terminal."""
    def __init__(self, output_dir: str=None, filename: str=None) -> None:
        if filename is not None:
            self.log = os.path.join(output_dir, filename)

    def write(self, message: Any, show_time: bool=True) -> None:
        "write the message"
        message = str(message)
        if show_time:
            # if message starts with \n, print the \n first before printing time
            if message.startswith('\n'): 
                message = '\n'+get_curr_time()+' >> '+message[1:]
            else:
                message = get_curr_time()+' >> '+message
        print (message)
        if hasattr(self, 'log'):
            with open(self.log, 'a') as f:
                f.write(message+'\n')


def set_all_seeds(seed: int) -> None:
    """Function to set seeds for all RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count()>0:
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    set_seed(seed)



class CycleIndex:
    """Class to generate batches of training ids, 
    shuffled after each epoch.""" 
    def __init__(self, indices:Union[int,list], batch_size: int,
                 shuffle: bool=True) -> None:
        if type(indices)==int:
            indices = np.arange(indices)
        self.indices = indices
        self.num_samples = len(indices)
        self.batch_size = batch_size
        self.pointer = 0
        if shuffle:
            np.random.shuffle(self.indices)
        self.shuffle = shuffle

    def get_batch_ind(self):
        """Get indices for next batch."""
        start, end = self.pointer, self.pointer + self.batch_size
        # If we have a full batch within this epoch, then get it.
        if end <= self.num_samples:
            if end==self.num_samples:
                self.pointer = 0
                if self.shuffle:
                    np.random.shuffle(self.indices)
            else:
                self.pointer = end
            return self.indices[start:end]
        # Otherwise, fill the batch with samples from next epoch.
        last_batch_indices_incomplete = self.indices[start:]
        remaining = self.batch_size - (self.num_samples-start)
        self.pointer = remaining
        if self.shuffle:
            np.random.shuffle(self.indices)
        return np.concatenate((last_batch_indices_incomplete, 
                               self.indices[:remaining]))
    



def save_confusion_matrix(
    y_true=None, y_pred=None, pos: str | int = 0,
    result_dir: str = './results',
    num_classes: int = 24,
    matrix_override: np.ndarray = None,
    filename: str = None
):
    """
    - y_true/y_pred가 주어지면 sklearn.confusion_matrix로 계산
    - matrix_override가 주어지면 그대로 그림만 그림(평균 혼동행렬용)
    - pos: 'pos_0', 'pos_1', ... 또는 'avg'
    - result_dir: 저장 폴더 (보통 args.output_dir)
    - filename: 파일 이름 직접 지정하고 싶을 때
    """
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
