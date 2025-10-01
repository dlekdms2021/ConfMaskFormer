# trainer.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix
import numpy as np

# -----------------------------
# Small helpers
# -----------------------------
class AverageMeter:
    def __init__(self):
        self.n = 0
        self.sum = 0.0
    def update(self, val, count):
        self.sum += float(val) * int(count)
        self.n += int(count)
    @property
    def avg(self):
        return self.sum / max(1, self.n)

def make_neighbor_soft_labels(y_int: torch.Tensor, n_classes: int, alpha: float = 0.10) -> torch.Tensor:
    """
    인접한 존(Z-1, Z+1)에 alpha/2씩 분배, 정답 Z에는 1-alpha.
    """
    B = y_int.shape[0]
    y = torch.zeros(B, n_classes, device=y_int.device)
    y[torch.arange(B), y_int] = 1.0 - alpha
    left  = torch.clamp(y_int - 1, min=0)
    right = torch.clamp(y_int + 1, max=n_classes - 1)
    y[torch.arange(B), left]  += alpha / 2.0
    y[torch.arange(B), right] += alpha / 2.0
    return y

# -----------------------------
# Train / Eval
# -----------------------------
def step_one_epoch(model, loader, optimizer, scheduler, device, n_classes: int,
                   train: bool = True, alpha_neighbor: float = 0.10) -> Dict[str, float]:
    model.train(train)
    loss_meter = AverageMeter()
    aux_meter  = AverageMeter()
    acc_meter  = AverageMeter()

    for x, y in loader:
        x = x.to(device)  # (B,T,5)
        y = y.to(device)  # (B,)
        y_soft = make_neighbor_soft_labels(y, n_classes, alpha=alpha_neighbor)

        out = model(x, do_aux=train)
        logits = out["logits"]
        aux = out["aux_mask_loss"]

        # KLDiv with soft targets
        cls_loss = F.kl_div(F.log_softmax(logits, dim=-1), y_soft, reduction="batchmean")
        loss = cls_loss + aux

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = (pred == y).float().mean()

        loss_meter.update(loss.item(), x.size(0))
        aux_meter.update(aux.item() if torch.is_tensor(aux) else float(aux), x.size(0))
        acc_meter.update(acc.item(), x.size(0))

    if train and scheduler is not None:
        scheduler.step()

    return {"loss": loss_meter.avg, "aux_loss": aux_meter.avg, "acc": acc_meter.avg}

def build_optimizer_scheduler(model, lr: float, weight_decay: float, epochs: int, warmup_epochs: int = 0):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    return optimizer, scheduler

@torch.no_grad()
def predict_all(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    ys, ps = [], []
    model.eval()
    for x, y in loader:
        x = x.to(device)
        logits = model(x, do_aux=False)["logits"]
        p = logits.argmax(dim=-1).cpu().numpy()
        ys.append(y.numpy()); ps.append(p)
    y = np.concatenate(ys); p = np.concatenate(ps)
    return y, p

@torch.no_grad()
def evaluate_confusion_matrix(model, loader, device, n_classes: int):
    model.eval()
    all_y, all_p = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x, do_aux=False)["logits"]
        pred = logits.argmax(dim=-1)
        all_y.append(y.cpu().numpy())
        all_p.append(pred.cpu().numpy())
    all_y = np.concatenate(all_y)
    all_p = np.concatenate(all_p)
    cm = confusion_matrix(all_y, all_p, labels=list(range(n_classes)))
    return cm
