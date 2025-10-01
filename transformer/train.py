# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TransformerClassifier
from dataset import BeaconDataset
from utils import set_seed
from sklearn.metrics import classification_report
from tqdm import tqdm
import time


def train_and_evaluate(cfg, train_path, test_path, pos):
    set_seed(cfg.seed)

    train_loader = DataLoader(BeaconDataset(train_path), batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(BeaconDataset(test_path), batch_size=cfg.batch_size)

    model = TransformerClassifier(cfg.n_beacons, cfg.n_classes, cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_preds, best_labels = 0.0, [], []
    start_time = time.time()

    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training", leave=False):
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        train_acc = correct / total

        model.eval()
        correct, total = 0, 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc=f"[Epoch {epoch+1}] Testing", leave=False):
                x, y = x.to(cfg.device), y.to(cfg.device)
                out = model(x)
                pred = out.argmax(dim=1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())
                correct += (pred == y).sum().item()
                total += y.size(0)
        test_acc = correct / total

        if test_acc > best_acc:
            best_acc = test_acc
            best_preds = y_pred
            best_labels = y_true
            best_model = model.state_dict()

    report = classification_report(best_labels, best_preds, output_dict=True, zero_division=0)
    return {
        "acc": best_acc,
        "labels": best_labels,
        "preds": best_preds,
        "report": report,
        "state_dict": best_model
    }
