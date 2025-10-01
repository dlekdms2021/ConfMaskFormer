# model.py: Transformer 기반 위치 분류 모델

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, n_beacons, n_classes, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 항상 2-tensor (x: (B, window, n_beacons)) 형식 사용
        self.embedding = nn.Linear(n_beacons, cfg.embedding_dim)
        self.pos_encoder = PositionalEncoding(cfg.embedding_dim, max_len=cfg.window_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embedding_dim,
            nhead=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cfg.embedding_dim, n_classes)

    def forward(self, x, mask=None):
        # x: (B, window, n_beacons)
        x = self.embedding(x)
        x = self.pos_encoder(x)

        src_mask = None
        src_key_padding_mask = (mask == 0) if mask is not None else None

        x = self.transformer(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        x = x.transpose(1, 2)  # for pooling
        x = self.pool(x).squeeze(-1)
        out = self.fc(x)
        return out
