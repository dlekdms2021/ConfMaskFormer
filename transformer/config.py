# config.py
import torch

class Config:
    def __init__(self):
        self.window_size = 10
        self.embedding_dim = 32
        self.n_heads = 8
        self.n_layers = 2
        self.dropout = 0.5
        self.n_beacons = 5
        self.n_classes = 24
        self.batch_size = 256
        self.num_epochs = 300
        self.lr = 1e-3
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 42