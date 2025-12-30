import torch.nn as nn
import argparse
from utils import Logger
import torch
import torch.nn.functional as F


def count_parameters(logger: Logger, model: nn.Module):
    """Print no. of parameters in model, no. of trainable parameters,
     no. of parameters in each dtype."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.write('\nModel details:')
    logger.write('# parameters: ' + str(total))
    logger.write('# trainable parameters: ' + str(trainable) + ', ' +
                 str(100 * trainable / total) + '%')

    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    logger.write('#params by dtype:')
    for k, v in dtypes.items():
        logger.write(str(k) + ': ' + str(v) + ', ' + str(100 * v / total) + '%')


class TimeSeriesModel(nn.Module):
    """
    Beacon 전용 STraTS에서 사용하는 base class.

    - demo_emb: 정적 feature 임베딩 (args.D>0일 때만 사용)
    - ts_demo_dim: time-series embedding + demo embedding의 최종 차원
    - cls_head: 다중 클래스 분류용 Linear layer (num_classes = args.num_classes, 있을 경우)
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args

        # ----- demo embedding (없으면 skip) -----
        if getattr(args, "D", 0) > 0:
            self.demo_emb = nn.Sequential(
                nn.Linear(args.D, args.hid_dim * 2),
                nn.Tanh(),
                nn.Linear(args.hid_dim * 2, args.hid_dim),
            )
            demo_dim = args.hid_dim
        else:
            self.demo_emb = None
            demo_dim = 0

        # time-series representation dimension
        if args.model_type == 'istrats':
            ts_dim = args.hid_dim
        elif args.model_type == 'sand':
            ts_dim = args.hid_dim * args.M
        else:
            ts_dim = args.hid_dim

        self.ts_demo_dim = ts_dim + demo_dim

        # ----- multi-class classifier (supervised에서만 사용) -----
        self.num_classes = getattr(args, "num_classes", None)
        if self.num_classes is not None:
            self.cls_head = nn.Linear(self.ts_demo_dim, self.num_classes)
        else:
            self.cls_head = None

    def classification_loss(self, logits, labels):
        """
        Multi-class cross-entropy.
        - logits: (B, C)
        - labels: LongTensor, shape (B,), 값은 [0, C-1]
        """
        if labels is None:
            # 평가 시에는 logits만 반환하게 사용.
            return logits
        return F.cross_entropy(logits, labels)
