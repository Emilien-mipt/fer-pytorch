import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG

from .bi_tempered_helper import bi_tempered_logistic_loss

sys.path.insert(0, "..")


# ====================================================
# Criterion - ['LabelSmoothing', 'FocalLoss' 'FocalCosineLoss', 'SymmetricCrossEntropyLoss',
#              'BiTemperedLoss', 'TaylorCrossEntropyLoss']
# ====================================================


def get_criterion():
    if CFG.criterion == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    elif CFG.criterion == "LabelSmoothing":
        criterion = LabelSmoothingLoss()
    elif CFG.criterion == "Bi-TemperedLoss":
        criterion = BiTemperedLogisticLoss()
    elif CFG.criterion == "FocalLoss":
        criterion = FocalLoss()
    elif CFG.criterion == "FocalCosineLoss":
        criterion = FocalCosineLoss()
    elif CFG.criterion == "SymmetricCrossEntropyLoss":
        criterion = SymmetricCrossEntropy()
    return criterion


# ====================================================
# Label Smoothing
# ====================================================
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=CFG.target_size, smoothing=CFG.smooth_alpha, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# ====================================================
# Bi-Tempered Loss
# ====================================================
class BiTemperedLogisticLoss(nn.Module):
    def __init__(self, t1=CFG.T1, t2=CFG.T2, smoothing=0.0):
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.smoothing = smoothing

    def forward(self, logit_label, truth_label):
        loss_label = bi_tempered_logistic_loss(
            logit_label, truth_label, t1=self.t1, t2=self.t2, label_smoothing=self.smoothing, reduction="none"
        )

        loss_label = loss_label.mean()
        return loss_label


# ====================================================
# Focal Loss
# ====================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=CFG.gamma, reduce=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


# ====================================================
# Focal Cosine Loss
# ====================================================
class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=CFG.gamma, xent=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.xent = xent

        self.y = torch.Tensor([1]).cuda(f"cuda:{CFG.GPU_ID}")

    def forward(self, input, target, reduction="mean"):
        cosine_loss = F.cosine_embedding_loss(
            input, F.one_hot(target, num_classes=input.size(-1)), self.y, reduction=reduction
        )

        cent_loss = F.cross_entropy(F.normalize(input), target, reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss

        if reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss


# ====================================================
#  Symmetric Cross-Entropy Loss
# ====================================================
class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=CFG.alpha, beta=CFG.beta, num_classes=CFG.target_size):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits, targets, reduction="mean"):
        onehot_targets = torch.eye(self.num_classes)[targets].cuda(f"cuda:{CFG.GPU_ID}")
        ce_loss = F.cross_entropy(logits, targets, reduction=reduction)
        rce_loss = (-onehot_targets * logits.softmax(1).clamp(1e-7, 1.0).log()).sum(1)
        if reduction == "mean":
            rce_loss = rce_loss.mean()
        elif reduction == "sum":
            rce_loss = rce_loss.sum()
        return self.alpha * ce_loss + self.beta * rce_loss
