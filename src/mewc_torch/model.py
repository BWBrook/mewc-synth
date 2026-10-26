"""The MEWC classifier on a timm backbone: average-pooled features, dropout, a compression layer sized by
mewc-train's rule, ELU, dropout and the class layer. Blocks are unfrozen from the top by timm's coarse
parameter groups; normalisation layers with running statistics stay frozen, as in mewc-train."""
import timm
import torch
import torch.nn as nn
from timm.models import group_parameters


def compression_width(n_features, n_classes, vit):
    if vit:
        return 128 if n_features <= 384 else 256 if n_classes <= 25 else 384 if n_features == 768 else 512
    return 256 if n_classes <= 25 or n_features < 768 else 384 if n_features == 768 else 512


class MewcClassifier(nn.Module):
    def __init__(self, backbone, n_classes, dropout=0.1, pretrained=True):
        super().__init__()
        self.base = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        n = self.base.num_features
        width = compression_width(n, n_classes, vit=hasattr(self.base, "blocks") and hasattr(self.base, "patch_embed"))
        self.drop1, self.compress, self.act = nn.Dropout(dropout), nn.Linear(n, width), nn.ELU()
        self.drop2, self.classify = nn.Dropout(dropout), nn.Linear(width, n_classes)
        cfg = timm.data.resolve_data_config({}, model=self.base)
        self.mean, self.std, self.native_size = cfg["mean"], cfg["std"], cfg["input_size"][-1]

    def set_dropout(self, rate):
        self.drop1.p = self.drop2.p = rate

    def forward(self, x):
        return self.classify(self.drop2(self.act(self.compress(self.drop1(self.base(x))))))

    def head_parameters(self):
        return [p for m in (self.compress, self.classify) for p in m.parameters()]


def block_groups(base):
    """Backbone parameter names in coarse groups, bottom to top (stem first, final norm last)."""
    groups = group_parameters(base, base.group_matcher(coarse=True))
    return [names for _, names in sorted(groups.items())]


BODY = {"blocks", "stages", "layers", "levels"}


def freeze_base(model, blocks):
    """blocks = 0 freezes the backbone, -1 trains all of it, n > 0 trains the top n body groups (stages or
    transformer blocks) and every group above them (final norms and head convolutions)."""
    names = block_groups(model.base)
    trainable = set()
    if blocks < 0:
        trainable = {n for g in names for n in g}
    elif blocks > 0:
        body = [i for i, g in enumerate(names) if any(n.split(".")[0] in BODY for n in g)]
        start = body[-blocks] if len(body) >= blocks else (body[0] if body else 0)
        trainable = {n for g in names[start:] for n in g}
    for name, p in model.base.named_parameters():
        p.requires_grad = name in trainable
    return sum(p.numel() for p in model.base.parameters() if p.requires_grad)


def frozen_norms(model):
    """BatchNorm-type layers stay in inference mode throughout, as mewc-train keeps them untrainable."""
    for m in model.base.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


class Deployable(nn.Module):
    """For export: takes raw RGB in 0..255 at the training size, returns class probabilities."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(model.mean).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(model.std).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        return torch.softmax(self.model((x - self.mean) / self.std), dim=1)
