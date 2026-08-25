"""Lightweight baseline backbones for the CMANAS-FER revision (EXP-020).

Reviewers (R1.4, R3, R4.Q2) asked for a fair, direct comparison against well-known
lightweight backbones evaluated under the *exact same* LOSO protocol and metrics as the
NAS-discovered architecture. To guarantee an apples-to-apples comparison, these baselines
are driven by the SAME training loop (`train_baseline_jaffe.py`) and the SAME evaluation /
metric code (`voting_baseline.py`) as the NAS model — only the architecture changes.

`BaselineWrapper` makes any torchvision backbone drop-in compatible with the NAS pipeline:
  * forward(x) returns a (logits, None) tuple, matching NetworkCIFAR's (logits, logits_aux);
  * it accepts a `drop_path_prob` attribute (set each epoch by the training loop) and ignores it.
"""

import torch.nn as nn
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# MobileFaceNet (Chen et al., 2018) — ~1M-param face-specific lightweight net.
# Reviewer R3 named it explicitly. Not in torchvision, so implemented here with
# the canonical bottleneck setting + an adaptive-pooling classification head so it
# runs at the paper's 48x48 input (native MobileFaceNet expects 112x112).
# ---------------------------------------------------------------------------
def _conv_bn_prelu(inp, oup, k=3, s=1, p=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, k, s, p, groups=groups, bias=False),
        nn.BatchNorm2d(oup),
        nn.PReLU(oup),
    )


class _Bottleneck(nn.Module):
    """Inverted residual with PReLU (MobileFaceNet style)."""
    def __init__(self, inp, oup, stride, expansion):
        super().__init__()
        hidden = inp * expansion
        self.use_res = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            _conv_bn_prelu(inp, hidden, k=1, s=1, p=0),                 # expand 1x1
            _conv_bn_prelu(hidden, hidden, k=3, s=stride, p=1, groups=hidden),  # dw 3x3
            nn.Conv2d(hidden, oup, 1, 1, 0, bias=False),               # project 1x1 (linear)
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class MobileFaceNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # (expansion t, out c, num n, stride s) — canonical MobileFaceNet
        setting = [(2, 64, 5, 2), (4, 128, 1, 2), (2, 128, 6, 1), (4, 128, 1, 2), (2, 128, 2, 1)]
        layers = [_conv_bn_prelu(3, 64, k=3, s=2, p=1),                # stem
                  _conv_bn_prelu(64, 64, k=3, s=1, p=1, groups=64)]    # depthwise
        inp = 64
        for t, c, n, s in setting:
            for i in range(n):
                layers.append(_Bottleneck(inp, c, s if i == 0 else 1, t))
                inp = c
        layers.append(_conv_bn_prelu(inp, 512, k=1, s=1, p=0))         # 1x1 -> 512
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)                            # adaptive: any input size
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# name -> (constructor, weights-enum-name). Weights enums used when available
# (torchvision >= 0.13); we fall back to pretrained=True on older versions.
_SUPPORTED = {
    "mobilenet_v3_small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights"),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
    "resnet18": ("resnet18", "ResNet18_Weights"),
}


def _replace_classifier(name, model, num_classes):
    """Swap the final classification layer to `num_classes` outputs."""
    if name in ("mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"):
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
    elif name == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported baseline backbone: {name}")
    return model


def build_backbone(name, num_classes, pretrained=True):
    name = name.lower()
    if name == "mobilefacenet":
        # No ImageNet weights exist for MobileFaceNet -> trained from scratch (noted in FINDINGS).
        return MobileFaceNet(num_classes=num_classes)
    if name not in _SUPPORTED:
        raise ValueError(
            f"Unknown backbone '{name}'. Supported: {sorted(_SUPPORTED)}"
        )
    ctor_name, weights_enum = _SUPPORTED[name]
    ctor = getattr(tvm, ctor_name)
    try:
        # New torchvision API (>=0.13): weights=<Enum>.IMAGENET1K_V1
        weights = None
        if pretrained:
            weights = getattr(getattr(tvm, weights_enum), "IMAGENET1K_V1")
        model = ctor(weights=weights)
    except (AttributeError, TypeError):
        # Legacy API: pretrained=True
        model = ctor(pretrained=pretrained)
    return _replace_classifier(name, model, num_classes)


class BaselineWrapper(nn.Module):
    """Adapts a torchvision backbone to the NAS pipeline interface.

    forward returns (logits, None) so `logits, logits_aux = model(x)` keeps working, and
    a `drop_path_prob` attribute can be set each epoch by the training loop (ignored here).
    """

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.drop_path_prob = 0.0  # accepted for interface parity; not used

    def forward(self, x):
        return self.net(x), None


def build_baseline(name, num_classes, pretrained=True):
    return BaselineWrapper(build_backbone(name, num_classes, pretrained))
