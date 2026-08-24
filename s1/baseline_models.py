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
