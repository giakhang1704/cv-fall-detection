"""Small sequential classifiers; history lengths distinguish padding from missing pose."""

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class ImageEncoder(nn.Module):
    """Frozen MobileNetV3-small (ImageNet weights) that maps a person crop to a 576-D vector."""
    def __init__(self):
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        """Return the global-average-pooled feature vector of each image in the batch."""
        return self.pool(self.features(x)).flatten(1)


class Classifier(nn.Module):
    """Small classifier over a window of observations: ``cnn2d``, ``pose_lstm``/``cnn_lstm`` or
    ``pose_transformer``. Always outputs logits of the 3 states for the last observation.
    """
    def __init__(self, kind, input_dim, hidden=16, dropout=0.4, transformer_dim=32, heads=2, layers=1):
        super().__init__()
        self.kind = kind
        if kind in ("pose_lstm", "cnn_lstm"):
            self.sequence = nn.LSTM(input_dim, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 3))
        elif kind == "pose_transformer":
            if (
                heads < 1
                or layers < 1
                or transformer_dim < 4
                or transformer_dim % 2
                or transformer_dim % heads
            ):
                raise ValueError("Transformer width must divide by attention heads")
            self.projection = nn.Linear(input_dim, transformer_dim)
            layer = nn.TransformerEncoderLayer(
                transformer_dim, heads, dim_feedforward=2 * transformer_dim, dropout=dropout, batch_first=True
            )
            self.sequence = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.head = nn.Sequential(
                nn.LayerNorm(transformer_dim), nn.Dropout(dropout), nn.Linear(transformer_dim, 3)
            )
        elif kind == "cnn2d":
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_dim, 3))
        else:
            raise ValueError(kind)

    def forward(self, x, lengths=None):
        """Return 3-class logits for the last valid observation of each sequence.

        ``lengths`` separates padding from real history. For the Transformer, padded and
        missing-pose tokens are zeroed and masked out of attention.
        """
        if lengths is None:
            lengths = torch.full((len(x),), x.shape[1], dtype=torch.long, device=x.device)
        if bool((lengths < 1).any()) or bool((lengths > x.shape[1]).any()):
            raise ValueError("Invalid history lengths")
        last = lengths.to(x.device) - 1
        batch = torch.arange(len(x), device=x.device)
        if self.kind == "cnn2d":
            return self.head(x[batch, last])
        if self.kind == "pose_transformer":
            padding = torch.arange(x.shape[1], device=x.device)[None] >= lengths.to(x.device)[:, None]
            missing = x[:, :, -1] < 0.5
            mask = padding | missing
            if bool(mask.all(dim=1).any()):
                raise ValueError("Transformer requires at least one observed token")
            # Erase masked tokens before projection: invalid values must not leak through attention.
            z = self.projection(x.masked_fill(mask[:, :, None], 0.0))
            positions = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]
            frequency = torch.exp(
                torch.arange(0, z.shape[2], 2, device=x.device, dtype=x.dtype)
                * (-math.log(10000.0) / z.shape[2])
            )
            pe = torch.zeros(x.shape[1], z.shape[2], device=x.device, dtype=x.dtype)
            pe[:, 0::2] = torch.sin(positions * frequency)
            pe[:, 1::2] = torch.cos(positions * frequency)
            z = self.sequence(z + pe, src_key_padding_mask=mask)
            return self.head(z[batch, last])
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.sequence(packed)
        return self.head(h[-1])
