from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: batch x time x hidden
        weights = torch.softmax(self.attention(x), dim=1)
        pooled = (x * weights).sum(dim=1)
        return pooled


class CNNBiLSTMAttentionV2(nn.Module):
    """
    This architecture must match scripts/08_train_cnn_bilstm_v2_balanced.py.

    Input:
        batch x 1 x 64 x 301

    Output:
        logits for [fluent, stutter]
    """

    def __init__(self, num_classes: int = 2, hidden_size: int = 128, dropout: float = 0.40):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Dropout2d(dropout),
        )

        # Mel bins: 64 -> 32 -> 16 -> 8
        # Channels: 96
        # LSTM input = 96 * 8 = 768
        self.lstm = nn.LSTM(
            input_size=96 * 8,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = AttentionPooling(hidden_size * 2)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: batch x 1 x 64 x 301
        z = self.cnn(x)

        # z: batch x channels x freq x time
        b, c, f, t = z.shape

        # convert to: batch x time x features
        z = z.permute(0, 3, 1, 2).contiguous()
        z = z.view(b, t, c * f)

        z, _ = self.lstm(z)
        z = self.attention(z)

        return self.classifier(z)
