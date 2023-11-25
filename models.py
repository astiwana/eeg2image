import torch
from typing import Any, Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F

class EEGFeatNet(nn.Module):
    def __init__(self, n_channels, hidden_size, num_layers=3, dropout=0.1) -> None:
        super().__init__()

        self.bilstm = nn.LSTM(
            n_channels,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

    def forward(self, x):
        out, _ = self.bilstm(x)

        return out.reshape(out.shape[0], -1)
    
class EEGCNN(nn.Module):
    def __init__(self, in_channels, out1, out2, kernel_size=3) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out1, kernel_size)
        self.conv2 = nn.Conv2d(out1, out2, kernel_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)

        return x.view(x.shape[0], -1)
    
class ClassificationHead(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        num_classes: int = 10,
        ):
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(
        self, 
        feats: torch.Tensor,
        ) -> torch.Tensor:
        out = self.fc(feats)

        return out

if __name__ == "__main__":
    n_epochs = 100
    batch_size = 256
    n_features = 128
    projection_dim = 128
    weight_decay = 1e-4
    lr = 1e-3

    n_channels = 5
    seq_len = 200
    n_classes = 10

    vis_freq = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    device = "cpu"
    eeg = torch.randn((batch_size, seq_len, n_channels)).to(device)
    model = EEGFeatNet(n_channels=n_channels, n_features=n_features, projection_dim=projection_dim).to(device)