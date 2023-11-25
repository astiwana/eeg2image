import torch
from typing import Any, Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F

class EEGFeatNet(nn.Module):
    def __init__(self, n_channels, n_features, projection_dim, num_layers=1, bidirectional=True):
        super(EEGFeatNet, self).__init__()
        self.hidden_size = n_features
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            n_channels, 
            self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True,
            bidirectional=bidirectional
            )
        
        n_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(self.hidden_size * n_directions, projection_dim)

    def forward(
        self,
        x
    ):
        _, (h_n, c_n) = self.encoder(x)
        x = h_n[-1]

        x = self.fc(x)
        
        x = F.normalize(x, dim=-1)

        return x
    
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
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(
        self, 
        feats: torch.Tensor,
        ) -> torch.Tensor:
        out = self.fc1(feats)
        out = self.relu(out)
        out = self.fc2(feats)
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