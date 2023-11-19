import torch
from typing import Any, Optional, Tuple, Union
import torch.nn as nn
import torch.nn.functional as F

class EEGFeatNet(nn.Module):
    def __init__(self, n_channels, n_features, projection_dim, num_layers=1):
        super(EEGFeatNet, self).__init__()
        self.hidden_size = n_features
        self.num_layers = num_layers

        self.encoder = nn.LSTM(input_size=n_channels, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=projection_dim, bias=False)

    def forward(
        self,
        x
    ):
        _, (h_n, c_n) = self.encoder(x)
        x = h_n[-1]

        x = self.fc(x)
        
        x = F.normalize(x, dim=-1)

        return x
    
class ClassificationHead(nn.Module):
    def __init__(
        self, 
        hidden_size: int,
        num_classes: int = 10,
        ):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self, 
        feats: torch.Tensor,
        ) -> torch.Tensor:
        out = self.classifier(feats)
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