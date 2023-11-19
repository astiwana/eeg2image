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
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        device = self.parameters()
        x = pixel_values
        h_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        _, (h_n, c_n) = self.encoder(x, (h_n, c_n))
        x = h_n[-1]

        x = F.normalize(x, dim=-1)

        return x
    
model = EEGFeatNet(1, 1, 1 ,1)
model()