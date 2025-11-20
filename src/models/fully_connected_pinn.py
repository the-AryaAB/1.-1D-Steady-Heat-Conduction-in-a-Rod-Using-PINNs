import torch
import torch.nn as nn


class FullyConnectedPINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i + 1]))
        self.layers = nn.ModuleList(layer_list)

        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        z = x
        for i in range(len(self.layers) - 1):
            z = self.layers[i](z)
            z = self.activation(z)
        z = self.layers[-1](z)
        return z
