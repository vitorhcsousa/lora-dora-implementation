import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        """

        Args:
            in_dim: The input dimension
            out_dim: The output dimension
            rank: The inner dimension of the matrices A and B
            alpha: The impact of the low-rank adaption on the layers output.
        """
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())

        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpa = alpha

    def forward(self, x):
        x = self.alpa * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(nn.Module):

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora
