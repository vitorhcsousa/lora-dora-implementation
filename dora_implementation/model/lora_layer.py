import torch
from torch import nn
import torch.nn.functional as F


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
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(nn.Module):
    """
    Return the x.W + x.A.B, where W is the pretrained weights and A and B are the LoRA matrices.
    """

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class LinearWithLoRAMerged(nn.Module):
    """
    With the distributive law o matrix multiplication (x.(W+A.B)= x.W + x.A.B) we can combine or merge the LoRA
    matrices and original weights.
    Return x.(W+A.B), where W is the pretrained weights and A and B are the LoRA matrices.
    """

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        lora = self.lora.A @ self.lora.B
        combined_weight = self.linear.weight + self.lora.alpha * lora.T

        return F.linear(x, combined_weight, self.linear.bias)
