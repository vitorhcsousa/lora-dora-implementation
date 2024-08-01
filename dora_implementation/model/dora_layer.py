from torch import nn
from dora_implementation.model.lora_layer import LoRALayer
import torch
import torch.nn.functional as F


class LinearWithDoRA(nn.Module):
    """
    A custom PyTorch neural network module that integrates a linear layer with a LoRA (Low-Rank Adaptation) layer.
    This class modifies the output of the linear layer using the LoRA layer's output in a specific manner.

    Attributes:
        linear (nn.Module): The pre-defined linear layer.
        lora (LoRALayer): The LoRA layer instance.
        m (nn.Parameter): A parameter initialized to ones with the same output features as the linear layer.
    """

    def __init__(self, linear: nn.Module, rank: int, alpha: float) -> None:
        """
        Initializes the LinearWithDoRA class with the given parameters.

        Args:
            linear (nn.Module): The pre-defined linear layer.
            rank (int): The rank parameter for the LoRA layer.
            alpha (float): The alpha parameter for the LoRA layer.
        """
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
        self.m = nn.Parameter(torch.ones(1, linear.out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the modified linear transformation.
        """
        # Compute the output of the linear layer
        linear_output = self.linear(x)

        # Compute the output of the LoRA layer
        lora_output = self.lora(x)

        # Normalize the LoRA output
        lora_output_norm = lora_output / (
            lora_output.norm(p=2, dim=1, keepdim=True) + 1e-9
        )

        # Apply the DoRA modification
        dora_modification = self.m * lora_output_norm

        # Return the sum of the linear output and the DoRA modification
        return linear_output + dora_modification


class LinearWithDoRAMerged(nn.Module):
    """
    A custom PyTorch neural network module that integrates a linear layer with a LoRA (Low-Rank Adaptation) layer.
    This class modifies the weights of the linear layer using the LoRA layer's output in a specific manner.

    Attributes:
        linear (nn.Module): The pre-defined linear layer.
        lora (LoRALayer): The LoRA layer instance.
        m (nn.Parameter): A parameter initialized to the norm of the linear layer's weights.
    """

    def __init__(self, linear: nn.Module, rank: int, alpha: float) -> None:
        """
        Initializes the LinearWithDoRAMerged class with the given parameters.

        Args:
            linear (nn.Module): The pre-defined linear layer.
            rank (int): The rank parameter for the LoRA layer.
            alpha (float): The alpha parameter for the LoRA layer.
        """
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
        self.m = nn.Parameter(self.linear.weight.norm(p=2, dim=0, keepdim=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the modified linear transformation.
        """
        # Compute the product of the LoRA layer's matrices A and B
        lora = self.lora.A @ self.lora.B

        # Calculate the numerator by adding the linear layer's weights to the scaled LoRA output
        numerator = self.linear.weight + self.lora.alpha * lora.T

        # Compute the denominator as the norm of the numerator
        denominator = numerator.norm(p=2, dim=0, keepdim=True)

        # Obtain the directional component by normalizing the numerator with the denominator
        directional_component = numerator / denominator

        # Calculate the new weights for the linear layer by scaling the directional component with the parameter m
        new_weight = self.m * directional_component

        # Apply the linear transformation using the new weights
        return F.linear(x, new_weight, self.linear.bias)
