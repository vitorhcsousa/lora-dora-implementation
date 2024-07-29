from torch import nn


class MultiLayerPerceptron(nn.Module):
    """
    Multi Layer Perceptron.
    """

    def __init__(self, num_features, num_hidden_1, num_hidden_2, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, num_hidden_1),
            nn.ReLU(),
            nn.Linear(num_hidden_1, num_hidden_2),
            nn.ReLU(),
            nn.Linear(num_hidden_2, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the layers of the model.

        Args:
            x: The input tensor to the model. The shape and type of the tensor
               should be compatible with the layers of the model.

        Returns:
            The output tensor after applying the model's layers to the input tensor.
        """
        x = self.layers(x)
        return x


def freeze_linear_layers(model: nn.Module):
    """
    Logic to freeze params on Linear Layers, recursively.
    Args:
        model: nn.Module, can be the model or child modules.

    Returns:
        The model with the linear params frozen.

    """
    for child in model.children():
        if isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad = False
        else:
            # recursively freeze linear layers in children modules
            freeze_linear_layers(child)
