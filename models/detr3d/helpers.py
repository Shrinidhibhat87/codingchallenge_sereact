import torch.nn as nn
from functools import partial
import copy


class BatchNormDim1Swap(nn.BatchNorm1d):
    """
    A custom BatchNorm1d layer to handle inputs of shape HW x N x C, used for transformers.
    """
    def forward(self, x):
        """
        Forward pass with reshaping to apply BatchNorm1d.

        Args:
            x (Tensor): The input tensor of shape HW x N x C.

        Returns:
            Tensor: The normalized output tensor of shape HW x N x C.
        """
        hw, n, c = x.shape
        x = x.permute(1, 2, 0)  # Reshape to N x C x HW
        x = super(BatchNormDim1Swap, self).forward(x)
        return x.permute(2, 0, 1)  # Reshape back to HW x N x C


# Dictionary for normalization functions
NORM_DICT = {
    "bn": BatchNormDim1Swap,
    "bn1d": nn.BatchNorm1d,
    "id": nn.Identity,
    "ln": nn.LayerNorm,
}

# Dictionary for activation functions
ACTIVATION_DICT = {
    "relu": partial(nn.ReLU, inplace=False),
    "gelu": nn.GELU,
    "leakyrelu": partial(nn.LeakyReLU, negative_slope=0.1),
}

# Dictionary for weight initialization methods
WEIGHT_INIT_DICT = {
    "xavier_uniform": nn.init.xavier_uniform_,
}


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
