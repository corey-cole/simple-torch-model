import torch
import torch.nn as nn

"""
The purpose of this model is to provide a somewhat simple model for experimenting with different
export modalities (e.g. TorchScript, torch.export, and torch.onnx.export)
"""


# NOTE: The conditional_forward method (and the associated functions) are being left in this model
# such that it could eventually have unit tests that confirm that the simple model returns similar
# results to the ConditionalModel when given the same input.
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def gt_zero_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def le_zero_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    @torch.jit.ignore
    def conditional_forward(self, x: torch.Tensor) -> torch.Tensor:
        """This implementation of forward uses torch.cond for compatibility with torch.export API"""
        return torch.cond(x.sum() > 0, self.gt_zero_fn, self.le_zero_fn, (x,))

    # This doesn't work with export API
    def forward(self, x):
        if x.sum() > 0:
            return self.fc(x)
        else:
            return torch.zeros_like(x)


class ConditionalModel(nn.Module):
    def __init__(self):
        super(ConditionalModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def gt_zero_fn(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def le_zero_fn(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This implementation of forward uses torch.cond for compatibility with torch.export API"""
        return torch.cond(x.sum() > 0, self.gt_zero_fn, self.le_zero_fn, (x,))
