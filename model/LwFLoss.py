import torch
import torch.nn as nn
import torch.nn.functional as F


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=True):
        super(_WeightedLoss, self).__init__(size_average)
        self.register_buffer('weight', weight)


##############################################
# Generate the responses of the original model
class LwFLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=True, tau=2):
        super(LwFLoss, self).__init__(weight, size_average)
        self.tau = tau

    def forward(self, input, target):
        _assert_no_grad(target)

        output = -F.softmax(input / self.tau, dim=1) * F.log_softmax(target / self.tau, dim=1)

        # output = -F.softmax(target / self.tau, dim=1) * F.log_softmax(input / self.tau, dim=1)

        # soft_input = torch.pow(F.softmax(input, dim=1), 1/self.tau)
        # soft_input = soft_input / soft_input.sum(1).view(soft_input.size(0), 1).expand_as(soft_input)
        # soft_input.clamp(min=1e-8)  # Block the Explosion of Gradients.

        # soft_target = torch.pow(F.softmax(target, dim=1), 1/self.tau)
        # soft_target = soft_target / soft_target.sum(1).view(soft_target.size(0), 1).expand_as(soft_target)
        #
        # output = -soft_target * torch.log(soft_input)

        if self.size_average:
            # output = torch.mean(output)  # 0.8162 / 0.5336
            output = torch.sum(output, 1)
            output = torch.mean(output)
        else:
            output = torch.sum(output)
        return output
