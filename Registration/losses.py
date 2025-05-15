import torch

class Grad:
    """
    N-D gradient loss.
    """
    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    @staticmethod
    def dvf_diff(input, dim):
        if dim == 0:
            diff = input[1:, :, :, :, :] - input[:-1, :, :, :, :]
        elif dim == 1:
            diff = input[:, 1:, :, :, :] - input[:, :-1, :, :, :]
        elif dim == 2:
            diff = input[:, :, 1:, :, :] - input[:, :, :-1, :, :]
        elif dim == 3:
            diff = input[:, :, :, 1:, :] - input[:, :, :, :-1, :]
        elif dim == 4:
            diff = input[:, :, :, :, 1:] - input[:, :, :, :, :-1]
        return diff

    def forward(self, dvf):
        dz = torch.abs(self.dvf_diff(dvf, dim=2))
        dy = torch.abs(self.dvf_diff(dvf, dim=3))
        dx = torch.abs(self.dvf_diff(dvf, dim=4))

        if self.penalty == 'l2':
                dy = dy * dy
                dx = dx * dx
                dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0
        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad