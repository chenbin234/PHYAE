import torch


class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        mask = (input.abs() <= 1).float()
        grad_input = grad_output * mask
        return grad_input
