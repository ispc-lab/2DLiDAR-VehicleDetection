import torch


def sine_delta_mse_loss(input, target, beta=5, size_average=True):
    sine_delta = torch.sin(input - target)
    loss = beta * sine_delta ** 2
    if size_average:
        return loss.mean()
    return loss.sum()
