import torch

def trapzInt2D(f_xy):
    out = torch.trapezoid(f_xy, torch.linspace(0, 1, f_xy.size(dim=0)), dim=0)
    return torch.trapezoid(out, torch.linspace(0, 1, f_xy.size(dim=1)), dim=0)