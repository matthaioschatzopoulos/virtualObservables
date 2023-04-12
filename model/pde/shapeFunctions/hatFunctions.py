import torch
def hat(x, x_node, h):
    # Reshape x and x_node to enable broadcasting
    x = x.unsqueeze(-1)
    x_node = x_node.view(-1, 1, 1)

    # Compute boolean masks
    mask1 = (x >= x_node - h) & (x <= x_node)
    mask2 = (x < x_node - h) | (x > x_node + h)

    # Compute intermediate tensor
    out1 = torch.where(mask1, (x - (x_node - h)) / h, 1 - (x - x_node) / h)

    # Apply final mask and reshape to get output matrix
    return torch.reshape(torch.where(mask2, torch.zeros_like(out1), out1), [x_node.size(dim=0), -1])


def hatGrad(x, x_node, h):
    # Reshape x and x_node to enable broadcasting
    x = x.unsqueeze(-1)
    x_node = x_node.view(-1, 1, 1)

    # Compute boolean masks
    mask1 = (x >= x_node - h) & (x <= x_node)
    mask2 = (x < x_node - h) | (x > x_node + h)

    # Compute intermediate tensor
    out1 = torch.where(mask1, 1 / h, -1 / h)

    # Apply final mask and reshape to get output matrix
    return torch.reshape(torch.where(mask2, torch.zeros_like(out1), out1), [x_node.size(dim=0), -1])