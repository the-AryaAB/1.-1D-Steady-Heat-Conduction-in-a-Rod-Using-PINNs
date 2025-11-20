import torch


def pde_residual(model, xi):
    xi = xi.requires_grad_(True)
    theta = model(xi)
    theta_xi = torch.autograd.grad(
        theta, xi,
        grad_outputs=torch.ones_like(theta),
        create_graph=True,
        retain_graph=True
    )[0]
    theta_xixi = torch.autograd.grad(
        theta_xi, xi,
        grad_outputs=torch.ones_like(theta_xi),
        create_graph=True,
        retain_graph=True
    )[0]
    return theta_xixi
