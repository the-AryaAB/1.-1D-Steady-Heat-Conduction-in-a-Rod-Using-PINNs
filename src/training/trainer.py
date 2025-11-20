import torch
from physics.heat1D_PDE import pde_residual
from physics.boundary_conditions import dirichlet_bc_0, dirichlet_bc_1


def compute_loss(model, xi_f, xi_b0, xi_b1):
    r_pde = pde_residual(model, xi_f)
    loss_pde = torch.mean(r_pde ** 2)

    r0 = dirichlet_bc_0(model, xi_b0)
    r1 = dirichlet_bc_1(model, xi_b1)

    loss_bc = 0.5 * (torch.mean(r0**2) + torch.mean(r1**2))

    return loss_pde + loss_bc, loss_pde.detach(), loss_bc.detach()
