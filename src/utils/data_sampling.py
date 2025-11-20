import numpy as np
import torch


def sample_interior_points(N_f, device):
    xi_f_np = np.random.rand(N_f, 1)
    return torch.tensor(xi_f_np, dtype=torch.float32).to(device)

def boundary_points(device):
    xi_b0 = torch.tensor([[0.0]], dtype=torch.float32).to(device)
    xi_b1 = torch.tensor([[1.0]], dtype=torch.float32).to(device)
    return xi_b0, xi_b1
