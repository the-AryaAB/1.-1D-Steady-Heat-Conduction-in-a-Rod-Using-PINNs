def dirichlet_bc_0(model, xi_b0):
    return model(xi_b0)

def dirichlet_bc_1(model, xi_b1):
    return model(xi_b1) - 1.0
