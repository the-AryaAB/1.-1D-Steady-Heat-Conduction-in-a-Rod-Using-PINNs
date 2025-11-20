import os
import numpy as np
import torch

from models.fully_connected_pinn import FullyConnectedPINN
from training.trainer import compute_loss
from utils.data_sampling import sample_interior_points, boundary_points
from utils.plotting import plot_theta
from utils.config_loader import load_config


def main():
    # 1. Load configuration
    # Build path to config.yaml relative to this file
    this_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(this_dir, "configs", "config.yaml")

    config = load_config(config_path)

    # Problem parameters
    L = config["problem"]["L"]
    T0 = config["problem"]["T0"]
    TL = config["problem"]["TL"]
    N_f = config["problem"]["N_f"]

    # Network + training parameters
    layers = config["network"]["layers"]          # e.g. [1, 20, 20, 20, 1]
    adam_lr = config["training"]["adam_lr"]
    adam_iters = config["training"]["adam_iters"]
    use_lbfgs = config["training"]["use_lbfgs"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # 2. Sample collocation and boundary points
    # ---------------------------------------------------------
    xi_f = sample_interior_points(N_f, device)   # interior points (0,1)
    xi_b0, xi_b1 = boundary_points(device)       # boundary xi = 0, 1

    # ---------------------------------------------------------
    # 3. Build PINN model
    # ---------------------------------------------------------
    model = FullyConnectedPINN(layers).to(device)
    print(model)

    # ---------------------------------------------------------
    # 4. Train with Adam
    # ---------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)

    print("Training with Adam...")
    for it in range(adam_iters):
        optimizer.zero_grad()
        total_loss, loss_pde, loss_bc = compute_loss(model, xi_f, xi_b0, xi_b1)
        total_loss.backward()
        optimizer.step()

        if it % max(1, adam_iters // 10) == 0:
            print(
                f"[Adam] Iter {it:6d} | "
                f"Total = {total_loss.item():.3e} | "
                f"PDE = {loss_pde.item():.3e} | "
                f"BC = {loss_bc.item():.3e}"
            )

    # ---------------------------------------------------------
    # 5. (Optional) refine with L-BFGS
    # ---------------------------------------------------------
    if use_lbfgs:
        print("Refining with L-BFGS...")

        def closure():
            optimizer_lbfgs.zero_grad()
            total_loss, _, _ = compute_loss(model, xi_f, xi_b0, xi_b1)
            total_loss.backward()
            return total_loss

        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=500,
            max_eval=500,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        optimizer_lbfgs.step(closure)

        final_loss, _, _ = compute_loss(model, xi_f, xi_b0, xi_b1)
        print(f"Final loss after L-BFGS: {final_loss.item():.3e}")

    # ---------------------------------------------------------
    # 6. Evaluation on a test grid
    # ---------------------------------------------------------
    model.eval()

    # Test points in [0,1]
    xi_test_np = np.linspace(0.0, 1.0, 200).reshape(-1, 1)
    xi_test = torch.tensor(xi_test_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        theta_pred = model(xi_test).cpu().numpy()

    # Analytical dimensionless solution: theta_exact(xi) = xi
    theta_exact = xi_test_np

    # Dimensional temperature (if needed)
    T_pred = T0 + (TL - T0) * theta_pred
    T_exact = T0 + (TL - T0) * theta_exact

    # Error (dimensionless)
    abs_err = np.abs(theta_pred - theta_exact)
    l2_rel_err = np.sqrt(np.mean(abs_err ** 2)) / np.sqrt(np.mean(theta_exact ** 2))
    max_err = np.max(abs_err)

    print(f"Relative L2 error (theta): {l2_rel_err:.3e}")
    print(f"Max abs error (theta):    {max_err:.3e}")

    # ---------------------------------------------------------
    # 7. Save results (optional but recommended)
    # ---------------------------------------------------------
    root_dir = os.path.dirname(this_dir)  # go one level up: project root
    exp_dir = os.path.join(root_dir, "experiments")
    models_dir = os.path.join(exp_dir, "saved_models")
    results_dir = os.path.join(exp_dir, "results")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Save model weights and results
    torch.save(model.state_dict(), os.path.join(models_dir, "heat1d_pinn.pt"))
    np.save(os.path.join(results_dir, "theta_pred.npy"), theta_pred)
    np.save(os.path.join(results_dir, "theta_exact.npy"), theta_exact)
    np.save(os.path.join(results_dir, "T_pred.npy"), T_pred)
    np.save(os.path.join(results_dir, "T_exact.npy"), T_exact)

    print(f"Saved model to: {os.path.join(models_dir, 'heat1d_pinn.pt')}")

    # ---------------------------------------------------------
    # 8. Plot results
    # ---------------------------------------------------------
    plot_theta(xi_test_np, theta_exact, theta_pred, abs_err)


if __name__ == "__main__":
    main()
