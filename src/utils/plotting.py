import numpy as np
from utils.save_figures import save_plot
import matplotlib.pyplot as plt

def plot_theta(xi_test_np, theta_exact, theta_pred, abs_err,
               L=1.0, T_ref=0.0, T_scale=1.0, x_start=0.0):
    
    # map dimensionless position to physical coordinate
    x = x_start + xi_test_np * L

    # convert to dimensional temperatures
    T_exact = T_ref + theta_exact * T_scale
    T_pred = T_ref + theta_pred * T_scale
    abs_err_T = np.abs(T_exact - T_pred)

    # dimensionless comparison
    fig1 = plt.figure()
    plt.plot(xi_test_np, theta_exact, label="Exact (dimensionless)")
    plt.plot(xi_test_np, theta_pred, "--", label="PINN (dimensionless)")
    plt.legend()
    plt.xlabel("Dimensionless position (ξ)")
    plt.ylabel("Dimensionless temperature (θ(ξ))")
    plt.grid()
    plt.title("Exact (analytical) vs. approximate (PINNs) — dimensionless")
    save_plot(fig1, "theta_comparison_dimensionless.png")

    # dimensionless absolute error
    fig2 = plt.figure()
    plt.plot(xi_test_np, abs_err)
    plt.title("Absolute Error |θ_exact - θ_pred| (dimensionless)")
    plt.xlabel("Dimensionless position (ξ)")
    plt.ylabel("Absolute Error")
    plt.grid()
    save_plot(fig2, "theta_absolute_error_dimensionless.png")

    # dimensional temperature distribution along the rod
    fig3 = plt.figure()
    plt.plot(x, T_exact, label="Exact (dimensional)")
    plt.plot(x, T_pred, "--", label="PINN (dimensional)")
    plt.legend()
    plt.xlabel("Position x (m)")
    plt.ylabel("Temperature (°C)")
    plt.grid()
    plt.title(f"Dimensional temperature distribution along rod (L={L})")
    save_plot(fig3, "temperature_distribution_dimensional.png")

    # dimensional absolute error
    fig4 = plt.figure()
    plt.plot(x, abs_err_T)
    plt.title("Absolute Temperature Error |T_exact - T_pred|")
    plt.xlabel("Position x (m)")
    plt.ylabel("Absolute Temperature Error (°C)")
    plt.grid()
    save_plot(fig4, "temperature_absolute_error_dimensional.png")

    plt.show()
