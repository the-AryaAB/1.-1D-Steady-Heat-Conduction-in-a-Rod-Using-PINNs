import matplotlib.pyplot as plt

def plot_theta(xi_test_np, theta_exact, theta_pred, abs_err):
    plt.figure()
    plt.plot(xi_test_np, theta_exact, label="Exact")
    plt.plot(xi_test_np, theta_pred, "--", label="PINN")
    plt.legend()
    plt.grid()
    plt.title("θ(x)")

    plt.figure()
    plt.plot(xi_test_np, abs_err)
    plt.title("Absolute Error")
    plt.grid()
    plt.show()
