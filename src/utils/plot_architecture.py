import os
import yaml
import matplotlib.pyplot as plt


def load_config():
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(os.path.dirname(current_file))  
    config_path = os.path.join(src_dir, "configs", "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_figures_dir():
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(os.path.dirname(current_file)) 
    project_root = os.path.dirname(src_dir)
    figures_dir = os.path.join(project_root, "experiments", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def plot_network_architecture(save=True, filename="network_architecture.png"):
    config = load_config()
    net_cfg = config.get("network", {})
    layers = net_cfg.get("layers", [])
    activation = net_cfg.get("activation", "tanh")

    if not layers:
        raise ValueError("No 'layers' found under 'network' in config.yaml")

    num_layers = len(layers)

    # Figure setup
    fig, ax = plt.subplots(figsize=(max(6, 2 * num_layers), 6))
    ax.set_title("PINN Network Architecture", fontsize=14)

    # Spacing settings
    x_spacing = 6.0
    y_spacing = 0.7

    # Store neuron positions for drawing connections
    neuron_positions = []

    # Determine vertical center to keep things symmetric
    max_neurons = max(layers)

    for layer_idx, n_neurons in enumerate(layers):
        x = layer_idx * x_spacing

        # Compute y positions so the layer is vertically centered
        total_height = (n_neurons - 1) * y_spacing
        y_start = -total_height / 2.0
        layer_positions = []

        for neuron_idx in range(n_neurons):
            y = y_start + neuron_idx * y_spacing
            layer_positions.append((x, y))

            # Draw neuron as a circle
            circle = plt.Circle((x, y), 0.15, edgecolor="black", facecolor="lightblue", zorder=3)
            ax.add_patch(circle)

        neuron_positions.append(layer_positions)

    for layer_idx in range(num_layers - 1):
        layer_curr = neuron_positions[layer_idx]
        layer_next = neuron_positions[layer_idx + 1]

        for (x0, y0) in layer_curr:
            for (x1, y1) in layer_next:
                ax.plot([x0, x1], [y0, y1], "k-", linewidth=0.5, zorder=1)

    for layer_idx, n_neurons in enumerate(layers):
        x = layer_idx * x_spacing
        if layer_idx == 0:
            label = f"Input layer\n({n_neurons})"
        elif layer_idx == num_layers - 1:
            label = f"Output layer\n({n_neurons})"
        else:
            label = f"Hidden layer {layer_idx}\n({n_neurons})"

        ax.text(
            x,
            max_neurons * y_spacing * 0.6,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Activation text
    ax.text(
        0.5,
        -0.1,
        f"Activation (hidden): {activation}",
        ha="center",
        va="top",
        fontsize=10,
        transform=ax.transAxes,
    )

    ax.set_aspect("equal", "box")
    ax.axis("off")
    plt.tight_layout()

    if save:
        figures_dir = get_figures_dir()
        save_path = os.path.join(figures_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Network architecture saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    plot_network_architecture()
