import os

def save_plot(fig, filename):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(this_dir))
    figures_dir = os.path.join(root, "experiments", "figures")
    
    save_path = os.path.join(figures_dir, filename)
    fig.savefig(save_path, bbox_inches='tight')
    