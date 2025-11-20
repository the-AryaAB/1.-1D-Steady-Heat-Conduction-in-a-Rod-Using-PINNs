
---

# **Project 1 — 1D Steady Heat Conduction in a rod Using Physics-Informed Neural Networks (PINNs)**

This repository contains a complete implementation of a **Physics-Informed Neural Network (PINN)** applied to the classical **1D steady-state heat conduction** problem.
The project is part of a larger series of PINNs-based mini-projects aimed at building my research portfolio in **computational mechanics**, **scientific machine learning**.



## **1. Problem Description**

We consider heat conduction in a 1D steady-state, homogeneous rod with no internal heat generation.

> **Assumptions:**
> * constant thermal conductivity: (k)
> * steady state heat transfer
> * no heat generation
> * 1D

From the above assumptions, we can now simplify the Fourier's law of heat transfer to a single equation.The non-dimensionalizied form of the governing equations will be:

$$
\theta(x) = \frac{T(x) - T_0}{T_1 - T_0}
$$

$$
\xi = \frac{x}{L} \in [0,1]
$$

$$
\frac{d^2 \theta}{d\xi^2} = 0
$$

### **Boundary Conditions**

$$
T(0) = T_0, \qquad T(L) = T_1.
$$

as it means:

$$
\theta(0) = 0, \qquad \theta(1) = 1.
$$

The analytical (dimensionless) solution is

$$
\theta_{\text{exact}}(\xi) = \xi
$$

### **Dimensional temperature reconstruction**

Given dimensional boundary conditions $(T(0) = T_0)$ and $(T(L) = T_L)$,

$$
T(x) = T_0 + (T_L - T_0)\,\theta(\xi)
$$


## **2. Why Use PINNs?**

**Physics-Informed Neural Networks (PINNs)** are often used instead of **classical computational fluid dynamics (CFD)** when flexibility, data integration, or differentiability are important. Classical CFD requires meshing, which can struggle with complex or moving geometries, and often becomes computationally expensive for high-dimensional or inverse problems. PINNs, by contrast, embed the governing PDEs directly into the loss function of a **neural network**, allowing them to learn solutions without meshing and to naturally combine sparse or noisy experimental data with physical laws. They are particularly useful for inverse problems—such as discovering unknown parameters or reconstructing fields from limited measurements—because gradients are computed automatically via backpropagation. While PINNs are not yet a full replacement for high-fidelity CFD, they offer a more flexible, mesh-free, and data-compatible framework for many modern physics and engineering applications.

As a conclusion the benefits of utilizing PINNs on this problem can be mentioned as:

* No meshing required
* Automatic differentiation computes derivatives
* Smooth solutions
* Extensible to:

  * Variable conductivity
  * Source terms
  * Nonlinear PDEs
  * Multi-dimensional domains

This simple 1D case is the foundation for future PINN projects (Poisson, Navier–Stokes, etc.).


## **3. Project Structure**

```
Project1-1D-Heat-Conduction-PINN/
│
├── LICENSE.txt
├── README.md
├── requirements.txt
│
├── src/
│   ├── main_project1.py
│   │
│   ├── models/
│   │   └── fully_connected_pinn.py
│   │
│   ├── physics/
│   │   ├── heat1D_PDE.py
│   │   └── boundary_conditions.py
│   │
│   ├── training/
│   │   └── trainer.py
│   │
│   ├── utils/
│   │   ├── plotting.py
│   │   ├── data_sampling.py
│   │   └── config_loader.py
│   │
│   └── configs/
│       └── config.yaml
├── docs/
│   ├── report_project1_heat1D.pdf
│   ├── report_project1_heat1D.tex
│   ├── figures/
│   │   ├── problem_setup.png
│   │   ├── pinn_architecture.png
│   │   ├── results_theta.png
│   │   ├── results_error.png
│   │   
│   └── references.bib
│
└── experiments/
    ├── logs/
    ├── saved_models/
    └── results/
```


## **4. Installation**

### Install dependencies

```bash
pip install -r requirements.txt
```

### Requirements include:

```
torch
numpy
matplotlib
pyyaml
tqdm
scipy
```


## **5. How to Run**

From the project root directory:

```bash
python src/main_project1.py
```

Optional settings such as learning rate, network depth, number of collocation points, etc., can be modified in:

```
src/configs/config.yaml
```

## **6. Results**

### Temperature Profile

A PINN learns the exact linear temperature distribution:

$$
\theta(\xi) = \xi
$$

### Error Analysis

Typical results for default configuration parameters:

* **Relative L2 Error:** ~1e-4
* **Max Absolute Error:** ~1e-4 (after L-BFGS refinement)

Plots generated:

* PINN vs Analytical Solution
* Absolute Error Distribution

These results are stored in:

```
experiments/results/
```


## **7. Theory Summary**

The PINN minimizes a composite loss function:

[
\mathcal{L}
= \underbrace{\frac{1}{N_f}\sum_{i=1}^{N_f} \left( \theta''(\xi_f^{(i)}) \right)^2}_{\text{PDE Loss}}

* \underbrace{
  \frac{1}{2}\left[
  (\theta(0)-0)^2 + (\theta(1)-1)^2
  \right]}_{\text{Boundary Loss}}
  ]

Where:

* Second derivatives computed via **automatic differentiation**
* No data points needed — only physics and boundary conditions


## 🧩 **8. Key PINN Components**

### ✔ Neural Network Architecture

* Fully connected MLP
* Tanh activation
* Xavier initialization

### ✔ Optimizers

* Adam for fast convergence
* L-BFGS for final refinement (common in PINNs)

### ✔ Sampling

* Random interior collocation points
* Exact boundary points
