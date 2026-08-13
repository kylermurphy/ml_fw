import numpy as np
from perturbed_input import generate_perturbations, plot_ensemble

rng = np.random.default_rng(42)
n = 200
t = np.arange(n)

y = np.sin(2 * np.pi * t / 40) + 0.3 * rng.normal(size=n)

ensemble = generate_perturbations(
    y,
    n_ensemble=30,
    method="auto",
    seed=42,
)

ax = plot_ensemble(
    x=t,
    y=y,
    ensemble=ensemble,
    n_show=30,
    plot_mean=True,
    plot_median=True,
    colormap="plasma",
)
