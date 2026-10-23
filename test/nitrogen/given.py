"""Given concentrations varying between elements and steps, importable by both environments (the Root_BRIDGES run
pickles them by reference): numpy only."""
import numpy as np


def varying_C(step, g):
    """C_hexose_root from 1e-5 to 1e-3 mol.g-1."""
    return {v: 1e-5 + 1e-3 * (0.5 + 0.5 * np.sin(0.7 * v + 0.3 * step)) for v in g.vertices(scale=1)}


def varying_AA(step, g):
    """AA from 1e-6 to 5e-4 mol.g-1, out of phase with the hexose: N limits here, C there."""
    return {v: 1e-6 + 5e-4 * (0.5 + 0.5 * np.cos(1.3 * v + 0.2 * step)) for v in g.vertices(scale=1)}
