"""
A population in one MPG against the legacy plant by plant: the legacy grows one plant per model (its MTG, its
loops), so N plants cost N runs; the port grows the N plants together, their elements in the same arrays, each step
one call (test/test_population.py: each plant identical to the legacy grown alone).

    left     the time of the run against the number of plants, legacy (N runs) and port (one population)
    right    the time per plant

The plants differ by their scenario (random_choice: their own draws, so their own root systems). Run from
rhizodep/example/comparison:

    python profile_population.py [--plants 1 2 4 8 16 32] [--days D] [--set NAME]     figures in ./outputs/profiling/
"""
import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from openalea.rhizodep.comparison.legacy_run import legacy_run
from openalea.rhizodep.comparison.parameter_sets import variants
from openalea.rhizodep.comparison.port_run import port_run

OUTPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "profiling")


def scenarios(parameters, n):
    return [dict(parameters, random_choice=1 + k) for k in range(n)]


def time_legacy(plants, n_steps):
    start = time.perf_counter()
    elements = 0
    for parameters in plants:
        run = legacy_run(parameters, n_steps=n_steps, record=True, record_every=n_steps)
        elements += len(run.steps[-1].vertices)
    return time.perf_counter() - start, elements


def time_port(plants, n_steps):
    start = time.perf_counter()
    run = port_run(plants, n_steps=n_steps, record=False)
    seconds = time.perf_counter() - start
    return seconds, run.scene.populations[0].data_structure.n_nodes()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plants", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--days", type=float, default=7.)
    parser.add_argument("--set", default="GB_soil_1.0/simple_growth_duration=False")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    os.makedirs(OUTPUTS, exist_ok=True)
    n_steps = int(round(args.days * 24))
    legacy, port = [], []
    for n in args.plants:
        plants = scenarios(variants()[args.set], n)
        legacy.append(time_legacy(plants, n_steps))
        port.append(time_port(plants, n_steps))
        print(f"{n:3d} plants, {legacy[-1][1]} elements: legacy {legacy[-1][0]:7.1f} s, port {port[-1][0]:6.1f} s "
              f"({legacy[-1][0] / port[-1][0]:.1f}× faster)")
    n = np.array(args.plants)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, runs, style in (("legacy, a run per plant", legacy, "ko-"), ("port, one population", port, "bs-")):
        seconds = np.array([r[0] for r in runs])
        axes[0].plot(n, seconds, style, label=label)
        axes[1].plot(n, seconds / n, style, label=label)
    for ax, ylabel in zip(axes, ("time of the run (s)", "time per plant (s)")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("plants")
        ax.set_ylabel(ylabel)
        ax.legend()
    axes[0].set_yscale("log")
    fig.suptitle(f"{args.set}, {args.days:g} days: {n[-1]} plants, the port "
                 f"{legacy[-1][0] / port[-1][0]:.1f}× faster")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS, f"population_{args.days:g}days.png"), dpi=120)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
