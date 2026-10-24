"""
The cost of rhizodep's growth component, the legacy (openalea.rhizodep.legacy, its loops over the MTG) against the
port (openalea.rhizodep.growth, vectorised steps on metafspm's DataStructure), on the same run: both are bitwise
identical (test/comparison), so the difference is the implementation only.

    left     the time of each step against the number of elements of the root system, legacy and port
    middle   the cumulative time against the simulated time
    right    where the time goes, per stage (potential growth, actual growth, segmentation, post-growth), for each

and, printed, the totals, the speed-up and, with --profile, the functions that cost most in each. Run from
rhizodep/example/comparison:

    python profile_growth.py [--set NAME] [--days D] [--profile]       figures in ./outputs/profiling/
    python profile_growth.py --list                                    the parameter sets

The stages are timed one after the other (the port's through its schedule's rows, as run_stage does); a step of
the port also includes the given hexose written to the DataStructure, as the legacy's harness writes it on the MTG.
"""
import argparse
import cProfile
import io
import os
import pstats
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from openalea.rhizodep.comparison.legacy_run import STAGES, _run_stage, legacy_model, write_C_hexose_root
from openalea.rhizodep.comparison.parameter_sets import variants
from openalea.rhizodep.comparison.port_run import port_scene, run_stage
from openalea.rhizodep.comparison.port_run import write_C_hexose_root as write_port_C_hexose_root

OUTPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "profiling")
STAGE_NAMES = {"potential": "potential growth", "actual": "actual growth", "segmentation": "segmentation",
               "post_growth": "post-growth"}


def time_legacy(parameters, n_steps, C_hexose_root, time_step=3600):
    """Per step: the number of elements and the seconds spent in each stage."""
    model = legacy_model(parameters, time_step)
    elements, seconds = [], {stage: [] for stage in STAGES}
    for step in range(n_steps):
        write_C_hexose_root(model, step, C_hexose_root)
        model.pull_available_inputs()
        for stage in STAGES:
            start = time.perf_counter()
            _run_stage(model, stage)
            seconds[stage].append(time.perf_counter() - start)
        elements.append(len(model.g) - 1)
    return np.array(elements), {stage: np.array(values) for stage, values in seconds.items()}


def time_port(parameters, n_steps, C_hexose_root, time_step=3600):
    scene = port_scene(parameters, time_step)
    ds = scene.populations[0].data_structure
    elements, seconds = [], {stage: [] for stage in STAGES}
    for step in range(n_steps):
        start = time.perf_counter()
        write_port_C_hexose_root(scene, step, C_hexose_root)
        writing = time.perf_counter() - start
        for stage in STAGES:
            start = time.perf_counter()
            run_stage(scene, stage)
            seconds[stage].append(time.perf_counter() - start + (writing if stage == "potential" else 0.))
        elements.append(ds.n_nodes())
    return np.array(elements), {stage: np.array(values) for stage, values in seconds.items()}


def profiled(function, *args):
    """The function's result, and its 15 most costly functions (cumulative time), as text."""
    profile = cProfile.Profile()
    profile.enable()
    result = function(*args)
    profile.disable()
    text = io.StringIO()
    pstats.Stats(profile, stream=text).sort_stats("cumulative").print_stats(15)
    return result, text.getvalue()


def figure(name, days, legacy, port, time_step=3600):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    colours = {"legacy": "black", "port": "tab:blue"}
    for label, (elements, seconds) in (("legacy", legacy), ("port", port)):
        per_step = sum(seconds.values())
        axes[0].plot(elements, per_step, ".", ms=3, color=colours[label], label=label)
        hours = np.arange(1, elements.size + 1) * time_step / 3600.
        axes[1].plot(hours / 24., np.cumsum(per_step), color=colours[label], label=label)
    axes[0].set_xlabel("elements of the root system")
    axes[0].set_ylabel("time of a step (s)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[1].set_xlabel("simulated time (days)")
    axes[1].set_ylabel("cumulative time (s)")
    axes[1].legend()
    bottom = np.zeros(2)
    for stage in STAGES:
        values = np.array([legacy[1][stage].sum(), port[1][stage].sum()])
        axes[2].bar(["legacy", "port"], values, bottom=bottom, label=STAGE_NAMES[stage])
        bottom += values
    axes[2].set_ylabel("total time (s)")
    axes[2].legend(fontsize=8)
    speedup = bottom[0] / bottom[1]
    fig.suptitle(f"{name}, {days:g} days, {legacy[0][-1]} elements: the port {speedup:.1f}× faster "
                 f"({bottom[0]:.0f} s against {bottom[1]:.0f} s)")
    fig.tight_layout()
    return fig, speedup


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="wheat_soisson/simple_growth_duration=True", help="the parameter set")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--days", type=float, default=15.)
    parser.add_argument("--C", type=float, default=1e-3, help="the given C_hexose_root (mol.g-1)")
    parser.add_argument("--profile", action="store_true", help="print the costliest functions of each")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    sets = variants()
    if args.list:
        print("\n".join(sets))
        return
    if not args.show:
        matplotlib.use("Agg")
    os.makedirs(OUTPUTS, exist_ok=True)
    parameters = sets[args.set]
    n_steps = int(round(args.days * 24))
    if args.profile:
        legacy, legacy_profile = profiled(time_legacy, parameters, n_steps, args.C)
        port, port_profile = profiled(time_port, parameters, n_steps, args.C)
    else:
        legacy = time_legacy(parameters, n_steps, args.C)
        port = time_port(parameters, n_steps, args.C)
    fig, speedup = figure(args.set, args.days, legacy, port)
    stem = args.set.replace("/", "_").replace("=", "-")
    fig.savefig(os.path.join(OUTPUTS, f"{stem}_{args.days:g}days.png"), dpi=120)
    print(f"{args.set}, {args.days:g} days, {legacy[0][-1]} elements (legacy) / {port[0][-1]} (port)")
    for label, (_, seconds) in (("legacy", legacy), ("port", port)):
        stages = ", ".join(f"{STAGE_NAMES[stage]} {seconds[stage].sum():.1f} s" for stage in STAGES)
        print(f"  {label}: {sum(s.sum() for s in seconds.values()):.1f} s ({stages})")
    print(f"  the port {speedup:.1f}× faster")
    if args.profile:
        print("\nlegacy, the costliest functions:\n" + legacy_profile)
        print("port, the costliest functions:\n" + port_profile)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
