"""
rhizodep's growth component against its legacy (openalea.rhizodep.legacy, frozen): the same model, the port with the
reviewed fixes (devplan §25) and its own draws, so the root systems compare statistically (the tag port-exact
reproduces the legacy bit for bit). For each parameter set:

    figure 1   the legacy's root system and the port's after the run (projected on x-z, widths by radius), coloured
               by --color (root_order by default)
    figure 2   their totals over time (length, structural mass, apices, lateral apices)
    printed    with --seeds N: the totals' mean and standard deviation over N seeds, legacy and port

Run from rhizodep/example/comparison:

    python compare_root_systems.py [set ...] [--days D] [--C C_hexose_root] [--color property] [--seeds N]
    python compare_root_systems.py --list             the sets

Figures in ./outputs/root_systems/.
"""
import argparse
import contextlib
import io
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from openalea.archisimple.drawing import draw_root_system, totals
from openalea.rhizodep.comparison.legacy_run import legacy_run
from openalea.rhizodep.comparison.parameter_sets import PARAMETER_SETS
from openalea.rhizodep.comparison.port_run import port_run

OUTPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "root_systems")
# The legacy's own calibrations (the ArchiSimple sets need the port's options)
SETS = {name: PARAMETER_SETS[name] for name in ("defaults", "wheat_soisson")}
KEYS = ("length", "struct_mass", "apices", "lateral_apices")


def quietly(function, *args, **keywords):
    with contextlib.redirect_stdout(io.StringIO()):            # the legacy prints its tables' notes
        return function(*args, **keywords)


def side_by_side(name, legacy, port, color):
    last_legacy, last_port = legacy.steps[-1], port.steps[-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharex=True, sharey=True)
    values = [float(v) for record in (last_legacy, last_port) for v in record.props[color].values() if v is not None]
    vmin, vmax = min(values), max(values)
    for ax, label, record in ((axes[0], "legacy", last_legacy), (axes[1], "port", last_port)):
        lines = draw_root_system(ax, record, color_by=color, vmin=vmin, vmax=vmax)
        t = totals(record)
        ax.set_title(f"{label}: {t['length']:.2f} m, {t['vertices']} elements, {t['apices']} apices")
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("z (m)")
    fig.colorbar(lines, ax=axes.tolist(), shrink=0.6, label=color)
    fig.suptitle(f"{name}: one realisation each (their draws differ)")
    return fig


def over_time(name, legacy, port, time_step):
    fig, axes = plt.subplots(1, len(KEYS), figsize=(16, 4))
    for label, run, style in (("legacy", legacy, "k-"), ("port", port, "b--")):
        hours = [step * time_step / 3600. for step in run.recorded_steps]
        series = [totals(s) for s in run.steps]
        for ax, key in zip(axes, KEYS):
            ax.plot(hours, [t[key] for t in series], style, label=label)
    for ax, key in zip(axes, KEYS):
        ax.set_title(key)
        ax.set_xlabel("time (h)")
    axes[0].legend()
    fig.suptitle(f"{name}: totals over time")
    fig.tight_layout()
    return fig


def ensembles(parameters, seeds, n_steps, C):
    """{"legacy" / "port": {total: [value per seed]}}."""
    out = {"legacy": {key: [] for key in KEYS}, "port": {key: [] for key in KEYS}}
    for seed in range(1, seeds + 1):
        scenario = dict(parameters, random_choice=seed)
        for label, run in (("legacy", quietly(legacy_run, scenario, n_steps=n_steps, C_hexose_root=C,
                                              record_every=n_steps)),
                           ("port", port_run(scenario, n_steps=n_steps, C_hexose_root=C, record_every=n_steps))):
            t = totals(run.steps[-1])
            for key in KEYS:
                out[label][key].append(t[key])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sets", nargs="*", help="parameter sets (default: all)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--days", type=float, default=10.)
    parser.add_argument("--C", type=float, default=1e-3, help="the given C_hexose_root (mol.g-1)")
    parser.add_argument("--color", default="root_order", help="the property the elements are coloured by")
    parser.add_argument("--every", type=int, default=6, help="record every N steps (the totals over time)")
    parser.add_argument("--seeds", type=int, default=0, help="the ensembles' size (0: none)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.list:
        print("\n".join(SETS))
        return
    if not args.show:
        matplotlib.use("Agg")
    os.makedirs(OUTPUTS, exist_ok=True)
    time_step = 3600
    n_steps = int(round(args.days * 24 * 3600 / time_step))
    for name in args.sets or SETS:
        parameters = SETS[name]
        legacy = quietly(legacy_run, parameters, n_steps=n_steps, C_hexose_root=args.C, time_step=time_step,
                         record_every=args.every)
        port = port_run(parameters, n_steps=n_steps, C_hexose_root=args.C, time_step=time_step,
                        record_every=args.every)
        side_by_side(name, legacy, port, args.color).savefig(os.path.join(OUTPUTS, f"{name}.png"), dpi=130)
        over_time(name, legacy, port, time_step).savefig(os.path.join(OUTPUTS, f"{name}_totals.png"), dpi=110)
        if args.seeds:
            values = ensembles(parameters, args.seeds, n_steps, args.C)
            print(f"{name}, {args.seeds} seeds, {args.days:g} days:")
            for key in KEYS:
                print("  " + key + ": " + ", ".join(f"{label} {np.mean(values[label][key]):.4g} ± "
                                                    f"{np.std(values[label][key], ddof=1):.2g}"
                                                    for label in ("legacy", "port")))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
