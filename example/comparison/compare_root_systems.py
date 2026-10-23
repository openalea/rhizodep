"""
rhizodep's growth component against its legacy, side by side (devplan_metafspm.md): for each parameter set, the
legacy's root system (openalea.rhizodep.legacy, frozen), the port's, and their differences, to check exactness
visually as well as by the tests.

    left     the legacy's root system after the run, its elements as lines (projected on x-z, widths by radius),
             coloured by --color (root_order by default)
    middle   the port's, drawn by the same function from its record
    right    both overlaid (legacy black, port blue), the elements that differ in red (topology or any property);
             the title counts them and names the properties that differ most

and a second figure per set: the totals over time (length, structural mass, apices, lateral apices), legacy and port.
The suptitle says whether the two records are identical. Run from rhizodep/example/comparison:

    python compare_root_systems.py [set ...] [--days D] [--C C_hexose_root] [--color property]
    python compare_root_systems.py --list             the sets

Figures in ./outputs/root_systems/. Until the port's growth steps exist (P1-P5), the middle and right panels say so
and only the legacy is drawn.
"""
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt

from openalea.rhizodep.comparison.legacy_run import legacy_run
from openalea.rhizodep.comparison.parameter_sets import variants
from openalea.rhizodep.comparison.drawing import draw_root_system, positions, totals
from openalea.rhizodep.comparison.differences import differences

try:
    from openalea.rhizodep.comparison.port_run import port_run
except ImportError:
    port_run = None

OUTPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "root_systems")


def not_yet(ax, text="the port: not yet (P1-P5)"):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="grey")
    ax.set_xticks([])
    ax.set_yticks([])


def side_by_side(name, legacy, port, color):
    last_legacy = legacy.steps[-1]
    last_port = port.steps[-1] if port is not None else None
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    values = [float(v) for v in last_legacy.props[color].values() if v is not None]
    vmin, vmax = (min(values), max(values)) if values else (None, None)

    lines = draw_root_system(axes[0], last_legacy, color_by=color, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"legacy: {len(last_legacy.vertices)} vertices")
    fig.colorbar(lines, ax=axes[0], shrink=0.6, label=color)

    if last_port is None:
        not_yet(axes[1])
        not_yet(axes[2])
        verdict = "the port is not there yet"
    else:
        draw_root_system(axes[1], last_port, color_by=color, vmin=vmin, vmax=vmax)
        axes[1].set_title(f"port: {len(last_port.vertices)} vertices")
        diff = differences(last_legacy, last_port)
        xyz_legacy, xyz_port = positions(last_legacy), positions(last_port)
        for snapshot, xyz, colour in ((last_legacy, xyz_legacy, "black"), (last_port, xyz_port, "tab:blue")):
            lines = draw_root_system(axes[2], snapshot, color_by="length", xyz=xyz, linewidth_by_radius=False)
            lines.set_array(None)
            lines.set_color(colour)
            lines.set_alpha(0.6)
        differing = [v for v in diff.differing_vertices if v in xyz_legacy]
        for v in differing:
            (x0, _, z0), (x1, _, z1) = xyz_legacy[v]
            axes[2].plot([x0, x1], [z0, z1], color="red", linewidth=2.)
        axes[2].set_title(f"differences: {len(diff.differing_vertices)} vertices\n{diff.summary(limit=3)}",
                          fontsize=8)
        verdict = "identical" if diff.identical else "they differ"
        for ax in axes[1:]:
            ax.set_xlim(axes[0].get_xlim())
            ax.set_ylim(axes[0].get_ylim())
    for ax in axes:
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("z (m)")
    fig.suptitle(f"{name}: {verdict}")
    return fig


def over_time(name, legacy, port, time_step):
    keys = ("length", "struct_mass", "apices", "lateral_apices")
    fig, axes = plt.subplots(1, len(keys), figsize=(16, 4))
    for label, run, style in (("legacy", legacy, "k-"), ("port", port, "b--")):
        if run is None:
            continue
        hours = [step * time_step / 3600. for step in run.recorded_steps]
        series = [totals(s) for s in run.steps]
        for ax, key in zip(axes, keys):
            ax.plot(hours, [t[key] for t in series], style, label=label)
    for ax, key in zip(axes, keys):
        ax.set_title(key)
        ax.set_xlabel("time (h)")
    axes[0].legend()
    fig.suptitle(f"{name}: totals over time")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sets", nargs="*", help="parameter sets (default: all)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--days", type=float, default=10.)
    parser.add_argument("--C", type=float, default=1e-3, help="the given C_hexose_root (mol.g-1)")
    parser.add_argument("--color", default="root_order", help="the property the elements are coloured by")
    parser.add_argument("--every", type=int, default=6, help="record every N steps (the totals over time)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    sets = variants()
    if args.list:
        print("\n".join(sets))
        return
    if not args.show:
        matplotlib.use("Agg")
    os.makedirs(OUTPUTS, exist_ok=True)
    time_step = 3600
    n_steps = int(round(args.days * 24 * 3600 / time_step))
    for name in args.sets or sets:
        parameters = sets[name]
        legacy = legacy_run(parameters, n_steps=n_steps, C_hexose_root=args.C, time_step=time_step,
                            record_every=args.every)
        port = None
        if port_run is not None:
            port = port_run(parameters, n_steps=n_steps, C_hexose_root=args.C, time_step=time_step,
                            record_every=args.every)
        stem = name.replace("/", "_").replace("=", "-")
        side_by_side(name, legacy, port, args.color).savefig(os.path.join(OUTPUTS, f"{stem}.png"), dpi=130)
        over_time(name, legacy, port, time_step).savefig(os.path.join(OUTPUTS, f"{stem}_totals.png"), dpi=110)
        print(f"{name}: {totals(legacy.steps[-1])}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
