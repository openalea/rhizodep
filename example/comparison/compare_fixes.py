"""
The reviewed fixes of the legacy (devplan_metafspm.md §25), one by one: for each, the port with the legacy's
behaviours (its LEGACY preset) and the same run with that fix alone, then with all of them (the defaults), drawn side
by side with the totals over time, to see what each changes.

    left     the root system with the legacy's behaviours (bitwise the legacy: test/comparison)
    middle   with the fix
    right    the totals over time (length, structural mass, apices, mean radius), legacy dashed

and, printed, a table of the totals after the run. Run from rhizodep/example/comparison:

    python compare_fixes.py [fix ...] [--set NAME] [--days D] [--C C_hexose_root]    figures in ./outputs/fixes/
    python compare_fixes.py --list                                                    the fixes

The fixes are the options of RootArchitecture and RootGrowth (their legacy_* parameters and collar_supply); "all" is
their defaults.
"""
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from openalea.rhizodep.comparison.drawing import draw_root_system, totals
from openalea.rhizodep.comparison.parameter_sets import variants
from openalea.rhizodep.comparison.port_run import port_run
from openalea.rhizodep.growth import RootGrowth

OUTPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "fixes")
FIXES = {
    "g16 pipe model": dict(legacy_pipe_model=False),
    "g1 segmentation masses": dict(legacy_segmentation_masses=False),
    "g4 growth duration draws": dict(legacy_growth_duration_draws=False),
    "g5 nodule radius": dict(legacy_nodule_radius=False),
    "g6 zero temperature": dict(legacy_zero_temperature=False),
    "g11 supports": dict(legacy_support_development=False),
    "g15 seedling seeds": dict(legacy_seedling_seeds=False),
    "g3 primordium soil temperature": dict(legacy_primordium_soil_temperature=False),
    "g9 collar supply": dict(collar_supply="collar"),
    "g18 lateral growth extender": dict(legacy_lateral_growth_extender=False),
    "g19 lateral diameter threshold": dict(legacy_lateral_diameter_threshold=False),
    "g20 branching positions": dict(legacy_branching_positions=False),
    "g22 dead living mass": dict(legacy_dead_living_mass=False),
    "all": {name: RootGrowth.__dataclass_fields__[name].default for name in RootGrowth.LEGACY},   # the defaults
}


def summary(snapshot):
    t = totals(snapshot)
    radii = [snapshot.props["radius"][v] for v in snapshot.vertices if snapshot.props["length"][v] > 0]
    t["mean_radius"] = float(np.mean(radii)) if radii else 0.
    return t


def compare(fix, parameters, days, C, time_step=3600, every=6):
    n_steps = int(round(days * 24 * 3600 / time_step))
    legacy = port_run(parameters, n_steps=n_steps, C_hexose_root=C, record_every=every)
    fixed = port_run(dict(parameters, **FIXES[fix]), n_steps=n_steps, C_hexose_root=C, record_every=every)
    return legacy, fixed


def figure(fix, name, legacy, fixed, time_step=3600):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), gridspec_kw=dict(width_ratios=[1, 1, 1.4]))
    for ax, run, title in ((axes[0], legacy, "legacy"), (axes[1], fixed, fix)):
        last = run.steps[-1]
        lines = draw_root_system(ax, last, color_by="radius", cmap="viridis")
        ax.set_title(f"{title}: {len(last.vertices)} elements")
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("z (m)")
    xlim = [min(a.get_xlim()[0] for a in axes[:2]), max(a.get_xlim()[1] for a in axes[:2])]
    ylim = [min(a.get_ylim()[0] for a in axes[:2]), max(a.get_ylim()[1] for a in axes[:2])]
    for ax in axes[:2]:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    fig.colorbar(lines, ax=axes[1], shrink=0.6, label="radius (m)")
    inner = axes[2].inset_axes([0, 0, 1, 1])
    axes[2].axis("off")
    keys = ("length", "struct_mass", "apices", "mean_radius")
    grid = [inner.inset_axes([0.05 + 0.5 * (k % 2), 0.55 - 0.5 * (k // 2), 0.42, 0.38]) for k in range(4)]
    inner.axis("off")
    for ax, key in zip(grid, keys):
        for run, style, label in ((legacy, "k--", "legacy"), (fixed, "b-", fix)):
            hours = np.array(run.recorded_steps) * time_step / 3600.
            ax.plot(hours / 24., [summary(s)[key] for s in run.steps], style, label=label)
        ax.set_title(key, fontsize=9)
        ax.tick_params(labelsize=7)
    grid[0].legend(fontsize=7)
    fig.suptitle(f"{name}: {fix}")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixes", nargs="*", help="the fixes (default: all of them, one by one, then together)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--set", default="wheat_soisson/simple_growth_duration=False")
    parser.add_argument("--days", type=float, default=10.)
    parser.add_argument("--C", type=float, default=1e-3, help="the given C_hexose_root (mol.g-1)")
    parser.add_argument("--with", dest="extra", nargs="*", default=[],
                        help="parameters set over the set's, name=value (e.g. SGC=1 to see the pipe model)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.list:
        print("\n".join(FIXES))
        return
    if not args.show:
        matplotlib.use("Agg")
    os.makedirs(OUTPUTS, exist_ok=True)
    parameters = dict(variants()[args.set], **{k: float(v) for k, v in (e.split("=") for e in args.extra)})
    rows = []
    for fix in args.fixes or FIXES:
        legacy, fixed = compare(fix, parameters, args.days, args.C)
        stem = f"{args.set.replace('/', '_').replace('=', '-')}_{fix.split()[0]}"
        figure(fix, args.set, legacy, fixed).savefig(os.path.join(OUTPUTS, f"{stem}.png"), dpi=110)
        before, after = summary(legacy.steps[-1]), summary(fixed.steps[-1])
        rows.append((fix, before, after))
        print(f"{fix:32s} " + "  ".join(f"{key} {before[key]:.4g} -> {after[key]:.4g}"
                                         for key in ("length", "struct_mass", "vertices", "apices", "mean_radius")))
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
