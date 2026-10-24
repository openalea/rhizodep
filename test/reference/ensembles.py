"""
Ensembles over seeds: the statistical reference of the growth components (devplan §29, D-X2), replacing the bitwise
comparisons once the code moves away from the legacy's float exactness.

    python test/reference/ensembles.py        writes test/reference/ensembles.json, from the code checked out

The reference was computed on the tag port-exact (the port bitwise the legacy, its fixes the defaults). Each
configuration: a component, a parameter set (its defaults otherwise, i.e. the reviewed fixes), 10 days, 8 seeds; the
totals after the run, their mean and standard deviation.
"""
import contextlib
import io
import json
import os

import numpy as np

from openalea.rhizodep.architecture import RootArchitecture
from openalea.rhizodep.comparison.drawing import totals
from openalea.rhizodep.comparison.port_run import port_run
from openalea.rhizodep.growth import RootGrowth
from openalea.rhizodep.growth_CN import RootGrowthCN
from openalea.rhizodep.parameter_sets import MAIZE, PEA, WHEAT_SOISSON

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = range(1, 9)
DAYS = 10


def own(parameters, component):
    return {k: v for k, v in parameters.items() if k in component.__dataclass_fields__}


CONFIGURATIONS = {
    "wheat RootGrowth": (RootGrowth, WHEAT_SOISSON, {}),
    "wheat RootGrowthCN": (RootGrowthCN, WHEAT_SOISSON, {"AA": 1e-3}),
    "pea RootArchitecture": (RootArchitecture, own(PEA, RootArchitecture), {}),
    "maize RootArchitecture": (RootArchitecture, own(MAIZE, RootArchitecture), {}),
}
METRICS = ("length", "struct_mass", "vertices", "apices", "lateral_apices", "mean_radius", "first_order_radius_spread")


def metrics(record):
    t = totals(record)
    radii = [record.props["radius"][v] for v in record.vertices if record.props["length"][v] > 0]
    first = [record.props["original_radius"][v] for v in record.vertices
             if record.props["root_order"][v] == 1 and record.props["label"][v] == 2]
    return dict(length=t["length"], struct_mass=t["struct_mass"], vertices=t["vertices"], apices=t["apices"],
                lateral_apices=t["lateral_apices"], mean_radius=float(np.mean(radii)),
                first_order_radius_spread=float(np.std(first)))


def ensemble(name, seeds=SEEDS, days=DAYS):
    """{metric: [value per seed]} of a configuration."""
    component, parameters, given = CONFIGURATIONS[name]
    C = None if component is RootArchitecture else 1e-3
    values = {metric: [] for metric in METRICS}
    for seed in seeds:
        with contextlib.redirect_stdout(io.StringIO()):
            run = port_run(dict(parameters, random_choice=seed), n_steps=24 * days, C_hexose_root=C,
                           record_every=24 * days, component=component, given=given)
        for metric, value in metrics(run.steps[-1]).items():
            values[metric].append(value)
    return values


if __name__ == "__main__":
    reference = {}
    for name in CONFIGURATIONS:
        values = ensemble(name)
        reference[name] = {metric: dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)), values=v)
                           for metric, v in values.items()}
        print(name, {m: round(reference[name][m]["mean"], 4) for m in METRICS})
    with open(os.path.join(HERE, "ensembles.json"), "w") as file:
        json.dump(dict(source="port-exact", days=DAYS, seeds=list(SEEDS), configurations=reference), file, indent=1)
