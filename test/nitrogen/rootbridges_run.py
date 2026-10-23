"""
Root_BRIDGES' growth component (RootGrowthModelCoupled) run on its own, recorded around its stages: the reference
the N subclass of the hooks is compared with (devplan_metafspm.md, §21, D-P6b).

Run with the Python of the Wheat-BRIDGES environment (its rhizodep and its former metafspm), never imported by the
rhizodep environment:

    <wheat-bridges python> rootbridges_run.py request.pkl records.pkl

request.pkl: dict(parameters, n_steps, C_hexose_root, AA, every, first, time_step, adventitious_to_emerge (the
shoot's delays of its next nodal emergences, or None)); records.pkl: a list of
(step, {"start": record, "potential": record, ...}), each record a plain dict (vertices, parent, edge_type, props,
plant, meta), as legacy_run's Snapshot.

What standalone running needs, Root_BRIDGES' composite providing it (Wheat-BRIDGES_framework is not edited):
- its seedling is built with rhizodep's ADDING_A_CHILD: Root_BRIDGES' own reads the mother's soil_temperature,
  which no element has before the inputs are linked (grass_bridges gives it an MTG);
- the properties it writes without creating them, created empty;
- the other models' variables its segmentation copies from the mother (C_sucrose_root, C_hexose_reserve, phloem_AA,
  xylem_AA, storage_protein), at 0;
- the given C_hexose_root and AA written on every element before each step, the carbon and nitrogen models' place;
- the Choregrapher's time step and a fresh schedule (as legacy_run).
"""
import copy
import functools
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # given.py, the varying concentrations

from openalea.metafspm.component_factory import Choregrapher
from openalea.rhizodep.root_growth import RootGrowthModel
from openalea.rootbridges.root_growth import RootGrowthModelCoupled

OTHER_MODELS = ("C_sucrose_root", "C_hexose_reserve", "phloem_AA", "xylem_AA", "storage_protein")
STAGES = ("potential", "actual", "segmentation", "post_growth")


def _schedule_copy(groups):
    return {family: {key: list(functors) for key, functors in steps.items()} for family, steps in groups.items()}


UNBOUND = _schedule_copy(Choregrapher().scheduled_groups)


def written_names():
    source = open(sys.modules[RootGrowthModelCoupled.__module__].__file__).read()
    return sorted(set(re.findall(r'(?:p|props)\["([a-zA-Z_]+)"\]', source)) - {"adventitious_to_emerge"})


def model(parameters, time_step):
    choregrapher = Choregrapher()
    choregrapher.scheduled_groups = _schedule_copy(UNBOUND)
    choregrapher.data_structure = {"soil": None, "root": None}
    choregrapher.add_simulation_time_step(time_step)
    own = RootGrowthModelCoupled.ADDING_A_CHILD
    RootGrowthModelCoupled.ADDING_A_CHILD = RootGrowthModel.ADDING_A_CHILD
    try:
        growth = RootGrowthModelCoupled(time_step=time_step, **parameters)
    finally:
        RootGrowthModelCoupled.ADDING_A_CHILD = own
    growth.pullable_inputs = {}
    for name in written_names() + ["root_hairs_volume", "AA"]:
        growth.props.setdefault(name, {})
    for name in OTHER_MODELS:
        growth.props[name] = {v: 0. for v in growth.g.vertices(scale=1)}
    return growth


def record(growth):
    g = growth.g
    vertices = sorted(g.vertices(scale=1))
    props = {}
    for name, values in g.properties().items():
        if name in ("total_living_struct_mass", "focus_elements") or not hasattr(values, "items"):
            continue
        props[name] = {vid: copy.copy(value) for vid, value in values.items()}
    meta = {"random_state": np.random.get_state(),
            "step_elongating_elements": list(getattr(growth, "step_elongating_elements", [])),
            "next_adventitious_primordium": getattr(growth, "next_adventitious_primordium", None),
            "adventitious_primordia_to_emerge": dict(getattr(growth, "adventitous_primordia_to_emerge", {}))}
    return dict(vertices=vertices, parent={v: g.parent(v) for v in vertices},
                edge_type={v: g.edge_type(v) for v in vertices}, props=props,
                plant={"total_living_struct_mass": g.property("total_living_struct_mass").get(1)}, meta=meta)


def write_given(growth, name, value, step):
    values = value(step, growth.g) if callable(value) else value
    prop = growth.props[name]
    for v in growth.g.vertices(scale=1):
        prop[v] = values[v] if isinstance(values, dict) else values


def run_stage(growth, stage):
    if stage == "potential":
        growth.potential_growth()
    elif stage == "actual":
        growth.actual_growth_and_corresponding_respiration()
    elif stage == "segmentation":
        growth.segmentation_and_primordia_formation()
    else:
        growth.post_growth_updating(modules_to_update=[], soil_boundaries_to_infer=[])


def run(request):
    growth = model(request["parameters"], request["time_step"])
    if request.get("adventitious_to_emerge") is not None:          # the shoot's next nodal emergences
        growth.props["adventitious_to_emerge"] = {1: list(request["adventitious_to_emerge"])}
    out = []
    for step in range(request["n_steps"]):
        write_given(growth, "C_hexose_root", request["C_hexose_root"], step)
        write_given(growth, "AA", request["AA"], step)
        growth.pull_available_inputs()
        recorded = step >= request["first"] and (step - request["first"]) % request["every"] == 0
        records = {"start": record(growth)} if recorded else None
        for stage in STAGES:
            run_stage(growth, stage)
            if recorded:
                records[stage] = record(growth)
        if recorded:
            out.append((step, records))
    return out


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as file:
        request = pickle.load(file)
    with open(os.devnull, "w") as quiet:                    # Root_BRIDGES prints at every step
        stdout, sys.stdout = sys.stdout, quiet
        try:
            records = run(request)
        finally:
            sys.stdout = stdout
    with open(sys.argv[2], "wb") as file:
        pickle.dump(records, file)
