"""
The port run as the legacy is (legacy_run), its records in the legacy's form, to compare them
(devplan_metafspm.md).

    run = port_run(parameters, n_steps=48, C_hexose_root=1e-3)
    run.scene              the metafspm Scene (one plant, the GrowthOnly model)
    run.steps[i]           Snapshots as legacy_run's, keyed by the legacy's vertex ids (vertex_index):
                           the SubOrgans, their parents through the MPG's topology (the `+` across axes included),
                           every declared variable, decoded where the port codes it (the MPG label as the legacy's
                           label 1 / 2, the EMERGENCE_* and AXIS_* codes as the legacy's strings), and the
                           per-element copies of parameters (RootGrowth.ELEMENT_PARAMETERS)

LEFT_OUT: the legacy properties the port does not have, and why.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from openalea.metafspm.scene.scene import Scene
from openalea.rhizodep.comparison.legacy_run import Snapshot
from openalea.rhizodep.growth import (AXIS_LATERAL, AXIS_NODAL, AXIS_SEMINAL, EMERGENCE_IMPOSSIBLE,
                                      EMERGENCE_POSSIBLE, RootGrowth)

LEFT_OUT = {
    "list_of_elongation_supporting_elements": "the supply zone's bookkeeping, replaced by path windows (devplan §12)",
    "list_of_elongation_supporting_elements_hexose": "idem",
    "list_of_elongation_supporting_elements_mass": "idem",
    "axis_index": "the Axis vertex (devplan §12); the legacy computes it only for its plots",
    "edge_type": "the MPG's topology (Snapshot.edge_type)",
}
EMERGENCE = {EMERGENCE_IMPOSSIBLE: "Impossible", EMERGENCE_POSSIBLE: "Possible"}
AXIS = {AXIS_SEMINAL: "seminal", AXIS_NODAL: "nodal", AXIS_LATERAL: "lateral"}
LABEL_SEGMENT, LABEL_APEX = 1, 2


class GrowthOnly:
    """A plant model of the growth component alone, the hexose given (as the legacy's harness does)."""
    initiators = (RootGrowth,)

    def __init__(self, data_structure, time_step):
        self.growth = RootGrowth(data_structure=data_structure, time_step=time_step)
        self.components = [self.growth]

    def run(self):
        self.growth()


@dataclass
class PortRun:
    scene: Scene
    steps: list = field(default_factory=list)
    recorded_steps: list = field(default_factory=list)


def port_scene(parameters=None, time_step=3600):
    table = pd.DataFrame([dict(plant="p0", model=GrowthOnly, x=0., y=0., z=0., rotation=0.,
                               scenario={"parameters": dict(parameters or {})})])
    table.attrs.update(xrange=1., yrange=1.)
    return Scene(table, time_step=time_step)


def snapshot(scene) -> Snapshot:
    """The port's state in the legacy's form (one plant)."""
    population = scene.populations[0]
    ds, growth = population.data_structure, population.instance.growth
    g = ds.mtg
    ids = np.asarray(ds.entity_ids("node"))
    index = np.asarray(ds.get("vertex_index")).astype(int)
    legacy_of = dict(zip(ids.tolist(), index.tolist()))
    apex_label = g.labels.SubOrgan.RootApex
    props = {}
    for name, spec in growth._variable_specs.items():
        if spec.variable_type in ("state_variable", "input") and ds.location(name) == "node":
            values = np.asarray(ds.get(name))
            if name == "lateral_root_emergence_possibility":
                props[name] = {legacy_of[v]: EMERGENCE[int(x)] for v, x in zip(ids, values)}
            elif name == "axis_type":
                props[name] = {legacy_of[v]: AXIS[int(x)] for v, x in zip(ids, values)}
            else:
                props[name] = {legacy_of[v]: x.item() for v, x in zip(ids, values)}
    props["label"] = {legacy_of[v]: LABEL_APEX if g.label(v) == apex_label else LABEL_SEGMENT for v in ids.tolist()}
    for name in RootGrowth.ELEMENT_PARAMETERS:
        values = np.asarray(growth.parameter_values(name))
        props[name] = {legacy_of[v]: x.item() for v, x in zip(ids, values)}
    parent, edge_type = {}, {}
    for v in ids.tolist():
        p = g.parent(v)
        parent[legacy_of[v]] = legacy_of.get(p) if p is not None else None
        edge_type[legacy_of[v]] = g.edge_type(v) if parent[legacy_of[v]] is not None else ""
    plant = {"total_living_struct_mass": float(np.asarray(ds.get("total_living_struct_mass"))[0])}
    return Snapshot(vertices=sorted(legacy_of.values()), parent=parent, edge_type=edge_type, props=props,
                    plant=plant)


def port_run(parameters=None, n_steps=24, C_hexose_root=1e-3, time_step=3600, record_every=1, record=True):
    """*n_steps* steps of the port; the records every *record_every* steps (and the last), as legacy_run's."""
    scene = port_scene(parameters, time_step)
    run = PortRun(scene=scene)
    if record:
        run.steps.append(snapshot(scene))
        run.recorded_steps.append(0)
    if n_steps:
        raise NotImplementedError("the port's growth steps come with P2-P5 (devplan §8)")
    return run
