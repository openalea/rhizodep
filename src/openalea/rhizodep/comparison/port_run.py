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

    scene = transplant_scene(snapshot, parameters)   the port's MPG built from a legacy record (devplan §14)
    run_stage(scene, "potential")                    only that stage's steps of the port

LEFT_OUT: the legacy properties the port does not have, and why.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.data_structure.configs import PropsConfig
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


# ── State transplant: the port on a legacy state (devplan §14) ──────────────────────────────────────────────────

CODES = {"lateral_root_emergence_possibility": {v: k for k, v in EMERGENCE.items()},
         "axis_type": {v: k for k, v in AXIS.items()}}


class FromLegacy(StructuralComponent):
    """An initiator building the plant from a legacy record (its scenario's "legacy_snapshot"), grouped as
    RootGrowth.initiate_plant groups its seedling: an Organ per `<` chain, an Axis per first-order root (an element
    branched with root_order 1), a lateral's Organ in its bearer's Axis."""
    scenario_keys = ("legacy_snapshot",)

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        snap = parameters["legacy_snapshot"]
        p = RootGrowth.plant_parameters(parameters)
        s, labels = g.scales, g.labels
        children = {v: [] for v in snap.vertices}
        roots = []
        for v in snap.vertices:
            (children[snap.parent[v]].append(v) if snap.parent[v] is not None else roots.append(v))
        names = [f for f, spec in RootGrowth.__dataclass_fields__.items()
                 if spec.metadata.get("variable_type") in ("state_variable", "input")
                 and spec.metadata.get("scale") == s.SubOrgan]

        def under(complex_, scales):
            for scale, label in scales:
                complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=label))
            return complex_

        axis_scales = ((s.Axis, labels.Axis.Root), (s.GrowthUnit, labels.GrowthUnit.Root),
                       (s.Phytomer, labels.Phytomer.Root), (s.Organ, labels.Organ.RootInternode))
        vids = {}
        stack = list(reversed(roots))
        while stack:
            v = stack.pop()
            label = labels.SubOrgan.RootApex if snap.props["label"][v] == LABEL_APEX else labels.SubOrgan.RootSegment
            parent = snap.parent[v]
            if parent is None:
                vid = g.add_component(under(plant, axis_scales), **PropsConfig(scale=s.SubOrgan, edge_type='/',
                                                                                label=label))
            elif snap.edge_type[v] == "<":
                vid = g.add_child(vids[parent], **PropsConfig(scale=s.SubOrgan, edge_type='<', label=label))
            else:
                if snap.props["root_order"][v] == 1:
                    organ = under(plant, axis_scales)
                else:
                    organ = under(g.complex_at_scale(vids[parent], s.Phytomer), axis_scales[-1:])
                vid = g.add_component_with_topo(organ, vids[parent], **PropsConfig(scale=s.SubOrgan, edge_type='+',
                                                                                   label=label))
            vids[v] = vid
            for name in names:
                values = snap.props.get(name, {})
                if v in values:
                    value = values[v]
                    g.property(name)[vid] = CODES[name][value] if name in CODES else value
            g.property("vertex_index")[vid] = v
            stack.extend(reversed(children[v]))
        g.property("total_living_struct_mass")[plant] = snap.plant["total_living_struct_mass"]
        # The legacy's state outside its MTG: numpy's global generator, its elements elongated in actual growth
        stream = np.random.RandomState()
        if "random_state" in snap.meta:
            stream.set_state(snap.meta["random_state"])
        g.property("legacy_random")[plant] = stream
        for v in snap.meta.get("step_elongating_elements", []):
            g.property("step_elongating")[vids[v]] = 1


class TransplantModel(GrowthOnly):
    initiators = (FromLegacy,)


def transplant_scene(snap, parameters=None, time_step=3600):
    """A Scene of the port whose plant is the legacy record *snap* (parameters: the legacy run's)."""
    table = pd.DataFrame([dict(plant="p0", model=TransplantModel, x=0., y=0., z=0., rotation=0.,
                               scenario={"parameters": dict(parameters or {}, legacy_snapshot=snap)})])
    table.attrs.update(xrange=1., yrange=1.)
    return Scene(table, time_step=time_step)


STAGE_INDEX = {"potential": 1, "actual": 4, "segmentation": 5, "postsegmentation": 6}


def run_stage(scene, stage):
    """Run only the growth component's steps of one stage (its row in the Choregrapher's growth schedule)."""
    growth = scene.populations[0].instance.growth
    growth.pull_available_inputs()
    _, groups = growth.__dict__["_choregraphy"]
    for priority, functors in groups.items():
        if eval(priority)[3] == STAGE_INDEX[stage]:
            for functor in functors:
                functor()
