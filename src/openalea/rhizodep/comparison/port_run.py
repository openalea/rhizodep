"""
The port run as the legacy is (legacy_run), its records in the legacy's form, to compare them
(devplan_metafspm.md).

    run = port_run(parameters, n_steps=48, C_hexose_root=1e-3)
    run.scene              the metafspm Scene (one plant, growth_model(): the component alone)
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


_MODELS = {}


def growth_model(component=RootGrowth, transplant=False):
    """A plant model of the growth component *component* (RootGrowth or a subclass) alone, its resources given (as
    the legacy's harness gives them); its plant built by the component, or from a legacy record (*transplant*)."""
    key = (component, transplant)
    if key not in _MODELS:
        def __init__(self, data_structure, time_step):
            self.growth = component(data_structure=data_structure, time_step=time_step)
            self.components = [self.growth]

        def run(self):
            self.growth()

        initiators = (type(f"FromLegacy{component.__name__}", (FromLegacy,), {"component": component}),
                      ) if transplant else (component,)
        name = f"{component.__name__}{'Transplant' if transplant else 'Only'}"
        _MODELS[key] = type(name, (), dict(initiators=initiators, __init__=__init__, run=run,
                                           __doc__=growth_model.__doc__))
    return _MODELS[key]


@dataclass
class PortRun:
    scene: Scene
    steps: list = field(default_factory=list)
    recorded_steps: list = field(default_factory=list)


def port_scene(parameters=None, time_step=3600, component=RootGrowth):
    table = pd.DataFrame([dict(plant="p0", model=growth_model(component), x=0., y=0., z=0., rotation=0.,
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
    for name in type(growth).ELEMENT_PARAMETERS:
        values = np.asarray(growth.parameter_values(name))
        props[name] = {legacy_of[v]: x.item() for v, x in zip(ids, values)}
    parent, edge_type = {}, {}
    for v in ids.tolist():
        mpg_parent = g.parent(v)
        parent[legacy_of[v]] = legacy_of.get(mpg_parent) if mpg_parent is not None else None
        edge_type[legacy_of[v]] = g.edge_type(v) if parent[legacy_of[v]] is not None else ""
    plant = {"total_living_struct_mass": float(np.asarray(ds.get("total_living_struct_mass"))[0])}
    return Snapshot(vertices=sorted(legacy_of.values()), parent=parent, edge_type=edge_type, props=props,
                    plant=plant)


def port_run(parameters=None, n_steps=24, C_hexose_root=1e-3, time_step=3600, record_every=1, record=True,
             component=RootGrowth, given=None, setup=None):
    """*n_steps* steps of the port; the records every *record_every* steps (and the last), as legacy_run's.
    *given*: other inputs written before each step ({name: a number, or a function as C_hexose_root}); *setup*: a
    function of the scene, called once it is built."""
    scene = port_scene(parameters, time_step, component)
    if setup is not None:
        setup(scene)
    run = PortRun(scene=scene)
    if record:
        run.steps.append(snapshot(scene))
        run.recorded_steps.append(0)
    for step in range(n_steps):
        write_given(scene, step, "C_hexose_root", C_hexose_root)
        for name, value in (given or {}).items():
            write_given(scene, step, name, value)
        scene.run()
        if record and ((step + 1) % record_every == 0 or step == n_steps - 1):
            run.steps.append(snapshot(scene))
            run.recorded_steps.append(step + 1)
    return run


class _LegacyIds:
    """What a C_hexose_root function of legacy_run reads of the MTG: its vertices' (legacy) ids."""

    def __init__(self, ids):
        self.ids = ids

    def vertices(self, scale=1):
        return list(self.ids)


def write_given(scene, step, name, value):
    """A given input on every element, as legacy_run writes the hexose (a number, or a function of the step and the
    legacy's ids)."""
    ds = scene.populations[0].data_structure
    if value is None or not ds.has(name):              # not given, or not an input of this component
        return
    index = np.asarray(ds.get("vertex_index")).astype(int)
    if callable(value):
        values = value(step, _LegacyIds(index.tolist()))
        if isinstance(values, dict):
            current = np.array(ds.get(name), dtype=float)
            for i, legacy_id in enumerate(index.tolist()):
                if legacy_id in values:
                    current[i] = values[legacy_id]
            ds.set(name, current)
            return
        value = values
    ds.set(name, np.full(index.size, float(value)))


def write_C_hexose_root(scene, step, C_hexose_root):
    write_given(scene, step, "C_hexose_root", C_hexose_root)


# ── State transplant: the port on a legacy state (devplan §14) ──────────────────────────────────────────────────

CODES = {"lateral_root_emergence_possibility": {v: k for k, v in EMERGENCE.items()},
         "axis_type": {v: k for k, v in AXIS.items()}}


class FromLegacy(StructuralComponent):
    """An initiator building the plant from a legacy record (its scenario's "legacy_snapshot"), grouped as
    RootGrowth.initiate_plant groups its seedling: an Organ per `<` chain, an Axis per first-order root (an element
    branched with root_order 1), a lateral's Organ in its bearer's Axis."""
    scenario_keys = ("legacy_snapshot",)
    component = RootGrowth                              # the growth component whose variables the record gives

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        record = parameters["legacy_snapshot"]
        # The scenario's growth parameters are read, as RootGrowth.initiate_plant reads them: the inputs a scenario
        # sets (root_tissue_density) are then the initiator's, the values themselves coming from the record
        cls.component.plant_parameters(parameters)
        scales, labels = g.scales, g.labels
        children = {legacy_id: [] for legacy_id in record.vertices}
        roots = []
        for legacy_id in record.vertices:
            parent = record.parent[legacy_id]
            (children[parent] if parent is not None else roots).append(legacy_id)
        names = [f for f, spec in cls.component.__dataclass_fields__.items()
                 if spec.metadata.get("variable_type") in ("state_variable", "input")
                 and spec.metadata.get("scale") == scales.SubOrgan]

        def under(complex_, levels):
            for scale, label in levels:
                complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=label))
            return complex_

        axis_scales = ((scales.Axis, labels.Axis.Root), (scales.GrowthUnit, labels.GrowthUnit.Root),
                       (scales.Phytomer, labels.Phytomer.Root), (scales.Organ, labels.Organ.RootInternode))
        vids = {}
        stack = list(reversed(roots))
        while stack:
            legacy_id = stack.pop()
            apex = record.props["label"][legacy_id] == LABEL_APEX
            label = labels.SubOrgan.RootApex if apex else labels.SubOrgan.RootSegment
            parent = record.parent[legacy_id]
            if parent is None:
                vid = g.add_component(under(plant, axis_scales), **PropsConfig(scale=scales.SubOrgan, edge_type='/',
                                                                                label=label))
            elif record.edge_type[legacy_id] == "<":
                vid = g.add_child(vids[parent], **PropsConfig(scale=scales.SubOrgan, edge_type='<', label=label))
            else:
                if record.props["root_order"][legacy_id] == 1:
                    organ = under(plant, axis_scales)
                else:
                    organ = under(g.complex_at_scale(vids[parent], scales.Phytomer), axis_scales[-1:])
                vid = g.add_component_with_topo(organ, vids[parent], **PropsConfig(scale=scales.SubOrgan, edge_type='+',
                                                                                   label=label))
            vids[legacy_id] = vid
            for name in names:
                values = record.props.get(name, {})
                if legacy_id in values:
                    value = values[legacy_id]
                    g.property(name)[vid] = CODES[name][value] if name in CODES else value
            g.property("vertex_index")[vid] = legacy_id
            stack.extend(reversed(children[legacy_id]))
        g.property("total_living_struct_mass")[plant] = record.plant["total_living_struct_mass"]
        # The legacy's state outside its MTG: numpy's global generator, its elements elongated in actual growth
        stream = np.random.RandomState()
        if "random_state" in record.meta:
            stream.set_state(record.meta["random_state"])
        g.property("legacy_random")[plant] = stream
        for legacy_id in record.meta.get("step_elongating_elements", []):
            g.property("step_elongating")[vids[legacy_id]] = 1
        # The component's own state outside the record's variables, if it has any (a subclass's)
        transplanted = getattr(cls.component, "transplanted", None)
        if transplanted is not None:
            transplanted(g, plant, record, vids)


def transplant_scene(record, parameters=None, time_step=3600, component=RootGrowth):
    """A Scene of the port (*component*) whose plant is the legacy record *record* (parameters: the legacy run's)."""
    table = pd.DataFrame([dict(plant="p0", model=growth_model(component, transplant=True), x=0., y=0., z=0.,
                               rotation=0.,
                               scenario={"parameters": dict(parameters or {}, legacy_snapshot=record)})])
    table.attrs.update(xrange=1., yrange=1.)
    return Scene(table, time_step=time_step)


STAGE_INDEX = {"potential": 1, "actual": 4, "segmentation": 5, "postsegmentation": 6, "post_growth": 6}


def run_stage(scene, stage):
    """Run only the growth component's steps of one stage (its row in the Choregrapher's growth schedule)."""
    growth = scene.populations[0].instance.growth
    growth.pull_available_inputs()
    _, groups = growth.__dict__["_choregraphy"]
    for priority, functors in groups.items():
        if eval(priority)[3] == STAGE_INDEX[stage]:
            for functor in functors:
                functor()
