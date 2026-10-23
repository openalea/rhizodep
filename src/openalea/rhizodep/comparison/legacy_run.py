"""
The legacy growth component run on its own: the reference the port is compared with (devplan_metafspm.md, P0).

    run = legacy_run(parameters, n_steps=48, C_hexose_root=1e-3)
    run.model              the legacy RootGrowthModel (openalea.rhizodep.legacy), after the last step
    run.steps[i]           the i-th record (the first: after the initiation), a Snapshot, after the step
                           run.recorded_steps[i] (0: the initiation, n: after n steps):
        .vertices            the MTG's vertices at scale 1, ascending
        .parent              {vid: parent vid or None} (the MTG's, not the collar's re-rooting in parent_id)
        .edge_type           {vid: "<" or "+"}
        .props               {name: {vid: value}}: every per-vertex property, declared or not
        .plant               {name: value}: the plant-scale states (total_living_struct_mass)

The growth component is driven as rhizodep's composite drives it, without the other components (Q-R6):
- the hexose concentration is given: *C_hexose_root* (a number, or a function (step, g) -> {vid: value} or a
  number) is written on every vertex before each step, the carbon model's place;
- soil_temperature and root_tissue_density are the inputs' defaults, or the scenario's;
- nothing is repartitioned (modules_to_update=[]), so new vertices keep their own C_hexose_root until it is written.

Two things the composite provides are set here, the legacy needing them standalone:
- the Choregrapher's global time step (rhizodep.py: Choregrapher().add_simulation_time_step);
- a fresh schedule for each model: the Choregrapher is a singleton whose steps are bound to the first model built
  (the composite ran one plant per process), so its unbound schedule, as registered at import, is restored before
  each model is built, and its data unset;
- an empty pullable_inputs (set by the coupling), and the root_hairs_volume property, which post_growth_updating
  writes but nothing in the growth component creates (another component's, in the composite).
"""
import copy
import functools
from dataclasses import dataclass, field

from openalea.rhizodep.legacy import RootGrowthModel
from openalea.rhizodep.legacy._metafspm.component_factory import Choregrapher


@dataclass
class Snapshot:
    vertices: list
    parent: dict
    edge_type: dict
    props: dict
    plant: dict


@dataclass
class LegacyRun:
    model: RootGrowthModel
    steps: list = field(default_factory=list)
    recorded_steps: list = field(default_factory=list)


PLANT_SCALE = ("total_living_struct_mass",)


def _schedule_copy(groups):
    return {family: {key: list(functors) for key, functors in steps.items()} for family, steps in groups.items()}


_UNBOUND_SCHEDULE = _schedule_copy(Choregrapher().scheduled_groups)
if any(isinstance(f, functools.partial) for steps in _UNBOUND_SCHEDULE.values() for fs in steps.values() for f in fs):
    raise RuntimeError("a legacy model was built before openalea.rhizodep.comparison.legacy_run was imported: "
                       "import it first, the Choregrapher's schedule is then still unbound")


def reset_choregrapher():
    """The Choregrapher as before any model was built: its schedule unbound, no data."""
    choregrapher = Choregrapher()
    choregrapher.scheduled_groups = _schedule_copy(_UNBOUND_SCHEDULE)
    choregrapher.data_structure = {"soil": None, "root": None}


def snapshot(model):
    """Every property of the legacy's MTG, copied."""
    g = model.g
    vertices = sorted(g.vertices(scale=1))
    props = {}
    for name, values in g.properties().items():
        if name in PLANT_SCALE or name == "focus_elements" or not hasattr(values, "items"):
            continue
        props[name] = {vid: copy.copy(value) for vid, value in values.items()}
    plant = {name: g.property(name).get(1) for name in PLANT_SCALE}
    parent = {vid: g.parent(vid) for vid in vertices}
    edge_type = {vid: g.edge_type(vid) for vid in vertices}
    return Snapshot(vertices=vertices, parent=parent, edge_type=edge_type, props=props, plant=plant)


def write_C_hexose_root(model, step, C_hexose_root):
    g = model.g
    values = C_hexose_root(step, g) if callable(C_hexose_root) else C_hexose_root
    C = g.property("C_hexose_root")
    if isinstance(values, dict):
        for vid, value in values.items():
            C[vid] = value
    else:
        for vid in g.vertices(scale=1):
            C[vid] = values


def legacy_model(parameters=None, time_step=3600, before_post_growth=False):
    """The legacy RootGrowthModel on its own seedling, with *parameters* as its scenario. *before_post_growth*: the
    model as its __init__ leaves it but for the post_growth_updating that ends it (the initial structure and values,
    before the root hairs, living masses and distances that post-growth derives)."""
    reset_choregrapher()
    Choregrapher().add_simulation_time_step(time_step)
    if before_post_growth:
        post_growth = RootGrowthModel.post_growth_updating
        RootGrowthModel.post_growth_updating = lambda self, *args, **kwargs: None
        try:
            model = RootGrowthModel(time_step_in_seconds=time_step, **(parameters or {}))
        finally:
            RootGrowthModel.post_growth_updating = post_growth
    else:
        model = RootGrowthModel(time_step_in_seconds=time_step, **(parameters or {}))
    model.pullable_inputs = {}
    model.props.setdefault("root_hairs_volume", {})
    return model


def legacy_run(parameters=None, n_steps=24, C_hexose_root=1e-3, time_step=3600, record_every=1, record=True):
    """*n_steps* steps of the legacy growth component; the snapshots every *record_every* steps (and the last)."""
    model = legacy_model(parameters, time_step)
    run = LegacyRun(model=model)
    if record:
        run.steps.append(snapshot(model))
        run.recorded_steps.append(0)
    for step in range(n_steps):
        write_C_hexose_root(model, step, C_hexose_root)
        model(modules_to_update=[], soil_boundaries_to_infer=[])
        if record and ((step + 1) % record_every == 0 or step == n_steps - 1):
            run.steps.append(snapshot(model))
            run.recorded_steps.append(step + 1)
    return run


STAGES = ("potential", "actual", "segmentation", "post_growth")


def _run_stage(model, stage):
    if stage == "potential":
        model.potential_growth()
    elif stage == "actual":
        model.actual_growth_and_corresponding_respiration()
    elif stage == "segmentation":
        model.segmentation_and_primordia_formation()
    else:
        model.post_growth_updating(modules_to_update=[], soil_boundaries_to_infer=[])


def legacy_stages(parameters=None, n_steps=240, C_hexose_root=1e-3, time_step=3600, every=12, first=0):
    """
    The legacy run step by step as its Choregrapher and __call__ do (potential_growth, actual growth, segmentation,
    post_growth_updating), recorded around each stage every *every* steps from step *first*: a list of
    (step, {"start": Snapshot, "potential": ..., "actual": ..., "segmentation": ..., "post_growth": ...}), each
    record after that stage. The stages of the port are compared on these (devplan §14, state transplant).
    """
    model = legacy_model(parameters, time_step)
    out = []
    for step in range(n_steps):
        write_C_hexose_root(model, step, C_hexose_root)
        model.pull_available_inputs()
        recorded = step >= first and (step - first) % every == 0
        records = {"start": snapshot(model)} if recorded else None
        for stage in STAGES:
            _run_stage(model, stage)
            if recorded:
                records[stage] = snapshot(model)
        if recorded:
            out.append((step, records))
    return out
