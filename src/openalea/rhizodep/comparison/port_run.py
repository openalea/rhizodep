"""
The port run as the legacy is (legacy_run), its records in the legacy's form, to compare them and draw them
(devplan_metafspm.md): archisimple's run helpers (openalea.archisimple.run) with RootGrowth, its hexose given. The port
no longer reproduces the legacy bit for bit (devplan §29, D-X6: the tag port-exact does); the records compare
statistically.

    run = port_run(parameters, n_steps=48, C_hexose_root=1e-3)
    run.scene              the metafspm Scene (one plant: the component alone)
    run.steps[i]           records as legacy_run's, keyed by the elements' MPG ids: the SubOrgans, their parents through
                           the MPG's topology, every declared variable, decoded where the port codes it (the EMERGENCE_*
                           and AXIS_* codes as the legacy's strings), the MPG label as the legacy's label (1 / 2), and
                           the per-element copies of parameters (RootGrowth.ELEMENT_PARAMETERS)

    run_stage(scene, "potential")                    only that stage's steps of the port

LEFT_OUT: the legacy properties the port does not have, and why.
"""
from openalea.archisimple import run as archisimple_run
from openalea.archisimple.run import run_stage, write_given  # noqa: F401  (the comparisons' names)
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
DECODE = {"lateral_root_emergence_possibility": EMERGENCE, "axis_type": AXIS}


def port_scene(parameters=None, time_step=3600, component=RootGrowth):
    """A Scene of the port: one plant, or one per element of *parameters* when it is a list (a population in one
    MPG, each plant its own scenario)."""
    return archisimple_run.scene(parameters, time_step, component)


def snapshot(scene, plant=0):
    """The port's state in the legacy's form: the *plant*-th plant's (in the planting table's order)."""
    return archisimple_run.record(scene, plant, decode=DECODE)


def port_run(parameters=None, n_steps=24, C_hexose_root=1e-3, time_step=3600, record_every=1, record=True,
             component=RootGrowth, given=None, setup=None):
    """*n_steps* steps of the port; the records every *record_every* steps (and the last), as legacy_run's (for a
    population, *parameters* a list of the plants' scenarios: a list of records, one per plant). *C_hexose_root* and
    *given*: inputs written before each step (write_given); *setup*: a function of the scene, called once it is
    built."""
    inputs = {"C_hexose_root": C_hexose_root, **(given or {})}
    return archisimple_run.run(parameters, n_steps, time_step, record_every, record, component, inputs, setup,
                               decode=DECODE)


def write_C_hexose_root(scene, step, C_hexose_root):
    write_given(scene, step, "C_hexose_root", C_hexose_root)
