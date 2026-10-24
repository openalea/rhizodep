"""RootArchitecture, the C-free root architecture (ArchiSimple), by its properties (devplan §23, D-P7c)."""
import numpy as np
import pytest

from openalea.rhizodep.architecture import RootArchitecture, TYPE_BEFORE_EMERGENCE, TYPE_EMERGED
from openalea.rhizodep.comparison.parameter_sets import WHEAT_SOISSON
from openalea.rhizodep.comparison.drawing import totals
from openalea.rhizodep.comparison.port_run import port_run, port_scene, run_stage
from openalea.rhizodep.growth import RootGrowth

# wheat_soisson's parameters the architecture has (the others are its growth's: resources, root hairs)
ARCHITECTURE_GB = {name: value for name, value in WHEAT_SOISSON.items() if name in RootArchitecture.__dataclass_fields__}


def stepped(scene, n_steps, struct_mass_input=None):
    """*n_steps* steps of a RootArchitecture scene, stage by stage; the arrays after the potential and after the
    actual growth of the last step."""
    ds = scene.populations[0].data_structure
    for step in range(n_steps):
        if struct_mass_input is not None:
            ds.set("struct_mass_input", np.full(len(ds.entity_ids("Plant")), struct_mass_input))
        for stage in ("potential", "actual", "segmentation", "post_growth"):
            run_stage(scene, stage)
            if step == n_steps - 1 and stage == "potential":
                potential = {name: np.array(ds.get(name)) for name in (
                    "length", "radius", "initial_length", "initial_radius", "potential_length", "potential_radius",
                    "struct_mass", "volume", "root_tissue_density", "type")}
            if step == n_steps - 1 and stage == "actual":
                actual = {name: np.array(ds.get(name)) for name in ("length", "radius", "struct_mass_produced")}
                return potential, actual


@pytest.fixture(scope="module")
def unlimited():
    return stepped(port_scene(ARCHITECTURE_GB, component=RootArchitecture), 72)


def test_unlimited_growth_is_the_potential_one(unlimited):
    potential, actual = unlimited
    grows = potential["potential_length"] > potential["initial_length"]
    assert grows.any()
    np.testing.assert_allclose(actual["length"][grows], potential["potential_length"][grows], rtol=1e-14)


def test_half_satisfied_growth_takes_half_the_increments():
    scene = port_scene(ARCHITECTURE_GB, component=RootArchitecture)
    stepped(scene, 71)
    ds = scene.populations[0].data_structure
    run_stage(scene, "potential")
    growth = scene.populations[0].instance.growth
    arrays = {name: np.array(ds.get(name)) for name in ("type", "initial_radius", "initial_length", "potential_radius",
                                                         "potential_length", "root_tissue_density", "length",
                                                         "volume", "root_tissue_density")}
    demand = np.sum((growth._volume(arrays["type"], arrays["potential_radius"], arrays["potential_length"])
                     - growth._volume(arrays["type"], arrays["initial_radius"], arrays["initial_length"]))
                    * arrays["root_tissue_density"])
    ds.set("struct_mass_input", np.full(1, demand / 2.))
    assert growth.satisfaction_coefficient(
        arrays["type"], arrays["initial_radius"], arrays["initial_length"], arrays["potential_radius"],
        arrays["potential_length"], arrays["root_tissue_density"])[0] == pytest.approx(0.5, rel=1e-12)
    run_stage(scene, "actual")
    increments = np.array(ds.get("length")) - arrays["initial_length"]
    expected = (arrays["potential_length"] - arrays["initial_length"]) * 0.5
    grows = expected > 0
    np.testing.assert_allclose(increments[grows], expected[grows], rtol=1e-9)
    assert np.sum(np.array(ds.get("struct_mass_produced"))) == pytest.approx(demand / 2., rel=1e-9)


def test_the_root_system_develops():
    run = port_run(ARCHITECTURE_GB, n_steps=240, C_hexose_root=None, record_every=240, component=RootArchitecture)
    first, last = run.steps[0], run.steps[-1]
    assert len(last.vertices) > 3 * len(first.vertices)                    # segmentation
    laterals = [v for v in last.vertices if last.props["root_order"][v] > 1]
    assert laterals                                                         # primordia
    assert any(last.props["type"][v] == TYPE_EMERGED for v in laterals)     # emergences


def test_as_root_growth_with_hexose_in_excess():
    """RootArchitecture against RootGrowth with hexose in excess: about the same root system (RootGrowth's laterals
    emerge one at a time per bearer, its thickening at a limited rate), over seeds."""
    architecture, growth = [], []
    for seed in (1, 2, 3):
        parameters = dict(WHEAT_SOISSON, random_choice=seed)
        architecture.append(totals(port_run({k: v for k, v in parameters.items()
                                             if k in RootArchitecture.__dataclass_fields__},
                                            n_steps=120, C_hexose_root=None, record_every=120,
                                            component=RootArchitecture).steps[-1])["length"])
        growth.append(totals(port_run(parameters, n_steps=120, C_hexose_root=1., record_every=120,
                                      component=RootGrowth).steps[-1])["length"])
    assert np.mean(growth) == pytest.approx(np.mean(architecture), rel=0.05)
