"""The reviewed fixes of the legacy (devplan §25), now the components' only behaviour (D-X6; the tag port-exact keeps
the legacy's as options): each fix by its property."""
import numpy as np
import pytest

from openalea.metafspm.data_structure import random_streams
from openalea.rhizodep.architecture import TYPE_NODULE, TYPE_SEMINAL_BEFORE_EMERGENCE
from openalea.rhizodep.comparison.parameter_sets import WHEAT_SOISSON
from openalea.rhizodep.comparison.port_run import port_run, port_scene, run_stage, snapshot, write_C_hexose_root
from openalea.rhizodep.growth import RootGrowth

# The life duration of RootGrowth's defaults shortened, so that elements die within the runs
SHORT_LIVED = dict(WHEAT_SOISSON, LDs=RootGrowth.LDs / 200, main_roots_growth_extender=1)


def seminal_radii(record):
    return {record.props["radius"][v] for v in record.vertices
            if record.props["type"][v] == TYPE_SEMINAL_BEFORE_EMERGENCE}


def test_g15_each_seminal_root_its_own_draws():
    seedling = port_run(WHEAT_SOISSON, n_steps=0).steps[0]
    assert len(seminal_radii(seedling)) == WHEAT_SOISSON["n_seminal_roots"] - 1


def test_g4_exclusive_duration_classes():
    """A draw in the low class gives a duration in (0, GD_low): the classes are exclusive."""
    scene = port_scene(dict(WHEAT_SOISSON, GD_by_frequency=True))
    growth = scene.populations[0].instance.growth
    keys = np.arange(1, 400)
    p = {name: np.full(keys.size, float(getattr(RootGrowth, name))) for name in (
        "GD_highest", "GD_high", "GD_medium", "GD_low", "GD_prob_low", "GD_prob_medium", "Dmin")}
    p.update(GD_low=np.full(keys.size, WHEAT_SOISSON["GD_low"]), GD_medium=np.full(keys.size, WHEAT_SOISSON["GD_medium"]))
    durations = growth._growth_durations(np.full(keys.size, 1e-4), keys, np.full(keys.size, 2), p)
    low = random_streams.uniform(keys, stream="growth_duration_class") < p["GD_prob_low"]
    assert low.sum() > 50
    assert ((durations[low] >= 0.) & (durations[low] <= WHEAT_SOISSON["GD_low"])).all()
    assert (durations[~low] >= WHEAT_SOISSON["GD_low"]).all()


def test_g1_segmentation_conserves_the_shared_amounts():
    """The segments of a split apex share its amounts exactly (by their mass after growth)."""
    scene = port_scene(WHEAT_SOISSON)
    for step in range(97):
        write_C_hexose_root(scene, step, 1e-3)
        for stage in ("potential", "actual", "segmentation", "post_growth"):
            if step == 96 and stage == "segmentation":
                break
            run_stage(scene, stage)
    before = snapshot(scene)
    run_stage(scene, "segmentation")
    after = snapshot(scene)
    new = set(after.vertices) - set(before.vertices)
    chains = {}
    for v in sorted(new):                                      # each new element to the apex it was split from
        root = v
        while after.parent[root] in new:
            root = after.parent[root]
        if after.edge_type[root] == "<" and after.parent[root] in before.vertices:
            chains.setdefault(after.parent[root], []).append(v)
    name = "hexose_consumption_by_growth_amount"
    sums = [(before.props[name][apex], after.props[name][apex] + sum(after.props[name][v] for v in vs))
            for apex, vs in chains.items() if before.props[name][apex] > 0]
    assert sums and all(total == pytest.approx(amount, rel=1e-12) for amount, total in sums)


def test_g6_actual_time_at_zero_temperature():
    scene = port_scene(WHEAT_SOISSON)
    growth = scene.populations[0].instance.growth
    assert growth._actual_time(np.array([3600.]), np.array([0.]))[0] == 0.
    assert growth._actual_time(np.array([3600.]), np.array([0.5]))[0] == 7200.


def test_g5_nodules_thicken_to_their_volume():
    """A nodule whose hexose allows less than its potential radius takes the sphere of the volume its hexose
    allows."""
    scene = port_scene(WHEAT_SOISSON)
    growth = scene.populations[0].instance.growth
    radius, potential = np.array([1e-4]), np.array([2e-4])
    volume = 4 / 3. * np.pi * radius ** 3
    density = np.array([1.4e5])
    # hexose for a tenth of the potential volume increase
    increase = 0.1 * 4 / 3. * np.pi * (potential ** 3 - radius ** 3)
    available = {"C_hexose_root": increase * density * RootGrowth.struct_mass_C_content / RootGrowth.yield_growth / 6.}
    reached, _ = growth.thickening(np.array([TYPE_NODULE]), radius, np.array([1e-4]), potential, density, available)
    expected = (3 * (volume + increase) / (4 * np.pi)) ** (1 / 3.)
    assert reached[0] == pytest.approx(expected[0], rel=1e-9)
    assert reached[0] < potential[0]


def test_g10_g18_simple_duration_rule_stays_limited_by_hexose():
    """growth_duration_rule = "simple": the laterals' durations are GDs·d², without the main roots' extender (g18),
    and elongation stays limited by hexose (g10)."""
    parameters = dict(WHEAT_SOISSON, growth_duration_rule="simple")
    last = port_run(parameters, n_steps=120, C_hexose_root=1e-3, record_every=120).steps[-1]
    laterals = [v for v in last.vertices if last.props["root_order"][v] > 1 and last.props["label"][v] == 2]
    assert laterals
    for v in laterals[:20]:
        diameter = 2. * last.props["original_radius"][v]
        assert last.props["growth_duration"][v] == pytest.approx(WHEAT_SOISSON["GDs"] * diameter ** 2, rel=1e-9)
    limited = port_run(parameters, n_steps=120, C_hexose_root=2e-5, record_every=120).steps[-1]
    assert sum(limited.props["length"].values()) < sum(last.props["length"].values())


def test_g19_laterals_by_their_diameter():
    """Laterals form when their diameter (not radius) reaches Dmin (g19)."""
    parameters = dict(WHEAT_SOISSON, D_ini=2.4e-4, Dmin=1.6e-4, CVDD=0.3)
    last = port_run(parameters, n_steps=168, record_every=168).steps[-1]
    radii = [last.props["original_radius"][v] for v in last.vertices if last.props["root_order"][v] > 1
             and last.props["label"][v] == 2]
    assert radii
    assert all(2 * r >= parameters["Dmin"] for r in radii)


def test_g21_life_duration_exponent():
    scene = port_scene(dict(WHEAT_SOISSON, life_duration_diameter_exponent=2.))
    ds = scene.populations[0].data_structure
    radius, life = np.asarray(ds.get("radius")), np.asarray(ds.get("life_duration"))
    base = 0
    assert life[base] == pytest.approx(WHEAT_SOISSON["LDs"] * (2 * radius[base]) ** 2 * RootGrowth.new_root_tissue_density,
                                       rel=1e-12)


def test_g22_the_dead_have_no_living_mass():
    last = port_run(SHORT_LIVED, n_steps=240, record_every=240).steps[-1]
    dead = [v for v in last.vertices if last.props["type"][v] in (10, 11) and last.props["struct_mass"][v] > 0]
    assert dead
    assert all(last.props["living_struct_mass"][v] == 0 for v in dead)
