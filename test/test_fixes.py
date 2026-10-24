"""The reviewed fixes of the legacy (devplan §25): each option's fixed behaviour, against its legacy one (the LEGACY
preset, bitwise the legacy: test/comparison)."""
import numpy as np
import pytest

from openalea.rhizodep.architecture import (TYPE_ADVENTITIOUS_SUPPORT, TYPE_NODULE, TYPE_SEMINAL_BEFORE_EMERGENCE,
                                            TYPE_SEMINAL_SUPPORT, RootArchitecture)
from openalea.rhizodep.comparison.legacy_run import legacy_stages
from openalea.rhizodep.comparison.parameter_sets import GB_SOIL_1_0
from openalea.rhizodep.comparison.port_run import port_run, port_scene, run_stage, snapshot, transplant_scene
from openalea.rhizodep.growth import RootGrowth
from openalea.rhizodep.legacy import RootGrowthModel


def seminal_radii(record):
    return {record.props["radius"][v] for v in record.vertices
            if record.props["type"][v] == TYPE_SEMINAL_BEFORE_EMERGENCE}


def test_g15_each_seminal_root_its_own_draws():
    legacy = port_run(GB_SOIL_1_0, n_steps=0).steps[0]
    fixed = port_run(GB_SOIL_1_0, n_steps=0, legacy=False).steps[0]
    assert len(seminal_radii(legacy)) == 1                     # every seminal root the same draws
    assert len(seminal_radii(fixed)) == GB_SOIL_1_0["n_seminal_roots"] - 1


def test_g4_exclusive_duration_classes():
    scene = port_scene(dict(GB_SOIL_1_0, GD_by_frequency=True))
    growth = scene.populations[0].instance.growth
    p = {name: getattr(RootGrowth, name) for name in ("GD_highest", "GD_high", "GD_medium", "GD_low", "GD_prob_low",
                                                      "GD_prob_medium", "Dmin")}
    p.update(random_choice=1, GD_low=GB_SOIL_1_0["GD_low"], GD_medium=GB_SOIL_1_0["GD_medium"])
    lows = 0
    for legacy_id in range(1, 400):
        stream = np.random.RandomState()
        stream.seed(legacy_id)
        if stream.random_sample() < p["GD_prob_low"]:          # a draw in the low class
            lows += 1
            for legacy, interval in ((True, (p["GD_low"], p["GD_medium"])), (False, (0., p["GD_low"]))):
                growth.legacy_growth_duration_draws = legacy
                duration = growth._growth_duration(1e-4, legacy_id, 2, p, np.random.RandomState())
                assert interval[0] <= duration <= interval[1]
    assert lows > 50


def test_g1_segmentation_conserves_the_shared_amounts():
    """With the mass after growth, the segments of a split apex share its amounts exactly (with the legacy's mass
    before growth, the fractions sum to more than 1)."""
    (_, records), = legacy_stages(GB_SOIL_1_0, n_steps=97, every=96, first=96)
    before = records["actual"]
    sums = {}
    for legacy in (True, False):
        scene = transplant_scene(before, dict(GB_SOIL_1_0, legacy_segmentation_masses=legacy))
        run_stage(scene, "segmentation")
        after = snapshot(scene)
        new = set(after.vertices) - set(before.vertices)
        chains = {}
        for v in sorted(new):                                  # each new element to the apex it was split from
            root = v
            while after.parent[root] in new:
                root = after.parent[root]
            if after.edge_type[root] == "<" and after.parent[root] in before.vertices:
                chains.setdefault(after.parent[root], []).append(v)
        name = "hexose_consumption_by_growth_amount"
        sums[legacy] = [(before.props[name][apex], after.props[name][apex] + sum(after.props[name][v] for v in vs))
                        for apex, vs in chains.items() if before.props[name][apex] > 0]
    assert sums[False] and all(total == pytest.approx(amount, rel=1e-12) for amount, total in sums[False])
    assert any(total > amount * (1 + 1e-6) for amount, total in sums[True])


def test_g11_supports_keep_their_type():
    """The primary root stops within the run (no extender on its duration): its supports take its stop in the
    legacy, keep their type fixed."""
    parameters = dict(GB_SOIL_1_0, LDs=RootGrowthModel.LDs / 200, main_roots_growth_extender=1)
    for legacy, changes in ((True, True), (False, False)):
        last = port_run(dict(parameters, legacy_support_development=legacy), n_steps=240, record_every=240).steps[-1]
        supports_legacy_ids = [v for v in port_run(parameters, n_steps=0).steps[0].vertices
                               if port_run(parameters, n_steps=0).steps[0].props["type"][v]
                               in (TYPE_SEMINAL_SUPPORT, TYPE_ADVENTITIOUS_SUPPORT)][:3]
        kept = all(last.props["type"][v] in (TYPE_SEMINAL_SUPPORT, TYPE_ADVENTITIOUS_SUPPORT)
                   for v in supports_legacy_ids)
        assert kept != changes


def test_g6_actual_time_at_zero_temperature():
    scene = port_scene(GB_SOIL_1_0, legacy=False)
    growth = scene.populations[0].instance.growth
    assert growth._actual_time(np.array([3600.]), np.array([0.]))[0] == 0.
    assert growth._actual_time(np.array([3600.]), np.array([0.5]))[0] == 7200.


def test_g5_nodules_thicken_to_their_volume():
    """A nodule whose hexose allows less than its potential radius: the legacy's constant radius (0.62 m) never fits,
    so it takes its potential radius; fixed, it takes the sphere of the volume its hexose allows."""
    scene = port_scene(GB_SOIL_1_0)
    growth = scene.populations[0].instance.growth
    radius, potential = np.array([1e-4]), np.array([2e-4])
    volume = 4 / 3. * np.pi * radius ** 3
    density = np.array([1.4e5])
    # hexose for a tenth of the potential volume increase
    increase = 0.1 * 4 / 3. * np.pi * (potential ** 3 - radius ** 3)
    available = {"C_hexose_root": increase * density * RootGrowth.struct_mass_C_content / RootGrowth.yield_growth / 6.}
    reached = {}
    for legacy in (True, False):
        growth.legacy_nodule_radius = legacy
        reached[legacy], _ = growth.thickening(np.array([TYPE_NODULE]), radius, np.array([1e-4]), potential, density,
                                               available)
    assert reached[True][0] == potential[0]
    expected = (3 * (volume + increase) / (4 * np.pi)) ** (1 / 3.)
    assert reached[False][0] == pytest.approx(expected[0], rel=1e-9)
    assert reached[False][0] < potential[0]


def test_g10_growth_duration_rule_without_c_free_growth():
    """growth_duration_rule = "simple": the laterals' durations are GDs·d² (no extender, g18), and elongation stays
    limited by hexose (simple_growth_duration, the legacy's merged switch, would make it C-free)."""
    parameters = dict(GB_SOIL_1_0, growth_duration_rule="simple")
    last = port_run(parameters, n_steps=120, C_hexose_root=1e-3, record_every=120, legacy=False).steps[-1]
    laterals = [v for v in last.vertices if last.props["root_order"][v] > 1 and last.props["label"][v] == 2]
    assert laterals
    for v in laterals[:20]:
        diameter = 2. * last.props["original_radius"][v]
        assert last.props["growth_duration"][v] == pytest.approx(GB_SOIL_1_0["GDs"] * diameter ** 2, rel=1e-9)  # g18
    limited = port_run(parameters, n_steps=120, C_hexose_root=2e-5, record_every=120, legacy=False).steps[-1]
    c_free = port_run(dict(parameters, simple_growth_duration=True), n_steps=120, C_hexose_root=2e-5,
                      record_every=120, legacy=False).steps[-1]
    length = sum(limited.props["length"].values())
    assert length < sum(c_free.props["length"].values())        # the hexose limits elongation


def test_g18_laterals_without_the_extender():
    """With the simple duration rule, the laterals' durations are GDs·d² (determinate laterals), not × the extender."""
    parameters = dict(GB_SOIL_1_0, growth_duration_rule="simple")
    last = port_run(parameters, n_steps=120, record_every=120, legacy=False).steps[-1]
    laterals = [v for v in last.vertices if last.props["root_order"][v] > 1 and last.props["label"][v] == 2]
    assert laterals
    for v in laterals[:20]:
        diameter = 2. * last.props["original_radius"][v]
        assert last.props["growth_duration"][v] == pytest.approx(GB_SOIL_1_0["GDs"] * diameter ** 2, rel=1e-9)


@pytest.mark.parametrize("fixed", [True, False])
def test_g19_g20_thin_laterals(fixed):
    """Laterals form when their diameter (not radius) reaches Dmin (g19), and a lateral too thin uses its branching
    position (g20): fewer laterals on thin mothers than the legacy's retries."""
    options = dict(legacy_lateral_diameter_threshold=not fixed, legacy_branching_positions=not fixed)
    parameters = dict(GB_SOIL_1_0, D_ini=2.4e-4, Dmin=1.6e-4, CVDD=0.3, **options)
    last = port_run(parameters, n_steps=168, record_every=168, legacy=False).steps[-1]
    radii = [last.props["original_radius"][v] for v in last.vertices if last.props["root_order"][v] > 1
             and last.props["label"][v] == 2]
    if fixed:
        assert all(2 * r >= parameters["Dmin"] for r in radii)
    else:
        assert all(r >= parameters["Dmin"] for r in radii)


def test_g21_life_duration_exponent():
    scene = port_scene(dict(GB_SOIL_1_0, life_duration_diameter_exponent=2.), legacy=False)
    ds = scene.populations[0].data_structure
    radius, life = np.asarray(ds.get("radius")), np.asarray(ds.get("life_duration"))
    base = 0
    assert life[base] == pytest.approx(GB_SOIL_1_0["LDs"] * (2 * radius[base]) ** 2 * RootGrowth.new_root_tissue_density,
                                       rel=1e-12)


def test_g22_the_dead_have_no_living_mass():
    parameters = dict(GB_SOIL_1_0, LDs=RootGrowthModel.LDs / 200, main_roots_growth_extender=1)
    for legacy in (True, False):
        last = port_run(dict(parameters, legacy_dead_living_mass=legacy), n_steps=240, record_every=240,
                        legacy=False).steps[-1]
        dead = [v for v in last.vertices if last.props["type"][v] in (10, 11) and last.props["struct_mass"][v] > 0]
        assert dead
        assert all((last.props["living_struct_mass"][v] > 0) == legacy for v in dead)
