"""
The N subclass of RootGrowth's hooks (nitrogen_growth.RootGrowthN) against Root_BRIDGES' RootGrowthModelCoupled
itself, bit for bit (devplan §21, D-P6b): each stage on Root_BRIDGES' own states, and whole runs. Root_BRIDGES runs
in the Wheat-BRIDGES environment (rootbridges_run.py); skipped when it is not there.
"""
import pytest

from openalea.rhizodep.comparison.differences import differences
from openalea.rhizodep.comparison.parameter_sets import GB_SOIL_1_0
from openalea.rhizodep.comparison.port_run import LEFT_OUT, port_run, run_stage, snapshot, transplant_scene
from openalea.rhizodep.legacy import RootGrowthModel

from given import varying_AA, varying_C
from nitrogen_growth import RootGrowthN
from rootbridges import available, rootbridges_stages

pytestmark = pytest.mark.skipif(not available(), reason="the Wheat-BRIDGES environment (Root_BRIDGES) is not there")

PREVIOUS = {"potential": "start", "actual": "potential", "segmentation": "actual", "post_growth": "segmentation"}
# Root_BRIDGES' properties the port does not have: the other models' variables its segmentation copies, the shoot's
# input of the synchronised emergence, its supply zone's bookkeeping
NOT_PORTED = {"C_sucrose_root", "C_hexose_reserve", "phloem_AA", "xylem_AA", "storage_protein",
              "adventitious_to_emerge", "list_of_elongation_supporting_elements_amino_acids"}
COVERAGE = dict(SGC=1.0, LDs=RootGrowthModel.LDs / 200, nodules=True, nodule_formation_probability=0.5)
CASES = {
    "defaults": ({}, 1e-3, 1e-3),
    "GB_soil_1.0": (GB_SOIL_1_0, 1e-3, 1e-3),
    "GB_soil_1.0_simple": (dict(GB_SOIL_1_0, simple_growth_duration=True), 1e-3, 1e-3),
    "coverage": (COVERAGE, 1e-3, 1e-3),
    "nitrogen_limited": (GB_SOIL_1_0, 1e-3, 2e-5),
    "varying": (COVERAGE, varying_C, varying_AA),
}


def compared(record):
    return [name for name in record.props if name not in LEFT_OUT and name not in NOT_PORTED]


@pytest.mark.parametrize("stage", list(PREVIOUS))
@pytest.mark.parametrize("case", sorted(CASES))
def test_stages(case, stage):
    parameters, C, AA = CASES[case]
    differing = []
    for step, records in rootbridges_stages(parameters, n_steps=240, C_hexose_root=C, AA=AA, every=12):
        scene = transplant_scene(records[PREVIOUS[stage]], parameters, component=RootGrowthN)
        run_stage(scene, stage)
        diff = differences(records[stage], snapshot(scene), names=compared(records[stage]))
        if not diff.identical:
            differing.append((step, diff.summary()))
    assert not differing, differing[:3]


@pytest.mark.parametrize("case", sorted(CASES))
def test_whole_runs(case):
    parameters, C, AA = CASES[case]
    reference = rootbridges_stages(parameters, n_steps=240, C_hexose_root=C, AA=AA, every=24, first=23)
    port = port_run(parameters, n_steps=240, C_hexose_root=C, record_every=24, component=RootGrowthN,
                    given={"AA": AA})
    assert len(reference) == len(port.steps) - 1
    for (step, records), state in zip(reference, port.steps[1:]):
        diff = differences(records["post_growth"], state, names=compared(records["post_growth"]))
        assert diff.identical, (step + 1, diff.summary())
    assert len(port.steps[-1].vertices) > len(port.steps[0].vertices)


# ---------------------------------------------------------------- the nodal roots synchronised with the shoot

SYNCHRONISED = dict(GB_SOIL_1_0, synchronize_adventitious_emergence=True, n_adventitious_roots=12)
SHOOT_DELAYS = [1.5e5 + 6e4 * k for k in range(8)]          # the shoot's next nodal emergences (thermal time, s)


def test_synchronised_emergence_stages():
    differing = []
    for step, records in rootbridges_stages(SYNCHRONISED, n_steps=240, every=6,
                                            adventitious_to_emerge=SHOOT_DELAYS):
        start = records["start"]
        scene = transplant_scene(start, SYNCHRONISED, component=RootGrowthN)
        plant = int(scene.populations[0].data_structure.entity_ids("Plant")[0])
        scene.populations[0].instance.growth.nodal_emergence_delays = {
            plant: list(start.props["adventitious_to_emerge"][1])}
        run_stage(scene, "potential")
        diff = differences(records["potential"], snapshot(scene), names=compared(records["potential"]))
        if not diff.identical:
            differing.append((step, diff.summary()))
    assert not differing, differing[:3]


def test_synchronised_emergence_whole_run():
    reference = rootbridges_stages(SYNCHRONISED, n_steps=240, every=24, first=23,
                                   adventitious_to_emerge=SHOOT_DELAYS)

    def setup(scene):
        plant = int(scene.populations[0].data_structure.entity_ids("Plant")[0])
        scene.populations[0].instance.growth.nodal_emergence_delays = {plant: list(SHOOT_DELAYS)}

    port = port_run(SYNCHRONISED, n_steps=240, record_every=24, component=RootGrowthN, given={"AA": 1e-3},
                    setup=setup)
    for (step, records), state in zip(reference, port.steps[1:]):
        diff = differences(records["post_growth"], state, names=compared(records["post_growth"]))
        assert diff.identical, (step + 1, diff.summary())
    delays = port.scene.populations[0].instance.growth.nodal_emergence_delays
    assert len(next(iter(delays.values()))) < len(SHOOT_DELAYS)        # some emergences were synchronised
