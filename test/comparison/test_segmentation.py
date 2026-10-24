"""The port's segmentation and primordia against the legacy's, on the legacy's states (devplan §14, P4)."""
import numpy as np
import pytest

from openalea.rhizodep.comparison.legacy_run import legacy_stages
from openalea.rhizodep.comparison.parameter_sets import WHEAT_SOISSON, variants
from openalea.rhizodep.comparison.port_run import run_stage, transplant_scene
from openalea.rhizodep.growth import TYPE_BEFORE_EMERGENCE, TYPE_NODULE

from stages import COVERAGE, assert_stage_exact, varying_C


@pytest.mark.parametrize("name", sorted(variants()))
def test_parameter_sets(name):
    assert_stage_exact(variants()[name], "segmentation", every=24)


@pytest.mark.parametrize("change", [{}, dict(simple_growth_duration=True), dict(random=False),
                                    dict(random=False, simple_growth_duration=True)],
                         ids=["coverage", "simple", "not_random", "not_random_simple"])
def test_primordia_and_nodules(change):
    """Lateral primordia, nodules; without random radii, a nodule's draw continues the legacy's stream from earlier
    seedings (the plant's stream, transplanted with the legacy's state)."""
    assert_stage_exact(dict(COVERAGE, **change), "segmentation", every=6)


def test_hexose_varying():
    assert_stage_exact(COVERAGE, "segmentation", every=12, C_hexose_root=varying_C)


def test_new_elements_in_the_mpg():
    """Segments extend their apex's Organ; a lateral primordium or a nodule is a new Organ of its bearer's Axis,
    branched (+); the former apex becomes a RootSegment, the new tip a RootApex."""
    for step, records in legacy_stages(COVERAGE, n_steps=240, every=6):
        before = records["actual"]
        scene = transplant_scene(before, COVERAGE)
        ds = scene.populations[0].data_structure
        old = set(np.asarray(ds.entity_ids("node")).tolist())
        run_stage(scene, "segmentation")
        g = ds.mtg
        s, labels = g.scales, g.labels
        types = dict(zip(np.asarray(ds.entity_ids("node")).tolist(), np.asarray(ds.get("type")).tolist()))
        new = [v for v in types if v not in old]
        laterals = [v for v in new if types[v] in (TYPE_BEFORE_EMERGENCE, TYPE_NODULE)]
        if not laterals or not any(types[v] == TYPE_NODULE for v in laterals):
            continue
        for v in new:
            bearer = g.parent(v)
            if v in laterals:
                assert g.edge_type(v) == "+"
                assert g.complex_at_scale(v, s.Organ) != g.complex_at_scale(bearer, s.Organ)
                assert g.complex_at_scale(v, s.Axis) == g.complex_at_scale(bearer, s.Axis)
            else:
                assert g.edge_type(v) == "<"
                assert g.complex_at_scale(v, s.Organ) == g.complex_at_scale(bearer, s.Organ)
                assert g.label(bearer) == labels.SubOrgan.RootSegment
        tips = [v for v in new if not g.children(v) and types[v] != TYPE_NODULE]
        assert tips and all(g.label(v) == labels.SubOrgan.RootApex for v in tips)
        return
    pytest.fail("no state with both a primordium and a nodule created")
