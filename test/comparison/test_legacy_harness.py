"""The legacy growth component run standalone (devplan_metafspm.md, P0): the reference the port is compared with."""
import copy

import pytest

from openalea.rhizodep.comparison.legacy_run import legacy_run
from openalea.rhizodep.comparison.parameter_sets import GB_SOIL_1_0, variants
from openalea.rhizodep.comparison.differences import differences
from openalea.rhizodep.comparison.drawing import positions, totals

N_STEPS = 72


@pytest.fixture(scope="module", params=sorted(variants()))
def pair(request):
    parameters = variants()[request.param]
    return (legacy_run(parameters, n_steps=N_STEPS, record_every=24),
            legacy_run(parameters, n_steps=N_STEPS, record_every=24))


def test_records(pair):
    run, _ = pair
    assert run.recorded_steps == [0, 24, 48, 72]
    first, last = run.steps[0], run.steps[-1]
    assert len(last.vertices) >= len(first.vertices)
    assert totals(last)["length"] > totals(first)["length"]
    # every element has a record of the properties it grows with (not vertex_index: the legacy numbers new vertices
    # only when coupled, in post_growth_updating's modules_to_update branch)
    for name in ("length", "radius", "struct_mass", "type", "label", "C_hexose_root"):
        assert set(last.props[name]) >= set(last.vertices), name


def test_repeatable(pair):
    """Two runs of a set are identical: the draws are seeded per vertex (devplan §5), and each model gets its own
    schedule (the Choregrapher reset)."""
    a, b = pair
    for sa, sb in zip(a.steps, b.steps):
        assert differences(sa, sb).identical


def test_differences_detects():
    run = legacy_run(GB_SOIL_1_0, n_steps=24, record_every=24)
    last = run.steps[-1]
    other = copy.deepcopy(last)
    vid = last.vertices[-1]
    other.props["length"][vid] *= 1 + 1e-12
    diff = differences(last, other)
    assert not diff.identical and diff.values["length"] == [vid]
    assert differences(last, other, rtol=1e-9).identical


def test_given_C_hexose_root_drives_growth():
    low = legacy_run(GB_SOIL_1_0, n_steps=N_STEPS, C_hexose_root=1e-5, record=True, record_every=N_STEPS)
    high = legacy_run(GB_SOIL_1_0, n_steps=N_STEPS, C_hexose_root=1e-3, record=True, record_every=N_STEPS)
    assert totals(low.steps[-1])["length"] < totals(high.steps[-1])["length"]


def test_positions():
    last = legacy_run(None, n_steps=24, record_every=24).steps[-1]
    xyz = positions(last)
    assert set(xyz) == set(last.vertices)
    for vid in last.vertices:
        parent = last.parent[vid]
        if parent is not None:
            assert (xyz[vid][0] == xyz[parent][1]).all()
