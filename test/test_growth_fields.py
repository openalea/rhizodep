"""The growth component's declarations and the seedling's MPG structure (devplan_metafspm.md, §12)."""
from dataclasses import fields

import numpy as np
import pytest

from openalea.rhizodep.comparison.legacy_run import legacy_run
from openalea.rhizodep.comparison.parameter_sets import GB_SOIL_1_0
from openalea.rhizodep.comparison.port_run import LEFT_OUT, port_scene
from openalea.rhizodep.growth import (TYPE_ADVENTITIOUS_SUPPORT, TYPE_SEMINAL_SUPPORT, RootGrowth)

DECLARED = {f.name for f in fields(RootGrowth) if "variable_type" in f.metadata}


def test_every_legacy_property_is_declared():
    """Each property the legacy writes over 10 days is a declared variable or parameter of the port, the MPG label,
    or left out with a reason."""
    last = legacy_run(dict(GB_SOIL_1_0, simple_growth_duration=True), n_steps=240, record_every=240).steps[-1]
    missing = sorted(set(last.props) - DECLARED - set(LEFT_OUT) - {"label"})
    assert missing == []


def test_kinds_and_scales():
    for f in fields(RootGrowth):
        meta = f.metadata
        if meta.get("variable_type") in ("state_variable", "input") and f.name != "total_living_struct_mass":
            assert meta["scale"] == RootGrowth.__dataclass_fields__["length"].metadata["scale"], f.name
        if meta.get("variable_type") == "state_variable":
            assert meta["state_variable_type"], f.name


@pytest.fixture(scope="module")
def seedling():
    parameters = dict(GB_SOIL_1_0, n_seminal_roots=3, n_adventitious_roots=4)
    scene = port_scene(parameters)
    return scene.populations[0].data_structure, parameters


def test_grouping(seedling):
    """One Axis (and Organ) per first-order root: the primary, each seminal, each nodal (§12)."""
    ds, p = seedling
    g = ds.mtg
    s = g.scales
    axes = [v for v in g.vertices(scale=s.Axis) if not g.property("isanchor").get(v)]
    organs = [v for v in g.vertices(scale=s.Organ) if not g.property("isanchor").get(v)]
    n_roots = 1 + (p["n_seminal_roots"] - 1) + p["n_adventitious_roots"]
    assert len(axes) == len(organs) == n_roots
    for axis in axes:
        assert len([o for o in organs if g.complex_at_scale(o, s.Axis) == axis]) == 1


def test_branching_on_the_supports(seedling):
    """Each seminal or nodal root's apex is a RootApex branched (+) on its zero-length support, in another Organ;
    the primary's Organ is its `<` chain: base, supports, apex."""
    ds, p = seedling
    g = ds.mtg
    s, labels = g.scales, g.labels
    ids = np.asarray(ds.entity_ids("node")).tolist()
    types = dict(zip(ids, np.asarray(ds.get("type")).tolist()))
    supports = [v for v in ids if types[v] in (TYPE_SEMINAL_SUPPORT, TYPE_ADVENTITIOUS_SUPPORT)]
    assert len(supports) == p["n_seminal_roots"] - 1 + p["n_adventitious_roots"]
    primary = g.complex_at_scale(supports[0], s.Organ)
    for support in supports:
        assert g.complex_at_scale(support, s.Organ) == primary
        branched = [c for c in g.children(support) if g.edge_type(c) == "+"]
        assert len(branched) == 1
        apex = branched[0]
        assert g.label(apex) == labels.SubOrgan.RootApex
        assert g.complex_at_scale(apex, s.Organ) != primary
    chain = [v for v in ids if g.complex_at_scale(v, s.Organ) == primary]
    assert [g.label(v) for v in chain].count(labels.SubOrgan.RootApex) == 1


def test_vertex_index_is_the_legacy_numbering(seedling):
    ds, _ = seedling
    index = np.asarray(ds.get("vertex_index"))
    assert sorted(index.tolist()) == list(range(1, index.size + 1))
