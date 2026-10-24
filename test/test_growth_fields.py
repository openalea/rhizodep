"""The growth component's declarations and the seedling's MPG structure (devplan_metafspm.md, §12)."""
from dataclasses import fields

import numpy as np
import pytest

from openalea.rhizodep.comparison.legacy_run import legacy_run
from openalea.rhizodep.comparison.parameter_sets import WHEAT_SOISSON
from openalea.rhizodep.comparison.port_run import LEFT_OUT, port_scene
from openalea.rhizodep.growth import TYPE_BASE, RootGrowth

DECLARED = {f.name for f in fields(RootGrowth) if "variable_type" in f.metadata}


def test_every_legacy_property_is_declared():
    """Each property the legacy writes over 10 days is a declared variable or parameter of the port, the MPG label,
    or left out with a reason (the legacy's numbering: the MPG's ids and topology, devplan §32.5)."""
    last = legacy_run(dict(WHEAT_SOISSON, simple_growth_duration=True), n_steps=240, record_every=240).steps[-1]
    missing = sorted(set(last.props) - DECLARED - set(LEFT_OUT) - {"label", "vertex_index", "parent_id"})
    assert missing == []


def test_kinds_and_scales():
    for f in fields(RootGrowth):
        meta = f.metadata
        if (meta.get("variable_type") in ("state_variable", "input")
                and f.name not in ("total_living_struct_mass", "struct_mass_input")):            # the Plant's
            assert meta["scale"] == RootGrowth.__dataclass_fields__["length"].metadata["scale"], f.name
        if meta.get("variable_type") == "state_variable":
            assert meta["state_variable_type"], f.name


@pytest.fixture(scope="module")
def seedling():
    parameters = dict(WHEAT_SOISSON, n_seminal_roots=3, n_adventitious_roots=4)
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


def test_branching_on_the_base(seedling):
    """Each seminal or nodal root's apex is a RootApex branched (+) on the base segment, in an Organ of its own; the
    primary's Organ is its `<` chain: the base, the apex (devplan §32.5)."""
    ds, p = seedling
    g = ds.mtg
    s, labels = g.scales, g.labels
    ids = np.asarray(ds.entity_ids("node")).tolist()
    types = dict(zip(ids, np.asarray(ds.get("type")).tolist()))
    base, = [v for v in ids if types[v] == TYPE_BASE]
    primary = g.complex_at_scale(base, s.Organ)
    branched = [c for c in g.children(base) if g.edge_type(c) == "+"]
    assert len(branched) == p["n_seminal_roots"] - 1 + p["n_adventitious_roots"]
    for apex in branched:
        assert g.label(apex) == labels.SubOrgan.RootApex
        assert g.complex_at_scale(apex, s.Organ) != primary
    chain = sorted(v for v in ids if g.complex_at_scale(v, s.Organ) == primary)
    assert [g.label(v) for v in chain] == [labels.SubOrgan.RootSegment, labels.SubOrgan.RootApex]


def test_keys_are_distinct(seedling):
    """Each element its own lineage key, and so its own draws (devplan §32.3)."""
    ds, _ = seedling
    keys = np.asarray(ds.get("key"))
    assert (keys >= 0).all() and np.unique(keys).size == keys.size
