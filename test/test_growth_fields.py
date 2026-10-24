"""The growth component's declarations; the seedling's structure is the architecture's,
tested in openalea.archisimple."""
from dataclasses import fields

from openalea.rhizodep.comparison.legacy_run import legacy_run
from openalea.rhizodep.comparison.parameter_sets import WHEAT_SOISSON
from openalea.rhizodep.comparison.port_run import LEFT_OUT
from openalea.rhizodep.growth import RootGrowth

DECLARED = {f.name for f in fields(RootGrowth) if "variable_type" in f.metadata}


def test_every_legacy_property_is_declared():
    """Each property the legacy writes over 10 days is a declared variable or parameter of the port, the MPG label,
    or left out with a reason (the legacy's numbering: the MPG's ids and topology)."""
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
