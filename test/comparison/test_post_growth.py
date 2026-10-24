"""The port's post-growth (distance from tip, root hairs, living mass, the plant's total) against the legacy's
post_growth_updating standalone, on the legacy's states (devplan §14, P5)."""
import pytest

from openalea.rhizodep.comparison.parameter_sets import WHEAT_SOISSON, variants

from stages import COVERAGE, assert_stage_exact, varying_C


@pytest.mark.parametrize("name", sorted(variants()))
def test_parameter_sets(name):
    assert_stage_exact(variants()[name], "post_growth", every=24)


@pytest.mark.parametrize("simple", [False, True])
def test_stops_deaths_nodules(simple):
    assert_stage_exact(dict(COVERAGE, simple_growth_duration=simple), "post_growth", every=6)


def test_hexose_varying():
    assert_stage_exact(WHEAT_SOISSON, "post_growth", every=12, C_hexose_root=varying_C)
