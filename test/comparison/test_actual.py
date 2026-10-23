"""The port's actual growth against the legacy's, on the legacy's states (devplan §14, P3)."""
import pytest

from openalea.rhizodep.comparison.parameter_sets import GB_SOIL_1_0, variants

from stages import COVERAGE, assert_stage_exact, varying_C


@pytest.mark.parametrize("name", sorted(variants()))
def test_parameter_sets(name):
    assert_stage_exact(variants()[name], "actual", every=24)


@pytest.mark.parametrize("simple", [False, True])
def test_stops_deaths_nodules(simple):
    assert_stage_exact(dict(COVERAGE, simple_growth_duration=simple), "actual", every=6)


@pytest.mark.parametrize("C", [2e-5, varying_C], ids=["low", "varying"])
@pytest.mark.parametrize("parameters", [GB_SOIL_1_0, COVERAGE], ids=["GB_soil_1.0", "coverage"])
def test_hexose_limited(parameters, C):
    """Most elongations limited by the supply zone's hexose; costs shared between unequal suppliers."""
    assert_stage_exact(parameters, "actual", every=12, C_hexose_root=C)
