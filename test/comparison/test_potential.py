"""The port's potential growth against the legacy's, on the legacy's states (devplan §14, P2)."""
import numpy as np
import pytest

from openalea.rhizodep.comparison.legacy_run import legacy_stages
from openalea.rhizodep.comparison.parameter_sets import WHEAT_SOISSON, variants
from openalea.rhizodep.comparison.port_run import run_stage, snapshot, transplant_scene

from stages import COVERAGE, assert_stage_exact, varying_C


@pytest.mark.parametrize("name", sorted(variants()))
def test_parameter_sets(name):
    assert_stage_exact(variants()[name], "potential", every=24)


@pytest.mark.parametrize("simple", [False, True])
def test_stops_deaths_nodules(simple):
    assert_stage_exact(dict(COVERAGE, simple_growth_duration=simple), "potential", every=6)


@pytest.mark.parametrize("C", [2e-5, varying_C], ids=["low", "varying"])
@pytest.mark.parametrize("parameters", [WHEAT_SOISSON, COVERAGE], ids=["wheat_soisson", "coverage"])
def test_hexose_limited(parameters, C):
    assert_stage_exact(parameters, "potential", every=12, C_hexose_root=C)


def test_pipe_model_without_the_legacy_bug():
    """legacy_pipe_model=False (g16 fixed): a segment's theoretical radius from its successor's and its laterals'
    sections, sqrt(r_son² + SGC Σ r_lateral²)."""
    parameters = dict(COVERAGE, nodules=False)
    (_, records), = legacy_stages(parameters, n_steps=118, every=117, first=117)
    scene = transplant_scene(records["start"], dict(parameters, legacy_pipe_model=False))
    run_stage(scene, "potential")
    port = snapshot(scene)
    start = records["start"]
    bearers = [v for v in start.vertices if start.props["label"][v] == 1 and start.props["length"][v] > 0
               and any(start.parent[k] == v and start.edge_type[k] == "+" and start.props["length"][k] > 0
                       for k in start.vertices)]
    assert bearers
    v = bearers[0]
    son = [k for k in start.vertices if start.parent[k] == v and start.edge_type[k] == "<"]
    laterals = [k for k in start.vertices if start.parent[k] == v and start.edge_type[k] == "+"
                and start.props["length"][k] > 0]
    expected = np.sqrt(sum(port.props["theoretical_radius"][k] ** 2 for k in son)
                       + sum(port.props["theoretical_radius"][k] ** 2 for k in laterals))
    assert port.props["theoretical_radius"][v] == pytest.approx(expected, rel=1e-12)
    assert port.props["theoretical_radius"][v] > start.props["radius"][v]
