"""Whole runs: the port's steps one after the other against the legacy's, every 6 h of 10 days (devplan §8, the
exactness of P1-P5 together: the numbering, the plant's stream across steps, the order of the stages)."""
import pytest

from openalea.rhizodep.comparison.differences import differences
from openalea.rhizodep.comparison.legacy_run import legacy_run
from openalea.rhizodep.comparison.parameter_sets import GB_SOIL_1_0, variants
from openalea.rhizodep.comparison.port_run import LEFT_OUT, port_run

from stages import COVERAGE, varying_C

CASES = dict(variants(), coverage=COVERAGE, coverage_not_random=dict(COVERAGE, random=False))


def assert_same_runs(parameters, n_steps=240, every=6, C_hexose_root=1e-3):
    legacy = legacy_run(parameters, n_steps=n_steps, record_every=every, C_hexose_root=C_hexose_root)
    port = port_run(parameters, n_steps=n_steps, record_every=every, C_hexose_root=C_hexose_root)
    assert port.recorded_steps == legacy.recorded_steps
    for step, a, b in zip(legacy.recorded_steps, legacy.steps, port.steps):
        diff = differences(a, b, names=[name for name in a.props if name not in LEFT_OUT])
        assert diff.identical, (step, diff.summary())
    return legacy, port


@pytest.mark.parametrize("name", sorted(CASES))
def test_ten_days(name):
    legacy, _ = assert_same_runs(CASES[name])
    assert len(legacy.steps[-1].vertices) > len(legacy.steps[0].vertices)


def test_hexose_varying():
    assert_same_runs(GB_SOIL_1_0, C_hexose_root=varying_C)
