"""Invariants of the growth components (devplan §29, D-X2), whatever the order of their operations: the hexose growth
consumes is the structural mass it produces (of the elements and their root hairs) at the cost of its carbon and
yield; the times since events increase."""
import numpy as np
import pytest

from openalea.rhizodep.comparison.port_run import port_run
from openalea.rhizodep.growth import RootGrowth
from openalea.rhizodep.parameter_sets import WHEAT_SOISSON


@pytest.fixture(scope="module")
def run():
    return port_run(dict(WHEAT_SOISSON, nodules=False), n_steps=96, C_hexose_root=1e-3, record_every=1,
                    legacy=False)


def test_the_hexose_consumed_makes_the_mass_produced(run):
    cost = WHEAT_SOISSON["struct_mass_C_content"] / WHEAT_SOISSON["yield_growth"] / 6.      # the plant's own
    for record in run.steps[1:]:
        consumed = sum(record.props["hexose_consumption_by_growth_amount"].values())
        produced = (sum(record.props["struct_mass_produced"].values())
                    + sum(record.props["root_hairs_struct_mass_produced"].values()))
        assert consumed == pytest.approx(produced * cost, rel=1e-9, abs=1e-18)


def test_clocks_never_go_back(run):
    for before, after in zip(run.steps[:-1], run.steps[1:]):
        for name in ("thermal_time_since_primordium_formation", "actual_time_since_primordium_formation"):
            common = [v for v in before.vertices if v in after.vertices]
            assert all(after.props[name][v] >= before.props[name][v] for v in common), name


def test_convergence_with_the_time_step():
    """Halving the time step from 30 to 15 min changes the root system's totals by little (the means over seeds).
    The hourly step is not converged: it grows about 18 % less than finer ones over 5 days, its cause still to find
    (devplan §31)."""
    lengths = {}
    for time_step in (1800, 900):
        values = []
        for seed in (1, 2, 3):
            run = port_run(dict(WHEAT_SOISSON, random_choice=seed), n_steps=int(5 * 86400 / time_step),
                           time_step=time_step, record_every=int(5 * 86400 / time_step), legacy=False)
            values.append(sum(run.steps[-1].props["length"].values()))
        lengths[time_step] = np.mean(values)
    assert lengths[900] == pytest.approx(lengths[1800], rel=0.05)
