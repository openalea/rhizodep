"""Invariants of the growth components, whatever the order of their operations: the hexose growth
consumes is the structural mass it produces (of the elements and their root hairs) at the cost of its carbon and
yield; the times since events increase."""
import numpy as np
import pytest

from openalea.rhizodep.comparison.port_run import port_run
from openalea.rhizodep.growth import RootGrowth
from openalea.rhizodep.parameter_sets import WHEAT_SOISSON


@pytest.fixture(scope="module")
def run():
    return port_run(dict(WHEAT_SOISSON, nodules=False), n_steps=96, C_hexose_root=1e-3, record_every=1)


def test_the_hexose_consumed_makes_the_mass_produced(run):
    cost = WHEAT_SOISSON["struct_mass_C_content"] / WHEAT_SOISSON["yield_growth"] / 6.      # the plant's own
    for record in run.steps[1:]:
        consumed = sum(record.props["hexose_consumption_by_growth_amount"].values())
        produced = (sum(record.props["struct_mass_produced"].values())
                    + sum(record.props["root_hairs_struct_mass_produced"].values()))
        assert consumed == pytest.approx(produced * cost, rel=1e-9, abs=1e-18)


def test_convergence_with_the_time_step():
    """Halving the time step from 30 to 15 min changes the root system's totals by little (the means over seeds).
    The hourly step is not converged: it grows about 18 % less than finer ones over 5 days, its apices' growth capped
    by the hexose their supply zones hold at the step's start."""
    lengths = {}
    for time_step in (1800, 900):
        values = []
        for seed in (1, 2, 3):
            run = port_run(dict(WHEAT_SOISSON, random_choice=seed), n_steps=int(5 * 86400 / time_step),
                           time_step=time_step, record_every=int(5 * 86400 / time_step))
            values.append(sum(run.steps[-1].props["length"].values()))
        lengths[time_step] = np.mean(values)
    assert lengths[900] == pytest.approx(lengths[1800], rel=0.05)


def test_a_supply_rate_removes_most_of_the_time_step_dependence():
    """With the hexose its elements receive over the step (hexose_supply_rate), a supply zone's cap no longer
    depends on the time step: hourly and 15-min steps agree within 5 % (about 19 % apart without)."""
    lengths = {}
    for time_step in (3600, 900):
        n_steps = int(5 * 86400 / time_step)
        run = port_run(WHEAT_SOISSON, n_steps=n_steps, time_step=time_step, record_every=n_steps,
                       given={"hexose_supply_rate": 1e-7})
        lengths[time_step] = sum(run.steps[-1].props["length"].values())
    assert lengths[3600] == pytest.approx(lengths[900], rel=0.05)
