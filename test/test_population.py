"""A population in one MPG: plants of different scenarios grown together, each identical to the same
plant grown alone, bit for bit: a plant's development and draws do not depend on its neighbours. The plants of one population differ by their numeric parameters
(stored per plant); a switch (a boolean option) is the population's: another switch, another population."""
import pytest

from openalea.rhizodep.comparison.differences import differences
from openalea.rhizodep.comparison.legacy_run import Snapshot
from openalea.rhizodep.comparison.parameter_sets import WHEAT_SOISSON
from openalea.rhizodep.comparison.port_run import LEFT_OUT, port_run

PLANTS = [dict(WHEAT_SOISSON),
          dict(WHEAT_SOISSON, D_ini=5e-4, random_choice=3),
          dict(WHEAT_SOISSON, n_adventitious_roots=10, EL=4e-4, IPD=6e-3),
          dict(WHEAT_SOISSON, D_ini=9e-4, n_seminal_roots=3, CVDD=0.3)]


@pytest.fixture(scope="module")
def population():
    return port_run(PLANTS, n_steps=120, record_every=120)


def by_key(record):
    """The record keyed by the elements' lineage keys (their ids differ between a population and a plant alone)."""
    key = {v: int(record.props["key"][v]) for v in record.vertices}
    props = {name: {key[v]: value for v, value in values.items()} for name, values in record.props.items()}
    props["axis_apex_id"] = {key[v]: key.get(int(a), -1) for v, a in record.props["axis_apex_id"].items()}
    props["lateral_primordium_index"] = {key[v]: key.get(int(a), -1)
                                         for v, a in record.props["lateral_primordium_index"].items()}
    return Snapshot(vertices=sorted(key.values()), parent={key[v]: key.get(p) for v, p in record.parent.items()},
                    edge_type={key[v]: e for v, e in record.edge_type.items()}, props=props, plant=record.plant)


@pytest.mark.parametrize("plant", range(len(PLANTS)))
def test_each_plant_is_the_plant_alone(population, plant):
    alone = port_run(PLANTS[plant], n_steps=120, record_every=120).steps[-1]
    together = population.steps[-1][plant]
    alone, together = by_key(alone), by_key(together)
    diff = differences(alone, together, names=[name for name in alone.props if name not in LEFT_OUT])
    assert diff.identical, diff.summary()
    assert len(together.vertices) > 50
