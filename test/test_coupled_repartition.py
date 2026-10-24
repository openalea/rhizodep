"""Coupled, the other components' variables shared between the elements by metafspm's repartition, once per call
(devplan §19): conservative, unlike the legacy's (g17)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig
from openalea.metafspm.scene.scene import Scene
from openalea.rhizodep.comparison.parameter_sets import WHEAT_SOISSON
from openalea.rhizodep.comparison.port_run import write_C_hexose_root
from openalea.rhizodep.growth import RootGrowth

DOC = dict(unit="", unit_comment="", description="", min_value="", max_value="", value_comment="", references="",
           DOI="")


@dataclass
class Pools(FunctionalComponent):
    """Another model's variables, one of each kind."""
    C_x: float = state_variable(**DOC, initialize=1., scale=ScalesConfig.SubOrgan,
                                state_variable_type="massic_concentration")
    X: float = state_variable(**DOC, initialize=1., scale=ScalesConfig.SubOrgan, state_variable_type="extensive")
    T_x: float = state_variable(**DOC, initialize=12., scale=ScalesConfig.SubOrgan, state_variable_type="intensive")


class GrowthAndPools:
    initiators = (RootGrowth,)

    def __init__(self, data_structure, time_step):
        self.growth = RootGrowth(data_structure=data_structure, time_step=time_step)
        self.pools = Pools(data_structure=data_structure)
        self.components = [self.growth, self.pools]

    def run(self):
        self.growth()


@pytest.fixture(scope="module")
def run():
    table = pd.DataFrame([dict(plant="p0", model=GrowthAndPools, x=0., y=0., z=0., rotation=0.,
                               scenario={"parameters": dict(WHEAT_SOISSON)})])
    table.attrs.update(xrange=1., yrange=1.)
    scene = Scene(table, time_step=3600)
    ds = scene.populations[0].data_structure
    records = []
    for step in range(120):
        write_C_hexose_root(scene, step, 1e-3)
        before = (float(np.sum(ds.get("C_x") * ds.get("living_struct_mass"))), float(np.sum(ds.get("X"))),
                  ds.n_nodes())
        scene.run()
        after = (float(np.sum(ds.get("C_x") * ds.get("living_struct_mass"))), float(np.sum(ds.get("X"))),
                 ds.n_nodes())
        records.append((before, after))
    return scene, records


def test_amounts_are_conserved(run):
    _, records = run
    assert records[-1][1][2] > records[0][0][2]                         # the root system grew
    for (massic, extensive, _), (massic_after, extensive_after, _) in records:
        assert extensive_after == pytest.approx(extensive, rel=1e-12)
        if massic > 0:
            assert massic_after == pytest.approx(massic, rel=1e-12)


def test_new_elements_take_their_parents_intensive_values(run):
    scene, _ = run
    ds = scene.populations[0].data_structure
    np.testing.assert_array_equal(ds.get("T_x"), 12.)
