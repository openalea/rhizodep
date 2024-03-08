import pickle

from rhizodep.rhizodep import Model


def test_small_run():
    rhizodep = Model(time_step=3600, random=False)

    mtg = rhizodep.g

    for step in range(10):
        assert rhizodep.root_carbon.C_hexose_soil == rhizodep.soil.C_hexose_soil == mtg.properties()["C_hexose_soil"]
        rhizodep.run()
