import pickle

from rhizodep.rhizodep import Model


def test_small_run():
    rhizodep = Model(time_step=3600, random=False)

    for step in range(20):
        rhizodep.run()
