import pickle

from rhizodep.rhizodep import Model


class TestRun:
    rhizodep = Model(time_step=3600, random=False)

    def test_soil(self):
        self.rhizodep.soil.run_exchanges_and_balance()

    def test_carbon(self):
        self.rhizodep.root_carbon.run_exchanges_and_balance()

    def test_growth(self):
        self.rhizodep.root_growth.run_time_step_growth()

    def test_anatomy(self):
        self.rhizodep.root_anatomy.run_actualize_anatomy()

