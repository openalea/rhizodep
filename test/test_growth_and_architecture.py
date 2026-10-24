"""RootGrowth against the architecture it specialises (openalea.archisimple's RootArchitecture)."""
import numpy as np
import pytest

from openalea.archisimple.architecture import RootArchitecture
from openalea.archisimple.drawing import totals
from openalea.rhizodep.comparison.parameter_sets import WHEAT_SOISSON
from openalea.rhizodep.comparison.port_run import port_run
from openalea.rhizodep.growth import RootGrowth


def test_as_root_growth_with_hexose_in_excess():
    """RootArchitecture against RootGrowth with hexose in excess: about the same root system (RootGrowth's laterals
    emerge one at a time per bearer, its thickening at a limited rate), over seeds."""
    architecture, growth = [], []
    for seed in (1, 2, 3):
        parameters = dict(WHEAT_SOISSON, random_choice=seed)
        architecture.append(totals(port_run({k: v for k, v in parameters.items()
                                             if k in RootArchitecture.__dataclass_fields__},
                                            n_steps=120, C_hexose_root=None, record_every=120,
                                            component=RootArchitecture).steps[-1])["length"])
        growth.append(totals(port_run(parameters, n_steps=120, C_hexose_root=1., record_every=120,
                                      component=RootGrowth).steps[-1])["length"])
    assert np.mean(growth) == pytest.approx(np.mean(architecture), rel=0.05)
