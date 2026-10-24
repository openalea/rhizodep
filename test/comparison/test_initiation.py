"""The port's seedling against the legacy's (devplan_metafspm.md, P1): the topology, every initial value, the plant's
total, for the parameter sets and the initiation's edge cases."""
import pandas as pd
import pytest

from openalea.rhizodep.comparison.differences import differences
from openalea.rhizodep.comparison.legacy_run import legacy_run
from openalea.rhizodep.comparison.parameter_sets import WHEAT_SOISSON, variants
from openalea.rhizodep.comparison.port_run import LEFT_OUT, port_run


def assert_same_seedling(parameters):
    legacy = legacy_run(parameters, n_steps=0).steps[0]
    port = port_run(parameters, n_steps=0).steps[0]
    names = [name for name in legacy.props if name not in LEFT_OUT]
    diff = differences(legacy, port, names=names)
    assert diff.identical, diff.summary()
    return legacy, port


@pytest.mark.parametrize("name", sorted(variants()))
def test_parameter_sets(name):
    legacy, port = assert_same_seedling(variants()[name])
    assert len(port.vertices) == len(legacy.vertices) > 1


@pytest.mark.parametrize("change", [
    dict(n_seminal_roots=1, n_adventitious_roots=0),                   # the primary root alone
    dict(random=False),                                                # the radii not drawn: the draws shift
    dict(initial_segment_length=0., initial_apex_length=2e-3),         # a zero-length base: a collar of its own
    dict(WHEAT_SOISSON, CVDD=0.3, random_choice=7),
], ids=["primary_only", "not_random", "zero_length_base", "GB_other_seed"])
def test_edge_cases(change):
    assert_same_seedling(change)


def test_events_table(tmp_path):
    """Seminal roots from a CSV table of events, two per event: the draws seeded by random_choice + i j differ."""
    pd.DataFrame({"emergence_delay_in_thermal_time": [1e5, 2e5, 3e5],
                  "number_of_seminal_roots_per_event": [2, 1, 2]}).to_csv(tmp_path / "seminal.csv", index=False)
    legacy, port = assert_same_seedling(dict(input_file_path=str(tmp_path), seminal_roots_events_file="seminal.csv",
                                             n_adventitious_roots=0))
    angles = {round(legacy.props["angle_roll"][v], 9) for v in legacy.vertices if legacy.props["type"][v] == 2}
    assert len(angles) > 1
