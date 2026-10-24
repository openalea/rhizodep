"""The growth components against their statistical reference: each configuration's ensemble over
seeds, its means within the sampling error of the reference's (computed on the tag port-exact: test/reference)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reference"))
from ensembles import CONFIGURATIONS, METRICS, ensemble  # noqa: E402

REFERENCE = json.load(open(os.path.join(os.path.dirname(__file__), "reference", "ensembles.json")))


def supports(name):
    """The zero-length supports the reference's seedlings had (one per seminal root but the primary, one per nodal
    root), which the seedlings no longer have."""
    component, parameters, _ = CONFIGURATIONS[name]
    return (parameters.get("n_seminal_roots", component.n_seminal_roots) - 1
            + parameters.get("n_adventitious_roots", component.n_adventitious_roots))


@pytest.mark.parametrize("name", list(CONFIGURATIONS))
def test_ensemble_agrees_with_the_reference(name):
    reference = REFERENCE["configurations"][name]
    values = ensemble(name, seeds=REFERENCE["seeds"], days=REFERENCE["days"])
    n = len(REFERENCE["seeds"])
    for metric in METRICS:
        new = np.array(values[metric])
        ref = reference[metric]
        mean = ref["mean"] - (supports(name) if metric == "vertices" else 0.)
        error = np.sqrt((ref["std"] ** 2 + new.std(ddof=1) ** 2) / n)          # the difference of two means
        tolerance = 3. * error + 1e-9 * abs(mean)
        assert abs(new.mean() - mean) <= tolerance, (metric, new.mean(), mean, tolerance)
