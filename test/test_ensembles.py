"""The growth components against their statistical reference (devplan §29, D-X2): each configuration's ensemble over
seeds, its means within the sampling error of the reference's (computed on the tag port-exact: test/reference)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "reference"))
from ensembles import CONFIGURATIONS, METRICS, ensemble  # noqa: E402

REFERENCE = json.load(open(os.path.join(os.path.dirname(__file__), "reference", "ensembles.json")))


@pytest.mark.parametrize("name", list(CONFIGURATIONS))
def test_ensemble_agrees_with_the_reference(name):
    reference = REFERENCE["configurations"][name]
    values = ensemble(name, seeds=REFERENCE["seeds"], days=REFERENCE["days"])
    n = len(REFERENCE["seeds"])
    for metric in METRICS:
        new = np.array(values[metric])
        ref = reference[metric]
        error = np.sqrt((ref["std"] ** 2 + new.std(ddof=1) ** 2) / n)          # the difference of two means
        tolerance = 3. * error + 1e-9 * abs(ref["mean"])
        assert abs(new.mean() - ref["mean"]) <= tolerance, (metric, new.mean(), ref["mean"], tolerance)
