"""The root systems' geometry against the frozen legacy's, statistically (the draws differ): each element's angle from
the vertical by root order, the laterals' angle to their bearer, and no lean of the plant to one side, from the
elements' directions drawn by rhizodep's turtle (comparison.drawing.positions), without and with its gravitropism."""
import contextlib
import io

import numpy as np
import pytest

from openalea.archisimple.drawing import positions
from openalea.rhizodep.comparison.legacy_run import legacy_run
from openalea.rhizodep.comparison.port_run import port_run
from openalea.rhizodep.parameter_sets import WHEAT_SOISSON

SEEDS = (1, 2, 3, 4)
STEPS = 240


def geometry(record, gravitropism):
    """{measure: values}: the angles (°) from the vertical by order, of the laterals to their bearer; the elements'
    mean horizontal position (x, y) relative to the plant's extent (signed: a lean every plant takes shows in its mean
    over seeds)."""
    xyz = positions(record, gravitropism=gravitropism)
    p = record.props
    heading = {v: (end - start) / np.linalg.norm(end - start) for v, (start, end) in xyz.items()
               if p["length"][v] > 0}
    out = {f"from the vertical, order {order}": np.degrees(np.arccos(np.clip(
        [-heading[v][2] for v in heading if p["root_order"][v] == order], -1., 1.))) for order in (1, 2)}
    out["lateral to bearer"] = np.degrees(np.arccos(np.clip(
        [heading[v] @ heading[record.parent[v]] for v in heading
         if record.edge_type[v] == "+" and record.parent[v] in heading], -1., 1.)))
    ends = np.array([end for _, end in xyz.values()])
    extent = np.abs(ends[:, :2]).max()
    out["lean"] = ends[:, :2].mean(axis=0) / extent
    return out


@pytest.fixture(scope="module")
def records():
    legacy, port = [], []
    for seed in SEEDS:
        scenario = dict(WHEAT_SOISSON, random_choice=seed)
        with contextlib.redirect_stdout(io.StringIO()):
            legacy.append(legacy_run(scenario, n_steps=STEPS, record_every=STEPS).steps[-1])
        port.append(port_run(scenario, n_steps=STEPS, record_every=STEPS).steps[-1])
    return legacy, port


@pytest.mark.parametrize("gravitropism", [0., 0.06])
def test_the_angles_are_the_legacys(records, gravitropism):
    pooled, leans = {}, []
    for side, side_records in zip(("legacy", "port"), records):
        for record in side_records:
            measures = geometry(record, gravitropism)
            if side == "port":
                leans.append(measures.pop("lean"))
            for measure, values in measures.items():
                if measure != "lean":
                    pooled.setdefault(measure, {}).setdefault(side, []).extend(values.tolist())
    # no side every plant takes (the roots leaving the base all in one plane gave 0.23 to 0.33 in x)
    assert (np.abs(np.mean(leans, axis=0)) < 0.15).all(), np.mean(leans, axis=0)
    for measure, sides in pooled.items():
        legacy, port = np.array(sides["legacy"]), np.array(sides["port"])
        assert port.mean() == pytest.approx(legacy.mean(), abs=5.), (measure, port.mean(), legacy.mean())
        assert port.std() == pytest.approx(legacy.std(), rel=0.25), (measure, port.std(), legacy.std())
