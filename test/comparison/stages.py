"""The port's stages on the legacy's states (devplan §14, state transplant): shared by the stage tests."""
import numpy as np

from openalea.rhizodep.comparison.differences import differences
from openalea.rhizodep.comparison.legacy_run import legacy_stages
from openalea.rhizodep.comparison.port_run import LEFT_OUT, run_stage, snapshot, transplant_scene
from openalea.rhizodep.legacy import RootGrowthModel

# Stops, deaths and nodules within 10 days (the parameter sets have none of the two latter): short lives, nodules
# (their legacy draw is unseeded, but each state is compared on its own), a lateral share in the pipe model
COVERAGE = dict(SGC=1.0, LDs=RootGrowthModel.LDs / 200, nodules=True, nodule_formation_probability=0.5)

PREVIOUS = {"potential": "start", "actual": "potential", "segmentation": "actual", "post_growth": "segmentation"}


def varying_C(step, g):
    """A hexose concentration varying between elements and steps, from 1e-5 to 1e-3 mol.g-1 (hexose-limited
    growth, supply zones summing unequal contributions)."""
    return {v: 1e-5 + 1e-3 * (0.5 + 0.5 * np.sin(0.7 * v + 0.3 * step)) for v in g.vertices(scale=1)}


def stage_differences(parameters, stage, n_steps=240, every=12, first=0, C_hexose_root=1e-3):
    """[(step, Differences)] of the port's *stage* on the legacy's state before it, against the legacy's after."""
    out = []
    for step, records in legacy_stages(parameters, n_steps=n_steps, every=every, first=first,
                                       C_hexose_root=C_hexose_root):
        scene = transplant_scene(records[PREVIOUS[stage]], parameters)
        run_stage(scene, stage)
        legacy = records[stage]
        out.append((step, differences(legacy, snapshot(scene), names=[n for n in legacy.props if n not in LEFT_OUT])))
    return out


def assert_stage_exact(parameters, stage, **options):
    differing = [(step, diff.summary()) for step, diff in stage_differences(parameters, stage, **options)
                 if not diff.identical]
    assert not differing, differing[:3]
