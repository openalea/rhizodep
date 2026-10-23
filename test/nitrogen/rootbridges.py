"""
Root_BRIDGES' recorded states, from the rhizodep environment: rootbridges_run.py runs in the Wheat-BRIDGES
environment (a subprocess), its records come back as legacy_run's Snapshots. Cached on disk by request.

    stages = rootbridges_stages(parameters, n_steps=240, C_hexose_root=1e-3, AA=1e-3, every=12)
    available()      whether the Wheat-BRIDGES environment is there (the tests are skipped otherwise)

The Python of that environment: $ROOTBRIDGES_PYTHON, else ~/miniconda3/envs/wheat-bridges/bin/python.
"""
import hashlib
import os
import pickle
import subprocess
import tempfile

from openalea.rhizodep.comparison.legacy_run import Snapshot

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.environ.get("ROOTBRIDGES_PYTHON", os.path.expanduser("~/miniconda3/envs/wheat-bridges/bin/python"))
CACHE = os.path.join(tempfile.gettempdir(), "rhizodep_rootbridges_records")


def available() -> bool:
    if not os.path.exists(PYTHON):
        return False
    probe = subprocess.run([PYTHON, "-c", "import openalea.rootbridges.root_growth"], capture_output=True)
    return probe.returncode == 0


def _snapshot(record) -> Snapshot:
    return Snapshot(**record)


def rootbridges_stages(parameters=None, n_steps=240, C_hexose_root=1e-3, AA=1e-3, every=12, first=0,
                       time_step=3600, adventitious_to_emerge=None):
    """[(step, {"start": Snapshot, "potential": ..., "actual": ..., "segmentation": ..., "post_growth": ...})] of
    Root_BRIDGES' growth on its own (C_hexose_root and AA: numbers, or functions (step, g) -> {vid: value})."""
    request = dict(parameters=dict(parameters or {}), n_steps=n_steps, C_hexose_root=C_hexose_root, AA=AA,
                   every=every, first=first, time_step=time_step, adventitious_to_emerge=adventitious_to_emerge)
    key = hashlib.sha1(pickle.dumps(request) + open(os.path.join(HERE, "rootbridges_run.py"), "rb").read())
    path = os.path.join(CACHE, key.hexdigest() + ".pkl")
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as file:
            pickle.dump(request, file)
        try:
            done = subprocess.run([PYTHON, os.path.join(HERE, "rootbridges_run.py"), file.name, path + ".part"],
                                  capture_output=True, text=True)
        finally:
            os.unlink(file.name)
        if done.returncode != 0:
            raise RuntimeError(f"Root_BRIDGES' run failed:\n{done.stderr[-3000:]}")
        os.replace(path + ".part", path)
    with open(path, "rb") as file:
        records = pickle.load(file)
    return [(step, {stage: _snapshot(record) for stage, record in stages.items()}) for step, stages in records]
