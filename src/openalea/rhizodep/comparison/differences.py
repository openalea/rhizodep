"""
Two records (Snapshots, keyed by the legacy's vertex ids) compared property by property: the tests' and the visual
comparison's measure of exactness.

    diff = differences(legacy, port, names=None, rtol=0., atol=0.)
    diff.topology          the vertices only one side has, and those whose parent or edge type differ
    diff.values[name]      the vertices where *name* differs (beyond rtol / atol), present on both sides
    diff.identical         nothing differs
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Differences:
    only_first: list = field(default_factory=list)
    only_second: list = field(default_factory=list)
    parent: list = field(default_factory=list)
    edge_type: list = field(default_factory=list)
    values: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)

    @property
    def topology(self):
        return sorted(set(self.only_first) | set(self.only_second) | set(self.parent) | set(self.edge_type))

    @property
    def differing_vertices(self):
        out = set(self.topology)
        for vids in self.values.values():
            out |= set(vids)
        return sorted(out)

    @property
    def identical(self):
        return not (self.topology or self.missing or any(self.values.values()))

    def summary(self, limit=8):
        if self.identical:
            return "identical"
        parts = []
        if self.only_first or self.only_second:
            parts.append(f"{len(self.only_first)} vertices only in the first, {len(self.only_second)} only in the second")
        if self.parent or self.edge_type:
            parts.append(f"{len(self.parent)} parents, {len(self.edge_type)} edge types differ")
        if self.missing:
            parts.append(f"missing on one side: {', '.join(self.missing[:limit])}")
        differing = {name: vids for name, vids in self.values.items() if vids}
        for name in sorted(differing, key=lambda n: -len(differing[n]))[:limit]:
            parts.append(f"{name}: {len(differing[name])}")
        return "; ".join(parts)


def _equal(a, b, rtol, atol):
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return list(a or []) == list(b or [])
    if a is None or b is None or isinstance(a, str) or isinstance(b, str):
        return a == b
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= atol + rtol * abs(b)


def differences(first, second, names=None, rtol=0., atol=0.):
    """*first* against *second*; *names*: the properties compared (default: those both sides have)."""
    diff = Differences()
    a, b = set(first.vertices), set(second.vertices)
    diff.only_first, diff.only_second = sorted(a - b), sorted(b - a)
    common = sorted(a & b)
    diff.parent = [v for v in common if first.parent[v] != second.parent[v]]
    diff.edge_type = [v for v in common if first.edge_type[v] != second.edge_type[v]]
    if names is None:
        names = sorted(set(first.props) & set(second.props))
        diff.missing = sorted(set(first.props) ^ set(second.props))
    for name in names:
        if name not in first.props or name not in second.props:
            diff.missing.append(name)
            continue
        pa, pb = first.props[name], second.props[name]
        diff.values[name] = [v for v in common if not _equal(pa.get(v), pb.get(v), rtol, atol)]
    return diff
