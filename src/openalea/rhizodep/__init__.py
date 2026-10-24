"""rhizodep.

The models below are written on the former metafspm API: they are imported on first access only, so the package (its
frozen ``legacy`` growth component, the port) can be imported with the current metafspm installed."""
_LAZY = {
    "RootCarbonModel": ".root_carbon",
    "RhizoInputsSoilModel": ".soil_module",
    "RhizoDep": ".rhizodep",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
