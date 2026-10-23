rhizodep's growth component, frozen: `src/openalea/rhizodep/root_growth.py` of branch `to_metafspm` at d65a3fa, verbatim.
Only its imports are rewritten, so that it runs on its own next to the current metafspm: it was written against
metafspm's `publish_WB` branch (5bd0645), whose modules it imports are vendored in `_metafspm/` (`component.py`,
`component_factory.py`, `utils.py`, `specializer.py`), verbatim but for their internal imports and the class paths their type checks compare as strings
(`<class 'openalea.metafspm.utils.ArrayDict'>`).

It is the reference the port is compared with (`test/comparison/`, `example/comparison/`). It is not developed further:
new work goes into the main package, which is the port (`devplan_metafspm.md`).
