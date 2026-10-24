
[![Documentation Status](https://readthedocs.org/projects/rhizodep/badge/?version=latest)](https://rhizodep.readthedocs.io/en/latest/?badge=latest)
[![Anaconda-Server Badge](https://anaconda.org/openalea3/openalea.rhizodep/badges/version.svg)](https://anaconda.org/openalea3/openalea.rhizodep)
[![Anaconda-Server Badge](https://anaconda.org/openalea3/openalea.rhizodep/badges/latest_release_date.svg)](https://anaconda.org/openalea3/openalea.rhizodep)
[![Anaconda-Server Badge](https://anaconda.org/openalea3/openalea.rhizodep/badges/platforms.svg)](https://anaconda.org/openalea3/openalea.rhizodep)
[![Anaconda-Server Badge](https://anaconda.org/openalea3/openalea.rhizodep/badges/license.svg)](https://anaconda.org/openalea3/openalea.rhizodep)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14900673.svg)](https://doi.org/10.5281/zenodo.14900673)

# OpenAlea.RhizoDep : a Functional-Structural Root Model to simulate rhizodeposition

**Authors:** Frederic Rees, Tristan Gerault, Marion Gauthier, Christophe Pradal3

**Licence:** [CeCILL-C](http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html)  

**Status:** An OpenAlea Python package

## About

### Model description

RhizoDep is a functional-structural root model aiming to simulate root growth, respiration and rhizodeposition along the whole root system. Root growth is based on a potential growth model adapted from ArchiSimple model, which is regulated by the local balance of carbon in each root segment. This local carbon balance also enables to simulate root respiration, which originates from growth and maintenance, and rhizodeposition, which has been represented as the net exudation of hexose, the secretion of mucilage and the release of cap cells. The model depends on i) the input of carbon allocated from the shoots to the roots, ii) the temperature of the soil, iii) the dynamics of rhizodeposits at the root-soil interface, which is currently simulated by a simple soil degradation function.

### Package description

The folder ‘src/openalea/rhizodep’ contains the main files for running RhizoDep, including ‘model.py’ that contains the main code of the model, ‘parameters.py’ that contains default values of parameters, and ‘running_simulation.py’ that allows to run a complete simulation. Additional scripts in the subfolder ‘tool’ allow plotting the root system computing different variables within the root system. The file ‘running_scenarios.py’ allows in particular to run different scenarios, on the basis of the input file ‘scenarios_list’ that is located in tutorial/inputs. A rapid test of the model can be launched with the file ‘test_rhizodep.py’ in the folder ‘test’.

### Installation

```bash
mamba install -c openalea3 -c conda-forge openalea.rhizodep
```

### The growth component on metafspm (branch `to_metafspm`)

On this branch, rhizodep's growth component (`root_growth.py`) is being ported to the current metafspm:
StructuralComponents growing the root system in metafspm's multiscale plant graph (MPG), their steps vectorised:

- `openalea.archisimple.architecture.RootArchitecture` (a package of its own,
  [archisimple](https://github.com/openalea/archisimple)): ArchiSimple's root architecture, C-free (development,
  segmentation, laterals; growth the potential one times a satisfaction coefficient), its species' parameter sets, its
  drawings;
- `openalea.rhizodep.growth.RootGrowth(RootArchitecture)`: the architecture with its growth limited by resources
  (hexose), and root hairs, the original component's model; its hooks let a subclass add resources;
- `openalea.rhizodep.growth_CN.RootGrowthCN`: Root_BRIDGES' carbon-nitrogen growth (amino acids co-limiting growth);
- `openalea.rhizodep.parameter_sets`: a wheat (cv. Soissons), RhizoDep's calibration in Wheat-BRIDGES.

The port reproduced the original component, and Root_BRIDGES', bit for bit: the tag `port-exact` keeps that state,
its bitwise tests and the original's behaviours as options. Since then, the port leaves the original's float exactness
and is validated statistically against that tag (`test/test_ensembles.py`) and by its
invariants (`test/test_invariants.py`).

metafspm is not installable from `openalea3` in
this version yet: clone the repositories side by side on their working branches and install them from source.

```bash
mkdir fspm && cd fspm
git clone -b data_structure_api https://github.com/openalea/metafspm.git metafspm
git clone -b to_metafspm https://github.com/openalea/rhizodep.git rhizodep
git clone https://github.com/openalea/archisimple.git archisimple

mamba env create -f rhizodep/conda/environment.yaml      # rhizodep's dependencies and metafspm's
mamba activate rhizodep
pip install -e metafspm
pip install -e archisimple
pip install -e rhizodep
```

Tutorial: [root growth limited by resources in metafspm](example/tutorials/tutorial_metafspm_01_root_growth.ipynb)
(a plant model and its scenario, the MPG's grouping, RootArchitecture and its satisfaction coefficient, a resource
subclass, the coupling with other models, populations). Run it with `jupyter lab example/tutorials/`. The
architecture's (the satisfaction coefficient, the species) is archisimple's.

Some behaviours of the original component were errors: the port has the fixes only
(the tag `port-exact` has the original's as options).

The original component is kept, frozen, as `openalea.rhizodep.legacy` (with the former metafspm it was written on,
vendored). To see both side by side, and their costs:

```bash
cd rhizodep/example/comparison
python compare_root_systems.py --days 10 --seeds 8   # figures in outputs/root_systems/, totals over seeds
python profile_growth.py --days 15              # the cost of each, per step and stage: outputs/profiling/
```

The other models of this package (`root_carbon.py`, `rhizodep.py`, ...) are still written on the former metafspm
API: they are imported on first access only.

### Contributors

<a href="https://github.com/openalea/rhizodep/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openalea/rhizodep" />
</a>

