
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

### Description

RhizoDep is a functional-structural root model aiming to simulate root growth, respiration and rhizodeposition along the whole root system. Root growth is based on a potential growth model adapted from ArchiSimple model, which is regulated by the local balance of carbon in each root segment. This local carbon balance also enables to simulate root respiration, which originates from growth and maintenance, and rhizodeposition, which has been represented as the net exudation of hexose, the secretion of mucilage and the release of cap cells. The model depends on i) the input of carbon allocated from the shoots to the roots, ii) the temperature of the soil, iii) the dynamics of rhizodeposits at the root-soil interface, which is currently simulated by a simple soil degradation function.

### Installation

```bash
mamba install -c openalea3 -c conda-forge openalea.rhizodep
```

### contributors

<a href="https://github.com/openalea/rhizodep/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openalea/rhizodep" />
</a>

