#!/usr/bin/env python
# -*- coding: utf-8 -*-

# format setup arguments

from os import walk
from os.path import abspath, normpath, splitext
from os.path import join as pj

from setuptools import setup, find_namespace_packages

short_descr = "RhizoDep"
readme = open('README.md').read()

# find packages
pkgs = packages = find_namespace_packages(where='src', include=['openalea.*'])


pkg_data = {}

nb = len(normpath(abspath("src/openalea/rhizodep"))) + 1
data_rel_pth = lambda pth: normpath(abspath(pth))[nb:]

data_files = []
for root, dnames, fnames in walk("src/openalea/rhizodep"):
    for name in fnames:
        if splitext(name)[-1] in [u'.json', u'.ini']:
            data_files.append(data_rel_pth(pj(root, name)))

pkg_data['openalea.rhizodep'] = data_files

# find version number in src/openalea/rhizodep/version.py
_version = {}
with open("src/openalea/rhizodep/version.py") as fp:
    exec(fp.read(), _version)

version = _version['__version__']

setup_kwds = dict(
    name='openalea.rhizodep',
    version=version,
    description=short_descr,
    long_description=readme,
    author="Frederic Rees",
    author_email="frederic.rees@inrae.fr",
    url='https://github.com/openalea/rhizodep',
    license='cecill-c',
    zip_safe=False,

    packages=pkgs,
    package_dir={'': 'src'},

    package_data=pkg_data,

    entry_points={},
    keywords='',
)


setup(**setup_kwds)
