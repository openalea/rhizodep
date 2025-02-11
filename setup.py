#!/usr/bin/env python
# -*- coding: utf-8 -*-

# {# pkglts, pysetup.kwds
# format setup arguments

from os import walk
from os.path import abspath, normpath, splitext
from os.path import join as pj

from setuptools import setup, find_namespace_packages

short_descr = "RhizoDeposition"
readme = open('README.rst').read()
history = open('HISTORY.rst').read()

# find packages
pkgs = packages = find_namespace_packages(where='src')#, include=['openalea.*'])


pkg_data = {}

nb = len(normpath(abspath("src/rhizodep"))) + 1
data_rel_pth = lambda pth: normpath(abspath(pth))[nb:]

data_files = []
for root, dnames, fnames in walk("src/rhizodep"):
    for name in fnames:
        if splitext(name)[-1] in [u'.json', u'.ini']:
            data_files.append(data_rel_pth(pj(root, name)))

pkg_data['rhizodep'] = data_files

# find version number in src/openalea/rhizodep/version.py
_version = {}
with open("src/rhizodep/version.py") as fp:
    exec(fp.read(), _version)


setup_kwds = dict(
    name='rhizodep',
    version="0.0.1",
    description=short_descr,
    long_description=readme + '\n\n' + history,
    author="Frederic Rees",
    author_email="frederic.rees@inrae.fr",
    url='',
    license='cecill-c',
    zip_safe=False,

    packages=pkgs,
    #namespace_packages=['openalea'],
    package_dir={'': 'src'},

    package_data=pkg_data,
    setup_requires=[
        "pytest-runner",
    ],
    install_requires=[
    ],
    tests_require=[
        "pytest",
        "pytest-mock",
    ],
    entry_points={},
    keywords='',
)
# #}
# change setup_kwds below before the next pkglts tag

# do not change things below
# {# pkglts, pysetup.call
setup_kwds['setup_requires'] = []
setup_kwds['tests_requires'] = []

setup(**setup_kwds)
# #}
