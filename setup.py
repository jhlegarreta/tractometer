#!/usr/bin/env python

import os
from glob import glob
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


with open('requirements.txt') as f:
    required_dependencies = f.read().splitlines()
    external_dependencies = []
    for dependency in required_dependencies:
        if dependency[0:2] == '-e':
            repo_name = dependency.split('=')[-1]
            repo_url = dependency[3:]
            external_dependencies.append('{} @ {}'.format(repo_name, repo_url))
        else:
            external_dependencies.append(dependency)

ext_modules = [
    Extension(
        'challenge_scoring.tractanalysis.robust_streamlines_metrics',
        ['challenge_scoring/tractanalysis/robust_streamlines_metrics.pyx']
    )
]


class CustomBuildExtCommand(build_ext):
    """Build_ext command to use when numpy headers are needed."""

    def run(self):
        # Now that the requirements are installed, get everything from numpy
        from Cython.Build import cythonize
        from numpy import get_include

        # Add everything requires for build
        self.swig_opts = None
        include_dirs = [get_include(),
                        os.path.join(
                            os.path.dirname(
                                os.path.realpath(__file__)),
                            'challenge_scoring/c_src')]
        self.include_dirs = include_dirs
        self.distribution.ext_modules[:] = cythonize(
            self.distribution.ext_modules,
            compiler_directives={'language_level': '3'})

        # Call original build_ext command
        build_ext.finalize_options(self)
        build_ext.run(self)


opts = dict(
    name='tractometer', version='1.0.1',
    description='Scoring system used for the ISMRM 2015 Tractography Challenge',
    url='https://github.com/scilus/ismrm_2015_tractography_challenge_scoring',
    ext_modules=ext_modules,
    author='The challenge team',
    author_email='jean-christophe.houde@usherbrooke.ca',
    packages=[
        'challenge_scoring',
        'challenge_scoring.io',
        'challenge_scoring.metrics',
        'challenge_scoring.tractanalysis',
        'challenge_scoring.utils'
    ],
    cmdclass={'build_ext': CustomBuildExtCommand},
    setup_requires=['Cython'],
    scripts=glob('scripts/*.py'),
    install_requires=external_dependencies,
    requires=['numpy', 'scipy']
)

setup(**opts)
