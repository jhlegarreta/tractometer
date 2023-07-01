#!/usr/bin/env python

import os
from glob import glob
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

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
    ext_modules=ext_modules,
    cmdclass={'build_ext': CustomBuildExtCommand},
)

setup(**opts)
