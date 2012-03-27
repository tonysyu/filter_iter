#!/usr/bin/env python

import os
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('filter_iter', parent_package, top_path)

    cython(['_filter_iter.pyx'], working_path=base_path)
    config.add_extension('_filter_iter', sources=['_filter_iter.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('example_C_filters', sources=['example_C_filters.c'])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer = 'scikits-image Developers',
          author = 'scikits-image Developers',
          maintainer_email = 'scikits-image@googlegroups.com',
          description = 'Filter iterator',
          url = 'https://github.com/scikits-image/scikits-image',
          license = 'SciPy License (BSD Style)',
          **(configuration(top_path='').todict())
          )

