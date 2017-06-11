#!/usr/bin/env python

from distutils.core import setup

#To prepare a new release
#python setup.py sdist upload

setup(name='pygeotools',
    version='0.3.0',
    description='Libraries and command-line utilities for geospatial data processing/analysis',
    author='David Shean',
    author_email='dshean@gmail.com',
    license='MIT',
    url='https://github.com/dshean/pygeotools',
    packages=['pygeotools', 'pygeotools.lib'],
    long_description=open('README.md').read(),
    install_requires=['gdal','numpy','scipy','matplotlib'],
    #Note: this will write to /usr/local/bin
    scripts=['pygeotools/warptool.py', 'pygeotools/make_stack.py', \
            'pygeotools/raster2shp.py', 'pygeotools/clip_raster_by_shp.py', 'pygeotools/apply_mask.py', \
            'pygeotools/mask_raster.sh', 'pygeotools/ogr_merge.sh', \
            'pygeotools/replace_ndv.py', 'pygeotools/trim_ndv.py', \
            'pygeotools/filter.py', 'pygeotools/proj_select.py']
)

