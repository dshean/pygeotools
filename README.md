# pygeotools
A collection of libraries and utilities for geospatial data processing/analysis

## Overview

This repository contains 

### pygeotools/lib - libraries containing many useful functions
- geolib - coordinate transformations, raster to vector, vector to raster
- malib - NumPy Masked Array, DEMStack class
- warplib - on-the-fly GDAL warp operations
- iolib - file input/output, primarily wrappers for GDAL I/O
- timelib - time conversions, useful when working with time series
- filtlib - raster filtering operations
- pltlib - some useful matplotlib plotting functions

### pygeotools

Useful command line utilities 

## Documentation

Is in the works...

## Installation

The easiest way to install in production is via `pip`. Installation requires a
recent version of `setuptools`:

    pip install -U setuptools

Then, to install the latest release from PyPI:

    pip install pygeotools 

### Building from source

Clone the repository and install:

    git clone https://github.com/dshean/pygeotools.git
    pip install pygeotools/

## Dependencies

- gdal >= 1.10
- numpy >= 1.7
