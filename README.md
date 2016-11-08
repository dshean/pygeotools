# pygeotools
Libraries and utilities for geospatial data processing/analysis

## Overview

## Usage

Warping multiple datasets to common grid and computing difference
```
from pygeotools import iolib, warplib, malib
fn1 = 'raster1.tif'
fn2 = 'raster2.tif'
ds_list = warplib.memwarp_multi_fn([fn1, fn2], res='max', extent='intersection', t_srs='first', r='cubic')
r1 = iolib.ds_getma(ds_list[0])
r2 = iolib.ds_getma(ds_list[1])
rdiff = r1 - r2
malib.print_stats(rdiff)
out_fn = 'raster_diff.tif'
iolib.writeGTiff(rdiff, out_fn, ds_list[0])
```

## Features

### pygeotools/lib - libraries containing many useful functions
- geolib - coordinate transformations, raster to vector, vector to raster
- malib - NumPy Masked Array, DEMStack class
- warplib - on-the-fly GDAL warp operations
- iolib - file input/output, primarily wrappers for GDAL I/O
- timelib - time conversions, useful when working with time series
- filtlib - raster filtering operations
- pltlib - some useful matplotlib plotting functions

### pygeotools/bin

Useful Python and shell command-line utilities
- warptool.py
- ndvtrim.py
- ...

## Documentation

Is in the works...

## Installation

Install the latest release from PyPI:

    pip install pygeotools 

### Building from source

Clone the repository and install:

    git clone https://github.com/dshean/pygeotools.git
    pip install pygeotools/

### Requirements 
- gdal
- numpy
- matplotlib

## Credits

This originated as a poorly-written, poorly-organized personal repo that I am finally cleaning up and distributing.   

## License

This project is licensed under the terms of the MIT License.

