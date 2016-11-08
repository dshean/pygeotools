# pygeotools
Libraries and utilities for geospatial data processing/analysis

## Overview

## Features
- quickly resample rasters to common resolution/extent/projection
- functions for NumPy masked arrays
- simple coordinate transformations
- automatic projection determination

### pygeotools/lib - libraries containing many useful functions
- geolib - coordinate transformations, raster to vector, vector to raster
- malib - NumPy Masked Array operations, DEMStack class
- warplib - on-the-fly GDAL warp operations
- iolib - file input/output, primarily wrappers for GDAL I/O, write out masked arrays
- timelib - time conversions, useful for raster time series analysis
- filtlib - raster filtering 

### pygeotools/bin

Useful command-line utilities
- warptool.py
- make_stack.py
- ndvtrim.py
- apply_mask.py
- ...

## Examples 

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
or, from the command line: 
`warptool.py -tr 'max' -te 'intersection' -t_srs 'first' raster1.tif raster2.tif`

Creating a nDarray "stack" object (currently called DEMStack, but any 2.5D raster inputs will do):
```
fn_list = ['20080101_dem.tif', '20090101_dem.tif', '20100101_dem.tif']
s = malib.DEMStack(fn_list, res='min', extent='union')
#Standard deviation
s.stack_std
#Linear fit 
s.stack_trend
```
or, from the command line: 
`make_stack.py -tr 'min' -te 'union' 20*.tif`

## Documentation

Is in the works...

## Installation

Install the latest release from PyPI:

    pip install pygeotools 

### Building from source

Clone the repository and install:

    git clone https://github.com/dshean/pygeotools.git
    pip install pygeotools/

### Core requirements 
- GDAL 
- NumPy 

### Optional requirements (needed for some functionality) 
- Matplotlib
- SciPy
- NASA Ames Stereo Pipeline (Precompiled binaries and documentation available from https://ti.arc.nasa.gov/tech/asr/intelligent-robotics/ngt/stereo/)

## Disclaimer 

This originated as a poorly-written, poorly-organized personal repo that I am finally cleaning up and distributing.  There are some things that work very well, and other things that were hastily written for a one-off task several years ago.  

This was all developed for Python 2.X, but will support Python 3.X in the near future.

## License

This project is licensed under the terms of the MIT License.

