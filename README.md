# pygeotools
Libraries and command line tools for geospatial data processing and analysis

## Features
- Resample/warp rasters to common resolution/extent/projection
- Many functions to handle rasters with NoData gaps using [NumPy masked arrays](https://docs.scipy.org/doc/numpy/reference/maskedarray.generic.html)
- Point data coordinate transformations, sampling, and interpolation routines (e.g., arrays of xyz points)
- Common raster filtering operations

### Libraries [pygeotools/lib](./pygeotools/lib) 
- geolib - Coordinate transformations, raster to vector, vector to raster
- malib - NumPy Masked Array operations, DEMStack class
- warplib - On-the-fly GDAL warp operations for abitrary number of input datasets
- iolib - File input/output, wrappers for GDAL I/O, masked array write to disk
- timelib - Time conversions, extract timestamps from filenames, useful for raster time series analysis
- filtlib - Collection of filters for 2D masked arrays (Gauss, rolling median, high pass, etc.)

### Command-line utilities (run with no arguments for usage)
- warptool.py - Warp arbitrary rasters to common res/extent/proj
- make_stack.py - Create a "stack" of input rasters (a raster time series object) and compute stats
- clip_raster_by_shp.py - Clip and mask an input raster using a polygon shapefile
- apply_mask.py - Apply mask from one raster to another
- filter.py - Apply various filters available in filtlib
- trim_ndv.py - Remove rows/cols containing only NoData from raster margins
- replace_ndv.py - Replace NoData value
- proj_select.py - Automatically determine projection for input lat/lon or raster
- raster2shp.py - Create polygon shapefile of input raster footprints
- ogr_merge.sh - Merge shapefiles
- ...

## Examples 

### Warping multiple datasets to common grid, computing difference, writing out
```
from pygeotools.lib import iolib, warplib, malib
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
or, from the command line... 

Warp all to match raster1.tif projection with common intersection and largest pixel size:

`warptool.py -tr max -te intersection -t_srs first raster1.tif raster2.tif raster3.tif`

Create version of raster1.tif that matches resolution, extent, and projection of raster2.tif:

`warptool.py -tr raster2.tif -te raster2.tif -t_srs raster2.tif raster1.tif`

Reproject and clip to user-defined extent, preserving original resolution of each input raster:

`warptool.py -tr source -te '439090 5285360 458630 5306450' -t_srs EPSG:32610 raster1.tif raster2.tif`

### Creating a time series "stack" object:
```
from pygeotools.lib import malib
fn_list = ['20080101_dem.tif', '20090101_dem.tif', '20100101_dem.tif']
s = malib.DEMStack(fn_list, res='min', extent='union')
#Stack standard deviation
s.stack_std
#Stack linear trend
s.stack_trend
```
or, from the command line...

`make_stack.py -tr 'min' -te 'union' 20*.tif`

## Documentation

http://pygeotools.readthedocs.io

## Installation

Install the latest release from PyPI:

    pip install pygeotools 

**Note**: by default, this will deploy executable scripts in /usr/local/bin

### Building from source

Clone the repository and install:

    git clone https://github.com/dshean/pygeotools.git
    pip install -e pygeotools

The -e flag ("editable mode", setuptools "develop mode") will allow you to modify source code and immediately see changes.

### Core requirements 
- [GDAL/OGR](http://www.gdal.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)

### Optional requirements (needed for some functionality) 
- [matplotlib](http://matplotlib.org/)
- [NASA Ames Stereo Pipeline (ASP)](https://ti.arc.nasa.gov/tech/asr/intelligent-robotics/ngt/stereo/)

## Disclaimer 

This originated as a personal repo that I am slowly cleaning up and distributing.  There are some useful things that work very well, other things that were hastily written for a one-off task several years ago, and some confusing things that were never finished. 

Contributions, bug reports, and general feedback are all welcome.  My time is limited, I have some bad habits, and I could really use some help.  Thanks in advance.

This was all originally developed for Python 2.X, but should now also work with Python 3.X thanks to [@dlilien](https://github.com/dlilien)

Some of this functionality now exists in the excellent, mature, well-supported [rasterio](https://github.com/mapbox/rasterio).  Eventually, I will integrate rasterio API calls where appropriate.

## License

This project is licensed under the terms of the MIT License.

