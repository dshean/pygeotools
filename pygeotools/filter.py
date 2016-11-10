#! /usr/bin/env python
"""
David Shean
dshean@gmail.com

Filter an input DEM using functions in filtlib
"""

import sys
import os

import numpy as np
from osgeo import gdal

from pygeotools.lib import iolib
from pygeotools.lib import malib
from pygeotools.lib import filtlib

def main(argv=None):
    #This is defualt filter size (pixels) 
    size = 19
    #Compute and print stats before/after
    stats = False 

    #Need to clean up with argparse
    #Accept filter list, filter size as argument

    if len(sys.argv) == 2:
        dem_fn = sys.argv[1]
    elif len(sys.argv) == 3:
        dem_fn = sys.argv[1]
        size = int(sys.argv[2])
    else:
        sys.exit("Usage is %s dem [size]" % sys.argv[0])
   
    print("Loading DEM into masked array")
    dem_ds = iolib.fn_getds(dem_fn)
    dem = iolib.ds_getma(dem_ds, 1)
    #Cast input ma as float32 so np.nan filling works
    #dem = dem.astype(np.float32)

    dem_fltr = dem

    #Should accept arbitrary number of ordered filter operations as cli argument
    #filt_list = ['gauss_fltr_astropy',]
    #for filt in filt_list:
    #    dem_fltr = filtlib.filt(dem_fltr, size=size, kwargs)

    #Percentile filter
    #dem_fltr = filtlib.perc_fltr(dem, perc=(15.865, 84.135))
    #dem_fltr = filtlib.perc_fltr(dem, perc=(2.275, 97.725))
    #dem_fltr = filtlib.perc_fltr(dem, perc=(0.135, 99.865))
    #dem_fltr = filtlib.perc_fltr(dem, perc=(0, 99.73))
    #dem_fltr = filtlib.perc_fltr(dem, perc=(0, 95.45))
    #dem_fltr = filtlib.perc_fltr(dem, perc=(0, 68.27))

    #Difference filter, need to specify refdem_fn
    #dem_fltr = filtlib.dz_fltr(dem_fn, refdem_fn, abs_dz_lim=(0,30))

    #Absolute range filter
    #dem_fltr = filtlib.range_fltr(dem_fltr, (15, 9999))

    #Median filter
    #dem_fltr = filtlib.rolling_fltr(dem_fltr, f=np.nanmedian, size=size)
    #dem_fltr = filtlib.median_fltr(dem_fltr, fsize=size, origmask=True)
    #dem_fltr = filtlib.median_fltr_skimage(dem_fltr, radius=4, origmask=True)

    #High pass filter
    #dem_fltr = filtlib.highpass(dem_fltr, size=size)

    #Gaussian filter (default)
    dem_fltr = filtlib.gauss_fltr_astropy(dem_fltr, size=size, origmask=True, fill_interior=False)

    if stats:
        print("Input DEM stats:")
        malib.print_stats(dem)
        print("Filtered DEM stats:")
        malib.print_stats(dem_fltr)
    
    dst_fn = os.path.splitext(dem_fn)[0]+'_filt%ipx.tif' % size
    print("Writing out filtered DEM: %s" % dst_fn)
    #Note: writeGTiff writes dem_fltr.filled()
    iolib.writeGTiff(dem_fltr, dst_fn, dem_ds) 

if __name__ == '__main__':
    main()
