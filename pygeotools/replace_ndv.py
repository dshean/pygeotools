#! /usr/bin/env python

#David Shean
#dshean@gmail.com

import sys
import os
import argparse

import numpy as np
from osgeo import gdal

from pygeotools.lib import iolib

#Can use ASP image_calc for multithreaded ndv replacement of huge images
#image_calc -o ${1%.*}_ndv.tif -c 'var_0' --output-nodata-value $2 $1

def getparser():
    parser = argparse.ArgumentParser(description="Replace raster NoData value")
    parser.add_argument('-overwrite', action='store_true', help='Overwrite original file')
    parser.add_argument('src_fn', type=str, help='Input raster filename')
    parser.add_argument('new_ndv', type=str, help='New NoData value (e.g., -9999)')
    return parser

def main():
    parser = getparser()
    args = parser.parse_args()

    src_fn = args.src_fn
    new_ndv = args.new_ndv

    #Input argument is a string, which is not recognized by set_fill_value
    #Must use np.nan object
    if new_ndv == 'nan' or new_ndv == 'np.nan':
        new_ndv = np.nan
    else:
        new_ndv = float(new_ndv)

    #Output filename will have ndv appended
    if args.overwrite:
        out_fn = src_fn
    else:
        out_fn = os.path.splitext(src_fn)[0]+'_ndv.tif'

    ds = gdal.Open(src_fn)
    b = ds.GetRasterBand(1)
    #Extract old ndv
    old_ndv = iolib.get_ndv_b(b)

    print(src_fn)
    print("Replacing old ndv %s with new ndv %s" % (old_ndv, new_ndv))

    #Load masked array
    bma = iolib.ds_getma(ds)
    #Set new fill value
    bma.set_fill_value(new_ndv)
    #Fill ma with new value and write out
    iolib.writeGTiff(bma.filled(), out_fn, ds, ndv=new_ndv)

if __name__ == '__main__':
    main()
