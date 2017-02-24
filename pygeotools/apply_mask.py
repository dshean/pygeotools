#! /usr/bin/env python

#David Shean
#dshean@gmail.com

#Utility to mask input raster using the mask from another raster 
#Mask dataset can be a standard raster with nodata value specified or a binary mask
#If a binary mask, values should be:
#True (1) for masked, False (0) for valid - consistent with np.ma

import sys, os
import argparse

import numpy as np
from osgeo import gdal

from pygeotools.lib import iolib
from pygeotools.lib import warplib

def getparser():
    parser = argparse.ArgumentParser(description="Apply existing mask to input raster")
    #Should add support for similar arguments as in warplib - arbitrary extent, res, etc
    parser.add_argument('-extent', type=str, default='raster', choices=['raster','mask','intersection','union'], \
                        help='Desired output extent')
    parser.add_argument('-invert', action='store_true', help='Invert input mask')
    parser.add_argument('-out_fn', type=str, default=None, help='Output filename')
    parser.add_argument('src_fn', type=str, help='Input raster filename')
    parser.add_argument('mask_fn', type=str, help='Input mask filename (can be existing raster with ndv, or binary mask)')
    return parser

def main():
    parser = getparser()
    args = parser.parse_args()

    src_fn = args.src_fn
    if not iolib.fn_check(src_fn):
        sys.exit("Unable to find src_fn: %s" % src_fn)

    mask_fn = args.mask_fn
    if not iolib.fn_check(mask_fn):
        sys.exit("Unable to find mask_fn: %s" % mask_fn)

    #Determine output extent, default is input raster extent 
    extent = args.extent 
    if extent == 'raster':
        extent = src_fn
    elif extent == 'mask':
        extent = mask_fn
    else:
        #This is a hack for intersection computation
        src_ds_list = [gdal.Open(fn, gdal.GA_ReadOnly) for fn in [src_fn, mask_fn]]
        #t_srs = geolib.get_ds_srs(src_ds_list[0])
        extent = warplib.parse_extent(extent, src_ds_list, src_fn)

    print("Warping mask_fn")
    mask_ds = warplib.memwarp_multi_fn([mask_fn,], res=src_fn, extent=extent, t_srs=src_fn)[0]

    print("Loading mask array")
    mask_ma_full = iolib.ds_getma(mask_ds)
    mask_ds = None

    print("Extracting mask")
    if mask_ma_full.std() != 0:
        #Input mask filename is a raster, or other masked array
        #Just need to extract mask
        mask = np.ma.getmaskarray(mask_ma_full)
    else:
        #Input mask filename is a mask, use directly
        #If input mask values are zero, valid values are nonzero
        #Bool True == 1, so need to invert
        if mask_ma_full.fill_value == 0:
            mask = ~((mask_ma_full.data).astype(bool))
        else:
            mask = (mask_ma_full.data).astype(bool)

    #Free up memory
    mask_ma_full = None

    #Add dilation step for buffer

    #newmask = np.logical_or(np.ma.getmaskarray(src_ma_full), mask)

    if args.invert:
        print("Inverting mask")
        mask = ~(mask)

    print("Loading src array and applying updated mask")
    if extent == src_fn:
        src_ds = gdal.Open(src_fn)
    else:
        src_ds = warplib.memwarp_multi_fn([src_fn,], res=src_fn, extent=extent, t_srs=src_fn)[0]

    #Now load source array with new mask
    src_ma_full = np.ma.array(iolib.ds_getma(src_ds), mask=mask)
    mask = None

    if args.out_fn is not None:
        src_fn_masked = args.out_fn
    else:
        src_fn_masked = os.path.splitext(src_fn)[0]+'_masked.tif'
    print("Writing out masked version of input raster: %s" % src_fn_masked )
    iolib.writeGTiff(src_ma_full, src_fn_masked, src_ds, create=True)

if __name__ == '__main__':
    main()
