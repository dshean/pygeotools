#! /usr/bin/env python

#David Shean
#dshean@gmail.com

#This script will trim empty rows and columns (with nodata value) from an input raster

import sys
import os
import argparse

from pygeotools.lib import iolib
from pygeotools.lib import malib

def getparser():
    parser = argparse.ArgumentParser(description="Remove NoData row/col from raster margins")
    parser.add_argument('src_fn', type=str, help='Input raster filename')
    return parser

def main():
    parser = getparser()
    args = parser.parse_args()

    src_fn = args.src_fn
    if not iolib.fn_check(src_fn):
        sys.exit("Unable to find src_fn: %s" % src_fn)

    #This is a wrapper around gdal.Open()
    src_ds = iolib.fn_getds(src_fn)
    src_gt = src_ds.GetGeoTransform()

    print("Loading input raster into masked array")
    bma = iolib.ds_getma(src_ds)

    print("Computing min/max indices for mask")
    edge_env = malib.edgefind2(bma).astype(int)

    print("Updating output geotransform")
    out_gt = list(src_gt)
    #This should be OK, as edge_env values are integer multiples, and the initial gt values are upper left pixel corner
    #Update UL_X
    out_gt[0] = src_gt[0] + src_gt[1]*edge_env[2]
    #Update UL_Y, note src_gt[5] is negative
    out_gt[3] = src_gt[3] + src_gt[5]*edge_env[0]
    out_gt = tuple(out_gt)

    #debug
    #print([0, bma.shape[0], 0, bma.shape[1]])
    #print(edge_env)
    #print(src_gt)
    #print(out_gt)

    out_fn = os.path.splitext(src_fn)[0]+'_trim.tif'
    print("Writing out: %s" % out_fn)
    #Extract valid subsection from input array
    #indices+1 are necessary to include valid row/col on right and bottom edges
    iolib.writeGTiff(bma[edge_env[0]:edge_env[1]+1, edge_env[2]:edge_env[3]+1], out_fn, src_ds, gt=out_gt)
    bma = None

if __name__ == '__main__':
    main()
