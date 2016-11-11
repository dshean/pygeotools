#! /usr/bin/env python

#David Shean
#dshean@gmail.com

#This script will trim empty rows and columns (with nodata value) from an input raster

import sys
import os

from pygeotools.lib import iolib
from pygeotools.lib import malib

def main():
    if len(sys.argv) != 2:
        sys.exit('Usage: %s raster.tif' % os.path.basename(sys.argv[0]))

    src_fn = sys.argv[1]
    #This is a wrapper around gdal.Open()
    src_ds = iolib.fn_getds(src_fn)
    src_gt = src_ds.GetGeoTransform()

    print("Loading input raster into masked array")
    bma = iolib.ds_getma(src_ds)

    print("Computing min/max indices for mask")
    edge_env = malib.edgefind2(bma)

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
