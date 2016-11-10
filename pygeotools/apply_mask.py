#! /usr/bin/env python

#David Shean
#dshean@gmail.com

#Utility to mask input raster using the mask from another raster 
#Mask dataset can be a standard raster with nodata value specified or a binary mask
#If a binary mask, values should be:
#True (1) for masked, False (0) for valid - consistent with np.ma

import sys, os

import numpy as np

from pygeotools.lib import iolib
from pygeotools.lib import warplib

def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: %s raster.tif mask.tif" % os.path.basename(sys.argv[0]))

    dem_fn = sys.argv[1]
    mask_fn = sys.argv[2]

    dem_ds, mask_ds = warplib.memwarp_multi_fn([dem_fn, mask_fn], res='first', extent='first', t_srs='first')

    dem_ma_full = iolib.ds_getma(dem_ds)
    mask_ma_full = iolib.ds_getma(mask_ds)

    if mask_ma_full.std() != 0:
        #Input mask filename is actually a DEM, or other masked array
        #Extract mask
        mask = np.ma.getmaskarray(mask_ma_full)
    else:
        #Input mask filename is truly a mask, use directly
        #If input mask values are zero, valid values are nonzero
        #Bool True == 1, so need to invert
        if mask_ma_full.fill_value == 0:
            mask = ~((mask_ma_full.data).astype(bool))
        else:
            mask = (mask_ma_full.data).astype(bool)

    #Add dilation step for buffer

    #newmask = np.logical_or(np.ma.getmaskarray(dem_ma_full), mask)

    print("Creating new array with updated mask")
    dem_ma_full = np.ma.array(dem_ma_full, mask=mask)

    print("Writing out masked full-res DEM")
    dem_fn_masked = os.path.splitext(dem_fn)[0]+'_masked.tif'
    iolib.writeGTiff(dem_ma_full, dem_fn_masked, dem_ds, create=True, sparse=True)

if __name__ == '__main__':
    main()
