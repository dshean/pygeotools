#! /usr/bin/env python

#David Shean
#dshean@gmail.com

#Clip input raster to polygons in input shapefile
#Finally ported clip_raster_by_shp.sh to Python

#TODO: Handle same arbitrary res/extent/t_srs as in warplib (isolate/generalize those functions)

import os
import sys
import argparse

import numpy as np  
from osgeo import ogr

from pygeotools.lib import iolib
from pygeotools.lib import geolib

def getparser():
    parser = argparse.ArgumentParser(description="Clip input raster by input shp polygons")
    #Should add support for similar arguments as in warplib - arbitrary extent, res, etc
    parser.add_argument('-extent', type=str, default='raster', choices=['raster','shp'], \
                        help='Desired output extent')
    parser.add_argument('r_fn', type=str, help='Input raster filename')
    parser.add_argument('shp_fn', type=str, help='Input shp filename')
    return parser

def main():
    parser = getparser()
    args = parser.parse_args()

    r_fn = args.r_fn
    if not os.path.exists(r_fn):
        sys.exit("Unable to find r_fn: %s" % r_fn)
        
    shp_fn = args.shp_fn
    #Convenience shortcut to clip to glacier polygons (global shp)
    #Requires demcoreg package: https://github.com/dshean/demcoreg
    if shp_fn == 'RGI' or shp_fn == 'rgi':
        from demcoreg.dem_mask import get_glacier_poly
        rgi_fn = get_glacier_poly() 
        shp_fn = rgi_fn

    if not os.path.exists(shp_fn):
        sys.exit("Unable to find shp_fn: %s" % shp_fn)

    extent=args.extent

    #Do the clipping
    r = geolib.raster_shpclip(r_fn, shp_fn, extent)

    #Write out
    out_fn = os.path.splitext(r_fn)[0]+'_shpclip.tif'
    #Note: passing r_fn here as the src_ds
    iolib.writeGTiff(r, out_fn, r_fn) 

if __name__ == "__main__":
    main()
