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
    parser.add_argument('-extent', type=str, default='raster', choices=['raster','shp','intersection','union'], 
                        help='Desired output extent')
    parser.add_argument('-bbox', action='store_true', help='Clip raster to shp bounding box, but dont mask')
    parser.add_argument('-pad', type=float, default=None, help='Padding around shp extent, in raster units')
    parser.add_argument('-invert', action='store_true', help='Invert the input polygons before clipping')
    parser.add_argument('-out_fn', type=str, default=None, help='Output raster filename (default: *_shpclip.tif)')
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

    #Do the clipping
    r, r_ds = geolib.raster_shpclip(r_fn, shp_fn, extent=args.extent, bbox=args.bbox, pad=args.pad, invert=args.invert)

    #Write out
    out_fn = args.out_fn
    if out_fn is None:
        out_fn = os.path.splitext(r_fn)[0]+'_shpclip.tif'
    #Note: passing r_fn here as the src_ds
    iolib.writeGTiff(r, out_fn, r_ds) 

if __name__ == "__main__":
    main()
