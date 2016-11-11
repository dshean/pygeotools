#! /usr/bin/env python

#David Shean
#dshean@gmail.com

#Little utility to create footprints for a list of input raster(s)

#To do:
#Update to use kml
#Update with improved get_outline method
#Output merge filename

import sys
import os

from osgeo import gdal

from pygeotools.lib import geolib

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: %s raster1.tif [raster2.tif raster3.tif ...]" % os.path.basename(sys.argv[0]))

    fn_list = sys.argv[1:]

    #Output filename
    merge_fn = 'merge.shp'

    #Note: ogr_merge.sh goes from first to last input shp
    #Reverse order here to preserve original order, with first input on top
    fn_list = fn_list[::-1]

    merge_fn_list = []
    for n,fn in enumerate(fn_list): 
        print(n, fn)
        if os.path.exists(fn):
            shp_fn = os.path.splitext(fn)[0]+'.shp'
            if not os.path.exists(shp_fn):
                ds = gdal.Open(fn)
                geom = geolib.get_outline(ds)
                geolib.geom2shp(geom, shp_fn, fields=True)
            merge_fn_list.append(shp_fn)

    #This is a hack to merge, should just create new output and write all features
    if len(merge_fn_list) > 1:
        print("Merging input shp")
        import subprocess
        cmd = ['ogr_merge.sh', merge_fn]
        cmd.extend(merge_fn_list)
        print(cmd)
        subprocess.call(cmd, shell=False)

if __name__ == '__main__':
    main()
