#! /usr/bin/env python

"""
Utility to create shp footprints for a list of input raster(s)

"""

#To do:
#Update to use kml
#Update with improved get_outline method
#Output merge filename

import sys
import os
import argparse

from osgeo import gdal

from pygeotools.lib import geolib

def getparser():
    parser = argparse.ArgumentParser(description="Create shp from input rasters and merge into one shp")
    parser.add_argument('-merge_fn', type=str, default='merge.shp', help='Output merge shp filename')
    parser.add_argument('fn_list', type=str, nargs='+', help='Input raster filename(s)')
    return parser

def get_geom(fn):
    shp_fn = None
    if os.path.exists(fn):
        shp_fn = os.path.splitext(fn)[0]+'.shp'
        ds = gdal.Open(fn)
        geom = geolib.get_outline(ds)
        geolib.geom2shp(geom, shp_fn, fields=True)
    return shp_fn

def main():
    parser = getparser()
    args = parser.parse_args()

    fn_list = args.fn_list
    merge_fn = args.merge_fn

    #Note: ogr_merge.sh goes from first to last input shp
    #Reverse order here to preserve original order, with first input on top
    fn_list = fn_list[::-1]
    print('%i input rasters' % len(fn_list))
    print(fn_list[0])

    import multiprocessing as mp
    pool = mp.Pool()
    results = pool.map(get_geom, fn_list)
    merge_fn_list = sorted(results, reverse=True)

    print('%i output shp' % len(merge_fn_list))
    print(merge_fn_list[0])

    #This is a hack to merge, should just create new output and write all features
    if len(merge_fn_list) > 1:
        print("Merging input shp")
        import subprocess
        cmd = ['ogr_merge.sh', merge_fn]
        cmd.extend(merge_fn_list)
        #print(cmd)
        subprocess.call(cmd, shell=False)

    #Now, add option to clean up intermediate shp - can generate 1000s of files

if __name__ == '__main__':
    main()
