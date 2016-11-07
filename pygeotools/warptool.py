#! /usr/bin/env python

#David Shean
#dshean@gmail.com

#Command line utility to access memwarp functions in warplib

import os
import argparse

from lib import warplib
from lib import geolib

def main():
    #Can't specify arbitrary fn, res when limiting choices
    tr_choices = ['first', 'last', 'min', 'max', 'mean', 'med', 'source', '"fn"', '"res"']
    te_choices = ['first', 'last', 'intersection', 'union', 'source', '"fn"', '"extent"']
    t_srs_choices = ['first', 'last', '"fn"', '"proj4str"']
    r_choices = ['near', 'bilinear', 'cubic', 'cubicspline', 'lanczos', 'average', 'mode']
    
    parser = argparse.ArgumentParser(description='Utility to warp stacks of rasters to the same res/extent/proj')
    #parser.add_argument('-tr', default='first', choices=tr_choices, help='Output resolution')
    parser.add_argument('-tr', default='first', help='Output resolution (default: %(default)s)')
    parser.add_argument('-te', default='intersection', help='Output extent (default: %(default)s)')
    parser.add_argument('-t_srs', default='first', help='Output projection (default: %(default)s)')
    parser.add_argument('-r', type=str, default='cubic', help='Resampling algorithm (default: %(default)s)', choices=r_choices)
    parser.add_argument('-outdir', default=None, help='Specify output directory')
    parser.add_argument('src_fn_list', nargs='+', help='Input filenames (img1.tif img2.tif ...)')
    args = parser.parse_args()

    #Can also provide filename for any of the tr, te, t_srs options

    #Automatically fix UL, LR coords?

    #Extent format is minx, miny, maxx, maxy
    #extent = [-197712.13, -2288712.83, -169650.72, -2253490.42] 
    #res = 2.0

    print
    print "Input parameters"
    print "Resolution: %s" % str(args.tr)
    print "Extent: %s" % str(args.te)
    print "Projection: %s" % str(args.t_srs)
    print "Resampling alg: %s" % str(args.r)
    print
    
    ds_list = warplib.memwarp_multi_fn(args.src_fn_list, res=args.tr, extent=args.te, t_srs=args.t_srs, r=args.r)

    if args.outdir is not None: 
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

    for i, ds in enumerate(ds_list):
        #Note: the following doesn't work for mem filenames
        #outfn = os.path.splitext(ds.GetFileList()[0])[0]+'_warp.tif'
        outfn = os.path.splitext(args.src_fn_list[i])[0]+'_warp.tif'
        if args.outdir is not None: 
            outfn = os.path.join(args.outdir, os.path.split(outfn)[-1])
        #Only write out ds that are not empty
        if not geolib.ds_IsEmpty(ds):
            warplib.writeout(ds, outfn)     
        else:
            print "Output ds is empty"

if __name__ == "__main__":
    main()
