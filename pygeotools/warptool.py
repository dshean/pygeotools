#! /usr/bin/env python

#David Shean
#dshean@gmail.com

#Command line utility to access memwarp functions in warplib

import os
import argparse

from pygeotools.lib import warplib
from pygeotools.lib import geolib

def fmt_choices(param, choices):
    outstr = 'Valid choices for %s: {%s}\n' % (param, ','.join(choices))
    return outstr

def getparser():
    #Note: Don't want to formally specify these choices, as it limits specification of arbitrary filenames
    tr_choices = ['first', 'last', 'min', 'max', 'mean', 'med', 'source', '"fn"', '"res"']
    te_choices = ['first', 'last', 'intersection', 'union', 'source', '"fn"', '"xmin ymin xmax ymax"']
    t_srs_choices = ['first', 'last', '"fn"', '"proj4str"']
    #Should read these directly from GDAL
    r_choices = ['near', 'bilinear', 'cubic', 'cubicspline', 'lanczos', 'average', 'mode']

    epilog = fmt_choices('-tr', tr_choices)
    epilog += fmt_choices('-te', te_choices)
    epilog += fmt_choices('-t_srs', t_srs_choices)

    #Note, the formatter_class is a hack to show the choices in the epilog
    parser = argparse.ArgumentParser(description='Utility to warp stacks of rasters to the same res/extent/proj', \
            epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-tr', default='first', help='Output resolution (default: %(default)s)') 
    parser.add_argument('-te', default='intersection', help='Output extent (default: %(default)s)')
    parser.add_argument('-t_srs', default='first', help='Output projection (default: %(default)s)')
    parser.add_argument('-dst_ndv', type=float, default=None, help='No data value for output')
    parser.add_argument('-r', type=str, default='cubic', help='Resampling algorithm (default: %(default)s)', choices=r_choices)
    parser.add_argument('-outdir', default=None, help='Specify output directory')
    parser.add_argument('src_fn_list', nargs='+', help='Input filenames (img1.tif img2.tif ...)')
    return parser

def main():
    parser = getparser()
    args = parser.parse_args()

    #Can also provide filename for any of the tr, te, t_srs options

    #Automatically fix UL, LR coords?

    #Extent format is minx, miny, maxx, maxy
    #extent = [-197712.13, -2288712.83, -169650.72, -2253490.42] 
    #res = 2.0

    print("\nInput parameters")
    print("Resolution: %s" % str(args.tr))
    print("Extent: %s" % str(args.te))
    print("Projection: %s" % str(args.t_srs))
    print("Resampling alg: %s\n" % str(args.r))
    
    if args.outdir is not None: 
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

    #Use the direct write to disk functionality in diskwarp
    ds_list = warplib.diskwarp_multi_fn(args.src_fn_list, \
            res=args.tr, extent=args.te, t_srs=args.t_srs, r=args.r, outdir=args.outdir, dst_ndv=args.dst_ndv)

if __name__ == "__main__":
    main()
