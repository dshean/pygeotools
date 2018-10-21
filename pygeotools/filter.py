#! /usr/bin/env python
"""
Command-line wrapper around raster filters in filtlib
"""

#Note: currently need to specify fn first, as -param accepts arbitrary number of arguments
#Need better way to record params than in filename - write history to header?
#Precision on float

import sys
import os
import argparse

import numpy as np

from pygeotools.lib import iolib
from pygeotools.lib import malib
from pygeotools.lib import filtlib
from pygeotools.lib import warplib

def getparser():
    filter_choices = ['range', 'absrange', 'perc', 'gauss', 'med', 'highpass', 'sigma', 'mad', 'dz']
    parser = argparse.ArgumentParser(description='Filter input raster')
    parser.add_argument('fn', help='Input filename (img1.tif)')
    parser.add_argument('--stats', action='store_true', help='Print stats before and after filtering')
    parser.add_argument('-outdir', default=None, help='Output directory')
    #Should implement subparser here to handle different number of args for different filter types
    #https://docs.python.org/2/library/argparse.html#sub-commands
    #Can call functions directly
    #Could specify sequence of filters here
    #Should accept arbitrary number of ordered filter operations as cli argument
    parser.add_argument('-filt', nargs=1, default='gauss', choices=filter_choices, help='Filter type (default: %(default)s)')
    #size is a param
    #parser.add_argument('-size', type=int, default=7, help='Filter size in pixels (default: %(default)s)')
    parser.add_argument('-param', nargs='+', default=None, help='Filter parameter list (e.g., size, min max, ref_fn min max)')
    return parser

def main():
    parser = getparser()
    args = parser.parse_args()

    fn = args.fn
    if not iolib.fn_check(fn):
        sys.exit("Unable to locate input file: %s" % fn)

    #Need some checks on these
    param = args.param

    print("Loading input raster into masked array")
    ds = iolib.fn_getds(fn)
    #Currently supports only single band operations
    r = iolib.ds_getma(ds, 1)

    #May need to cast input ma as float32 so np.nan filling works
    #r = r.astype(np.float32)
    #Want function that checks and returns float32 if necessary 
    #Should filter, then return original dtype

    r_fltr = r 

    #Loop through all specified input filters
    #for filt in args.filt:
    filt = args.filt[0]

    if len(param) == 1:
        param = param[0]
    param_str = ''

    if filt == 'range':
        #Range filter
        param = [float(i) for i in param[1:]]
        r_fltr = filtlib.range_fltr(r_fltr, param)
        param_str = '_{0:0.2f}-{1:0.2f}'.format(*param)
    elif filt == 'absrange':
        #Range filter of absolute values
        param = [float(i) for i in param[1:]]
        r_fltr = filtlib.absrange_fltr(r_fltr, param)
        param_str = '_{0:0.2f}-{1:0.2f}'.format(*param)
    elif filt == 'perc':
        #Percentile filter
        param = [float(i) for i in param[1:]]
        r_fltr = filtlib.perc_fltr(r, perc=param)
        param_str = '_{0:0.2f}-{1:0.2f}'.format(*param)
    elif filt == 'med':
        #Median filter
        param = int(param)
        r_fltr = filtlib.rolling_fltr(r_fltr, f=np.nanmedian, size=param)
        #r_fltr = filtlib.median_fltr(r_fltr, fsize=param, origmask=True)
        #r_fltr = filtlib.median_fltr_skimage(r_fltr, radius=4, origmask=True)
        param_str = '_%ipx' % param
    elif filt == 'gauss':
        #Gaussian filter (default)
        param = int(param)
        r_fltr = filtlib.gauss_fltr_astropy(r_fltr, size=param, origmask=False, fill_interior=False)
        param_str = '_%ipx' % param
    elif filt == 'highpass':
        #High pass filter
        param = int(param)
        r_fltr = filtlib.highpass(r_fltr, size=param)
        param_str = '_%ipx' % param
    elif filt == 'sigma':
        #n*sigma filter, remove outliers
        param = int(param)
        r_fltr = filtlib.sigma_fltr(r_fltr, n=param)
        param_str = '_n%i' % param
    elif filt == 'mad':
        #n*mad filter, remove outliers
        #Maybe better to use a percentile filter
        param = int(param)
        r_fltr = filtlib.mad_fltr(r_fltr, n=param)
        param_str = '_n%i' % param
    elif filt == 'dz':
        #Difference filter, need to specify ref_fn and range
        #Could let the user compute their own dz, then just run a standard range or absrange filter
        ref_fn = param[0]
        ref_ds = warplib.memwarp_multi_fn([ref_fn,], res=ds, extent=ds, t_srs=ds)[0]
        ref = iolib.ds_getma(ref_ds)
        param = [float(i) for i in param[1:]]
        r_fltr = filtlib.dz_fltr_ma(r, ref, rangelim=param)
        #param_str = '_{0:0.2f}-{1:0.2f}'.format(*param)
        param_str = '_{0:0.0f}_{1:0.0f}'.format(*param)
    else:
        sys.exit("No filter type specified")

    #Compute and print stats before/after
    if args.stats:
        print("Input stats:")
        malib.print_stats(r)
        print("Filtered stats:")
        malib.print_stats(r_fltr)
    
    #Write out
    dst_fn = os.path.splitext(fn)[0]+'_%sfilt%s.tif' % (filt, param_str)
    if args.outdir is not None:
        outdir = args.outdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        dst_fn = os.path.join(outdir, os.path.split(dst_fn)[-1])
    print("Writing out filtered raster: %s" % dst_fn)
    iolib.writeGTiff(r_fltr, dst_fn, ds) 

if __name__ == '__main__':
    main()
