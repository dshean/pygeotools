#! /usr/bin/env python

#David Shean
#dshean@gmail.com

#Library containing various functions for masked arrays

import sys
import os
import glob

import numpy as np
from osgeo import gdal

from pygeotools.lib import iolib

#Notes on geoma
#Note: Need better init overloading
#http://stackoverflow.com/questions/141545/overloading-init-in-python
#Might make more sense to create ma subclass, and add gdal ds as new object
#http://stackoverflow.com/questions/12597827/how-to-subclass-numpy-ma-core-masked-array
#http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

#Want to implement basic array indexing with map coordinates, display in plt

#See pyresample
#https://github.com/talltom/PyRaster/blob/master/rasterIO.py
#http://www2-pcmdi.llnl.gov/cdat/tutorials/training/cdat_2004/06-arrays-variables-etc.pdf
#http://docs.scipy.org/doc/numpy/user/basics.subclassing.html

#=======================
#Masked array stack
#=======================

#Want to add error attributes
#Want to make consistent with stack_count vs count keywords/attributes
class DEMStack:
    def __init__(self, fn_list=[], stack_fn=None, outdir=None, res=None, extent=None, srs=None, trend=False, med=False, stats=True, save=True, sort=True, datestack=True):
        self.sort = sort
        if self.sort:
            #This sorts filenames, should probably sort by datetime to be safe
            fn_list = sorted(fn_list, key=lambda x: os.path.split(x)[-1])
        self.fn_list = list(fn_list)
        self.stack_fn = stack_fn
        if not self.fn_list and stack_fn is None:
            raise ValueError('Must specify input filename list or existing stack filename')
        self.res = res
        self.extent = extent
        self.srs = srs 
        self.trend = trend
        self.med = med
        self.stats = stats
        self.save = save
        self.datestack = datestack
        #This is the minimum number of arrays in stack to compute trend
        self.n_thresh = 2
        self.min_dt_ptp = np.nan 

        #Use this to limit memory use and filesizes
        #self.dtype = 'float32'
        self.dtype = np.float32

        #Want to do this before filenames, etc. are determined
        #self.get_date_list() 
        #if sort:
        #    idx = np.argsort(self.date_list_o)
        #    self.date_list = self.date_list[idx]
        #    self.date_list_o = self.date_list_o[idx]
        #    self.fn_list = (np.array(self.fn_list)[idx]).tolist()

        #Determine appropriate stack filename
        if outdir is None:
            #Use directory of first filename
            if self.fn_list:
                self.outdir = os.path.split(self.fn_list[0])[0]
            else:
                self.outdir = os.getcwd()
        else:
            if os.path.exists(outdir):
                self.outdir = outdir
            else:
                raise IOError('Specified output directory does not exist')

        if self.stack_fn is None:
            if self.fn_list:
                self.get_stack_fn()
            else:
                raise ValueError('Must specify input filename list or existing stack filename')

        if os.path.exists(self.stack_fn):
            self.loadstack()
            #This was an attempt to ensure that input fn/res/extent is consistent with saved npz
            #Should really check to see if new extent falls within existing extent
            #Only regenerate if new extent is larger
            #Res check is not working correctly
            #print extent, self.extent
            #print res, self.res
            #if (self.extent != extent) or (self.res != res):
                #self.res = res
                #self.extent = extent
                #self.makestack()
                #if self.stats: 
                    #self.compute_stats()
                    #self.write_stats()
                #self.savestack()
        else:
            self.makestack()
            #Initialize source and error lists
            self.source = ['None' for i in self.fn_list]
            #TODO: This needs to be fixed, source_dict moved to stack_view.py 
            #self.get_source()
            self.error_dict_list = [None for i in self.fn_list]
            #TODO: This needs to be fixed, source_dict moved to stack_view.py 
            #self.get_error_dict_list()
            self.error = np.ma.zeros(len(self.fn_list))
            #TODO: This needs to be fixed, source_dict moved to stack_view.py 
            #self.get_error()
            self.get_date_list() 
            if sort:
                sort_idx = self.get_sortorder()
                if np.any(self.date_list_o != self.date_list_o[sort_idx]):
                    self.sort_in_place(sort_idx)
            self.finish()

    def finish(self):
        if self.datestack:
            #Only do this if we have valid dates
            if self.date_list_o.count() > 1: 
                #self.make_datestack()
                #self.write_datestack()
                self.compute_dt_stats()
                self.write_datestack()
        if self.stats: 
            self.compute_stats()
            if self.save:
                self.write_stats()
        if self.trend:
            if not self.datestack:
                self.make_datestack()
            self.compute_trend()
            if self.save:
                self.write_trend()
        if self.save:
            self.savestack()

    def get_stack_fn(self):
        self.stack_fn = os.path.splitext(os.path.split(self.fn_list[0])[-1])[0] + '_' \
                        + os.path.splitext(os.path.split(self.fn_list[-1])[1])[0] \
                        + '_stack_%i' % len(self.fn_list) + '.npz'
        self.stack_fn = os.path.join(self.outdir, self.stack_fn)

    #p = os.path.join(topdir, d, '*_warp.tif'))
    def get_fn_list(p):
        fn_list = glob.glob(p)
        return fn_list

    #This stores float
    def get_res(self):
        #Should check to make sure gt is defined
        self.res = np.abs([self.gt[1], self.gt[5]]).mean()

    #This stores list
    def get_extent(self):
        from pygeotools.lib import geolib
        #Should check to make sure gt is defined
        self.extent = geolib.gt_extent(self.gt, self.ma_stack.shape[2], self.ma_stack.shape[1])

    #This returns a dummy dataset for the stack
    #Useful for some applications
    def get_ds(self):
        nl = self.ma_stack.shape[1]
        ns = self.ma_stack.shape[2]
        gdal_dtype = iolib.np_gdal_dtype(np.dtype(self.dtype))
        m_ds = gdal.GetDriverByName('MEM').Create('', ns, nl, 1, gdal_dtype)
        m_gt = [self.extent[0], self.res, 0, self.extent[3], 0, -self.res]
        m_ds.SetGeoTransform(m_gt)
        #this should already be WKT
        m_ds.SetProjection(self.proj)
        return m_ds

    """
    #TODO: Need to clean up the source_dict and error_dict code below
    #This is pretty clunky
    def get_source(self):
        for i, fn in enumerate(self.fn_list):
            for key, d in source_dict.items():
                if d['fn_pattern'] in fn:
                    self.source[i] = key
                    break

    #Should probably just preserve the error dictionary here
    def get_error_dict_list(self):
        import error_analysis
        for i, fn in enumerate(self.fn_list):
            error_log = error_analysis.parse_pc_align_log(fn)
            if error_log is not None:
                self.error_dict_list[i] = error_log

    def get_error(self):
        for i, fn in enumerate(self.fn_list):
            if self.error_dict_list[i] is not None:
                self.error[i] = self.error_dict_list[i]['Output Sampled Median Error']
            elif self.source[i] is not 'None':
                if 'error_perc' in source_dict[self.source[i]]:
                    istat = fast_median(self.ma_stack[i])
                    #Probably want to avoid using max, as could have bogus values
                    #istat = calcperc(self.ma_stack[i], clim=(2,98))[1]
                    self.error[i] = source_dict[self.source[i]]['error_perc'] * istat
                else:
                    self.error[i] = source_dict[self.source[i]]['error']
    """

    def makestack(self):
        from pygeotools.lib import warplib
        print("Creating stack of %i files" % len(self.fn_list))
        #Jako front 
        #res = 16
        res = 'min'
        if self.res is not None:
            res=self.res
        #extent = '-195705.297256 -2286746.61662 -170642.601955 -2256442.61662'
        #extent = 'intersection'
        extent = 'union'
        if self.extent is not None:
            extent = self.extent
        srs='first'
        if self.srs is not None:
            srs = self.srs
        ds_list = warplib.memwarp_multi_fn(self.fn_list, res=res, extent=extent, t_srs=srs)
        #Check to eliminate empty datasets
        from pygeotools.lib import geolib
        #Returns True if empty
        bad_ds_idx = np.array([geolib.ds_IsEmpty(ds) for ds in ds_list])
        if np.any(bad_ds_idx):
            print("\n%i empty ds removed:" % len(bad_ds_idx.nonzero()[0]))
            print(np.array(self.fn_list)[bad_ds_idx])
            self.fn_list = np.array(self.fn_list)[~bad_ds_idx].tolist()
            print("%i valid input ds\n" % len(self.fn_list)) 
            self.get_stack_fn()
        print("Creating ma_stack")
        #Note: might not need ma here in the 0 axis - shouldn't be any missing data
        #self.ma_stack = np.ma.array([iolib.ds_getma(ds) for ds in ds_list], dtype=self.dtype)
        self.ma_stack = np.ma.array([iolib.ds_getma(ds) for ds in np.array(ds_list)[~bad_ds_idx]], dtype=self.dtype)
        #Might want to convert to proj4
        self.proj = ds_list[0].GetProjectionRef()
        self.gt = ds_list[0].GetGeoTransform()
        #Now set these for stack, regardless of input
        self.get_res()
        self.get_extent()

    def get_sortorder(self):
        sort_idx = np.argsort(self.date_list)
        return sort_idx

    def sort_in_place(self, sort_idx):
        self.fn_list = (np.array(self.fn_list)[sort_idx]).tolist()
        self.get_stack_fn()
        self.ma_stack = self.ma_stack[sort_idx]
        self.date_list = self.date_list[sort_idx]
        self.date_list_o = self.date_list_o[sort_idx]
        self.source = (np.array(self.source)[sort_idx]).tolist()
        self.error = self.error[sort_idx]
        self.error_dict_list = (np.array(self.error_dict_list)[sort_idx]).tolist()
        
    #This is depreciated, but is useful for computing mean, median or std
    #Create separate array of datetime objects
    def make_datestack(self):
        self.datestack = True
        print("Creating datestack")
        self.dt_stack = np.ma.copy(self.ma_stack).astype(self.dtype)
        for n, dt_o in enumerate(self.date_list_o):
            self.dt_stack[n].data[:] = dt_o
        self.dt_stack_min = np.ma.min(self.dt_stack, axis=0)
        self.dt_stack_max = np.ma.max(self.dt_stack, axis=0)
        self.dt_stack_ptp = np.ma.masked_equal((self.dt_stack_max - self.dt_stack_min), 0)
        self.dt_stack_center = self.dt_stack_min + self.dt_stack_ptp/2.0
        #self.dt_stack_mean = np.ma.mean(self.dt_stack, axis=0)

    def compute_dt_stats(self):
        self.datestack = True
        print("Computing date stats")
        allmask = np.ma.getmaskarray(self.ma_stack).all(axis=0)
        minidx = np.argmin(np.ma.getmaskarray(self.ma_stack), axis=0)
        maxidx = np.argmin(np.ma.getmaskarray(self.ma_stack[::-1]), axis=0)
        dt_stack_min = np.zeros(minidx.shape, dtype=self.dtype)
        dt_stack_max = np.zeros(maxidx.shape, dtype=self.dtype)
        for n, dt_o in enumerate(self.date_list_o):
            dt_stack_min[minidx == n] = dt_o
            dt_stack_max[maxidx == (len(self.date_list_o)-1 - n)] = dt_o
        self.dt_stack_min = np.ma.array(dt_stack_min, mask=allmask)
        self.dt_stack_max = np.ma.array(dt_stack_max, mask=allmask)
        self.dt_stack_ptp = np.ma.masked_equal((self.dt_stack_max - self.dt_stack_min), 0)
        self.dt_stack_center = self.dt_stack_min + self.dt_stack_ptp.filled(0)/2.0
            
        #Should pull out unmasked indices at each pixel along axis 0
        #Take min index along axis 0
        #Then create grids by pulling out corresponding value from date_list_o

    def write_datestack(self):
        #stat_list = ['dt_stack_ptp', 'dt_stack_mean', 'dt_stack_min', 'dt_stack_max', 'dt_stack_center']
        stat_list = ['dt_stack_ptp', 'dt_stack_min', 'dt_stack_max', 'dt_stack_center']
        if any([not hasattr(self, i) for i in stat_list]):
            #self.make_datestack()
            self.compute_dt_stats()
        print("Writing out datestack stats")
        #Create dummy ds - might want to use vrt here instead
        driver = gdal.GetDriverByName("MEM")
        ds = driver.Create('', self.dt_stack_ptp.shape[1], self.dt_stack_ptp.shape[0], 1, gdal.GDT_Float32)
        ds.SetGeoTransform(self.gt)
        ds.SetProjection(self.proj)
        #Write out with malib, should preserve ma type
        out_prefix = os.path.splitext(self.stack_fn)[0]
        iolib.writeGTiff(self.dt_stack_ptp, out_prefix+'_dt_ptp.tif', ds)
        self.dt_stack_ptp.set_fill_value(-9999)
        #iolib.writeGTiff(self.dt_stack_mean, out_prefix+'_dt_mean.tif', ds)
        #self.dt_stack_mean.set_fill_value(-9999)
        iolib.writeGTiff(self.dt_stack_min, out_prefix+'_dt_min.tif', ds)
        self.dt_stack_min.set_fill_value(-9999)
        iolib.writeGTiff(self.dt_stack_max, out_prefix+'_dt_max.tif', ds)
        self.dt_stack_max.set_fill_value(-9999)
        iolib.writeGTiff(self.dt_stack_center, out_prefix+'_dt_center.tif', ds)
        self.dt_stack_center.set_fill_value(-9999)

    #Note: want ot change the variable names min/max here
    #Might be better to save out as multiband GTiff here
    def savestack(self):
        print("Saving stack to: %s" % self.stack_fn)
        out_args = {}
        out_args['ma_stack_full'] = self.ma_stack.filled(np.nan)
        out_args['proj'] = str(self.proj)
        out_args['gt'] = self.gt
        out_args['res'] = self.res
        out_args['extent'] = self.extent
        out_args['n_thresh'] = self.n_thresh
        out_args['min_dt_ptp'] = self.min_dt_ptp
        out_args['fn_list'] = np.array(self.fn_list)
        out_args['source'] = np.array(self.source)
        out_args['error'] = self.error.filled(np.nan)
        out_args['error_dict_list'] = self.error_dict_list
        out_args['date_list'] = self.date_list.astype('str').filled('None')
        out_args['date_list_o'] = self.date_list_o.filled(np.nan)
        #Should really write out flags used for stack creation
        #out_args['flags']={'datestack':self.datestack, 'stats':self.stats, 'med':self.med, 'trend':self.trend, 'sort':self.sort, 'save':self.save} 
        if self.datestack:
            #out_args['dt_stack'] = self.dt_stack.filled(np.nan)
            #out_args['dt_mean'] = self.dt_stack_mean.filled(np.nan)
            out_args['dt_ptp'] = self.dt_stack_ptp.filled(np.nan)
            out_args['dt_min'] = self.dt_stack_min.filled(np.nan)
            out_args['dt_max'] = self.dt_stack_max.filled(np.nan)
            out_args['dt_center'] = self.dt_stack_center.filled(np.nan)
        if self.stats:
            out_args['count'] = self.stack_count.filled(0)
            out_args['mean'] = self.stack_mean.filled(np.nan)
            out_args['min'] = self.stack_min.filled(np.nan)
            out_args['max'] = self.stack_max.filled(np.nan)
            out_args['std'] = self.stack_std.filled(np.nan)
        if self.med:
            out_args['med'] = self.stack_med.filled(np.nan)
        if self.trend:
            out_args['trend'] = self.stack_trend.filled(np.nan)
            out_args['intercept'] = self.stack_intercept.filled(np.nan)
            out_args['detrended_std'] = self.stack_detrended_std.filled(np.nan)
            #out_args['rsquared'] = self.stack_rsquared.filled(np.nan)
        np.savez_compressed(self.stack_fn, **out_args)
        #Now write out a filename list for reference
        #Could also add metadata like extent, res, etc.
        #Might be best to dump as json
        list_fn = os.path.splitext(self.stack_fn)[0]+'_fn_list.txt'
        f = open(list_fn,'w')
        for i in self.fn_list:
            f.write('%s\n' % i)
        f.close()

    def loadstack(self):
        print("Loading stack from: %s" % self.stack_fn)
        data = np.load(self.stack_fn)
        self.fn_list = list(data['fn_list'])
        #Load flags originally used for stack creation
        #self.flags = data['flags']
        #{'datestack':self.datestack, 'stats':self.stats, 'med':self.med, 'trend':self.trend, 'sort':self.sort, 'save':self.save} 
        if 'source' in data:
            self.source = list(data['source'])
        else:
            self.source = ['None' for i in self.fn_list]
        if 'error' in data:
            self.error = np.ma.fix_invalid(data['error'], fill_value=-9999)
        else:
            self.error = np.ma.zeros(len(self.fn_list))
        if 'error_dict_list' in data:
            self.error_dict_list = data['error_dict_list'][()]
        else:
            self.error_dict_list = [None for i in self.fn_list]
        #This is a shortcut, should load from the data['date_list'] arrays
        if 'date_list_o' in data:
            from pygeotools.lib import timelib
            from datetime import datetime
            self.date_list_o = np.ma.fix_invalid(data['date_list_o'], fill_value=1.0)
            #This is a hack - need universal timelib time zone support or stripping
            self.date_list = np.ma.masked_equal([i.replace(tzinfo=None) for i in timelib.o2dt(self.date_list_o)], datetime(1,1,1))
        else:
            self.get_date_list()
        print("Loading ma stack")
        self.ma_stack = np.ma.fix_invalid(data['ma_stack_full']).astype(self.dtype)
        #Note: the str is an intermediate fix - all new stacks should have str written
        self.proj = str(data['proj'])
        #If we don't have gt, we're in trouble - can't recompute res/extent
        if 'gt' in data:
            self.gt = data['gt']
        else:
            print("No geotransform found in stack")
            #Check if res and extent are defined - can reconstruct
            #Should throw error
        #Note: Once we have gt, could just run get_res() and get_extent() to avoid the following
        #Or could check to make sure consistent
        #Some stacks in Oct 2015 and Nov 2015 did not have res/extent saved properly
        """
        if 'res' in data:
            if data['res'] != 'None':
                #self.res = float(data['res'])
                self.res = float(np.atleast_1d(data['res'])[0])
            else:
                self.get_res()
        else:
            self.get_res()
        if 'extent' in data:
            if data['extent'] != 'None':
                #self.extent = list(data['extent'])
                #self.extent = list(np.atleast_1d(data['extent'])[0])
                extent = np.atleast_1d(data['extent'])[0]
                if isinstance(extent, str):
                    self.extent = [float(x) for x in extent.split()]
                else:
                    self.extent = list(extent)
            else:
                self.get_extent()
        else:
            self.get_extent() 
        """
        #Just do this to be safe, if gt is bad, no point in proceeding
        self.get_res()
        self.get_extent()

        saveflag=False
        if self.datestack:
            #statlist = ['dt_stack', 'dt_mean', 'dt_ptp', 'dt_min', 'dt_max', 'dt_center']
            statlist = ['dt_ptp', 'dt_min', 'dt_max', 'dt_center']
            if all([s in data for s in statlist]):
                print("Loading datestack")
                #self.dt_stack = np.ma.fix_invalid(data['dt_stack']).astype(self.dtype)
                #self.dt_stack_mean = np.ma.fix_invalid(data['dt_mean'], fill_value=-9999).astype(self.dtype)
                self.dt_stack_ptp = np.ma.fix_invalid(data['dt_ptp'], fill_value=-9999).astype(self.dtype)
                self.dt_stack_min = np.ma.fix_invalid(data['dt_min'], fill_value=-9999).astype(self.dtype)
                self.dt_stack_max = np.ma.fix_invalid(data['dt_max'], fill_value=-9999).astype(self.dtype)
                self.dt_stack_center = np.ma.fix_invalid(data['dt_center'], fill_value=-9999).astype(self.dtype)
            else:
                if self.date_list_o.count() > 1: 
                    #self.make_datestack()
                    self.compute_dt_stats()
                    self.write_datestack()
                    saveflag=True
        if self.stats:
            #Could do this individually to save time
            statlist = ['count', 'mean', 'std', 'min', 'max']
            if self.med:
                statlist.append('med')
            if all([s in data for s in statlist]):
                print("Loading stats")
                self.stack_count = np.ma.masked_equal(data['count'], 0).astype(np.uint16)
                self.stack_mean = np.ma.fix_invalid(data['mean'], fill_value=-9999).astype(self.dtype)
                self.stack_std = np.ma.fix_invalid(data['std'], fill_value=-9999).astype(self.dtype)
                self.stack_min = np.ma.fix_invalid(data['min'], fill_value=-9999).astype(self.dtype)
                self.stack_max = np.ma.fix_invalid(data['max'], fill_value=-9999).astype(self.dtype)
                if self.med:
                    self.stack_med = np.ma.fix_invalid(data['med'], fill_value=-9999).astype(self.dtype)
            else:
                if self.ma_stack.shape[0] > 1:
                    self.compute_stats()
                    self.write_stats()
                    saveflag=True
        if self.trend:
            if 'n_thresh' in data: 
                self.n_thresh = data['n_thresh']
            if 'min_dt_ptp' in data:
                self.min_dt_ptp = data['min_dt_ptp']
            #statlist = ['trend', 'intercept', 'detrended_std', 'rsquared']
            statlist = ['trend', 'intercept', 'detrended_std']
            if all([s in data for s in statlist]):
                print("Loading trend")
                self.stack_trend = np.ma.fix_invalid(data['trend'], fill_value=-9999).astype(self.dtype)
                self.stack_intercept = np.ma.fix_invalid(data['intercept'], fill_value=-9999).astype(self.dtype)
                self.stack_detrended_std = np.ma.fix_invalid(data['detrended_std'], fill_value=-9999).astype(self.dtype)
                #self.stack_rsquared = np.ma.fix_invalid(data['rsquared'], fill_value=-9999).astype(self.dtype)
            else:
                if self.ma_stack.shape[0] >= self.n_thresh:
                    self.compute_trend()
                    self.write_trend()
                    saveflag=True
        if saveflag: 
            self.savestack()
        data.close()

    #This needs some work - will break with nonstandard filenames
    def get_date_list(self):
        from pygeotools.lib import timelib
        import matplotlib.dates
        from datetime import datetime
        #self.date_list = np.ma.array([dateutil.parser.parse(os.path.split(fn)[1][0:13], fuzzy=True) for fn in self.fn_list])
        #This will return None if no date in fn
        date_list = [timelib.fn_getdatetime(os.path.split(fn)[-1]) for fn in self.fn_list]
        self.date_list = np.ma.masked_equal([datetime(1,1,1) if d is None else d for d in date_list], datetime(1,1,1))
        #self.date_list = np.ma.array([dateutil.parser.parse(os.path.split(fn)[1][3:12], fuzzy=True) for fn in self.fn_list])
        self.date_list_o = np.ma.array([matplotlib.dates.date2num(d) for d in self.date_list.filled()], mask=self.date_list.mask)

    def compute_stats(self):
        print("Compute stack count")
        self.stack_count = np.ma.masked_equal(self.ma_stack.count(axis=0), 0).astype(np.uint16)
        self.stack_count.set_fill_value(0)
        print("Compute stack mean")
        self.stack_mean = self.ma_stack.mean(axis=0).astype(self.dtype)
        self.stack_mean.set_fill_value(-9999)
        print("Compute stack std")
        self.stack_std = self.ma_stack.std(axis=0).astype(self.dtype)
        #Only want to preserve values where count > 1
        self.stack_std.mask = (self.stack_count <= 1) 
        self.stack_std.set_fill_value(-9999)
        print("Compute stack min")
        self.stack_min = self.ma_stack.min(axis=0).astype(self.dtype)
        self.stack_min.set_fill_value(-9999)
        print("Compute stack max")
        self.stack_max = self.ma_stack.max(axis=0).astype(self.dtype)
        self.stack_max.set_fill_value(-9999)
        if self.med:
            print("Compute stack med")
            #For numpy >1.9.0
            if 'nanmedian' in dir(np):
                self.stack_med = nanfill(self.ma_stack, np.nanmedian, axis=0).astype(self.dtype)
            else:
                self.stack_med = np.ma.median(self.ma_stack, axis=0).astype(self.dtype)
            self.stack_med.set_fill_value(-9999)

    def compute_trend(self):
        print("Compute stack linear trend")
        self.linreg()
        self.stack_trend.set_fill_value(-9999)
        self.stack_intercept.set_fill_value(-9999) 
        self.stack_detrended_std.set_fill_value(-9999)
        #self.stack_rsquared.set_fill_value(-9999) 

    def write_stats(self):
        #if not hasattr(self, 'stack_count'):
        stat_list = ['stack_count', 'stack_mean', 'stack_std', 'stack_min', 'stack_max']
        if self.med:
            stat_list.append('stack_med')
        if any([not hasattr(self, i) for i in stat_list]):
            self.compute_stats()
        print("Writing out stats")
        #Create dummy ds - might want to use vrt here instead
        driver = gdal.GetDriverByName("MEM")
        ds = driver.Create('', self.ma_stack.shape[2], self.ma_stack.shape[1], 1, gdal.GDT_Float32)
        ds.SetGeoTransform(self.gt)
        ds.SetProjection(self.proj)
        #Write out with malib, should preserve ma type
        out_prefix = os.path.splitext(self.stack_fn)[0]
        iolib.writeGTiff(self.stack_count, out_prefix+'_count.tif', ds)
        iolib.writeGTiff(self.stack_mean, out_prefix+'_mean.tif', ds)
        iolib.writeGTiff(self.stack_std, out_prefix+'_std.tif', ds)
        iolib.writeGTiff(self.stack_min, out_prefix+'_min.tif', ds)
        iolib.writeGTiff(self.stack_max, out_prefix+'_max.tif', ds)
        if self.med:
            iolib.writeGTiff(self.stack_med, out_prefix+'_med.tif', ds)

    def write_trend(self):
        #stat_list = ['stack_trend', 'stack_intercept', 'stack_detrended_std', 'stack_rsquared']
        stat_list = ['stack_trend', 'stack_intercept', 'stack_detrended_std']
        if any([not hasattr(self, i) for i in stat_list]):
            self.compute_trend()
        print("Writing out trend")
        #Create dummy ds - might want to use vrt here instead
        driver = gdal.GetDriverByName("MEM")
        ds = driver.Create('', self.ma_stack.shape[2], self.ma_stack.shape[1], 1, gdal.GDT_Float32)
        ds.SetGeoTransform(self.gt)
        ds.SetProjection(self.proj)
        #Write out with malib, should preserve ma type
        out_prefix = os.path.splitext(self.stack_fn)[0]
        iolib.writeGTiff(self.stack_trend, out_prefix+'_trend.tif', ds)
        iolib.writeGTiff(self.stack_intercept, out_prefix+'_intercept.tif', ds)
        iolib.writeGTiff(self.stack_detrended_std, out_prefix+'_detrended_std.tif', ds)
        #iolib.writeGTiff(self.stack_rsquared, out_prefix+'_rsquared.tif', ds)
    
    """
    def linreg_mstats(self):
        #import scipy.mstats
        from scipy.mstats import linregress
        x = self.date_list_o
        out = np.zeros_like(self.ma_stack[0])
        for row in np.arange(self.ma_stack.shape[1]):
            print '%i of %i rows' % (row, self.ma_stack.shape[1])
            for col in np.arange(self.ma_stack.shape[2])
                out[row, col] = linregress(x, self.ma_stack[:, row, col])[0]
        #out is in units of m/day since x is ordinal date
        out *= 365.25
        self.stack_trend = out
    """

    #Compute linear regression for every pixel in stack
    def linreg(self, rsq=False, conf_test=False):
        #Only compute where we have n_min unmasked values in time
        if self.stats:
            count = self.stack_count
        else:
            count = np.ma.masked_equal(self.ma_stack.count(axis=0), 0).astype(np.uint16)
            count.set_fill_value(0)
        print("Excluding pixels with count < %i" % self.n_thresh)
        valid_idx = (count.data >= self.n_thresh)
        #Want to avoid computing trend where dt is small
        #Note, actual minimum depends on magnitude of trend
        #Should force datestack here
        if not self.datestack:
            self.compute_dt_stats()
            #self.write_datestack()
        if np.isnan(self.min_dt_ptp):
            #max_dt_ptp = self.dt_stack_ptp.max()
            #This could fail if stack contains a small number of inputs
            max_dt_ptp = calcperc(self.dt_stack_ptp, (4, 96))[1]
            #If no datestack
            #max_dt_ptp = np.ptp(calcperc(self.date_list_o, (4, 96)))
            self.min_dt_ptp = 0.10 * max_dt_ptp
        print("Excluding pixels with dt range < %0.2f days" % self.min_dt_ptp) 
        valid_idx = valid_idx & (self.dt_stack_ptp > self.min_dt_ptp)
        y_orig = self.ma_stack[:, valid_idx]
        #Reshape to 2D
        #origshape = self.ma_stack.shape
        #newshape = (origshape[0], origshape[1] * origshape[2])
        #y = self.ma_stack.reshape(newshape)
        #Extract mask for axis 0 - invert, True where data is available
        mask = ~(np.ma.getmaskarray(y_orig))
        #Remove masks, fills with fill_value
        y = y_orig.data
        #Independent variable is time ordinal
        x = self.date_list_o
        x_mean = x.mean()
        x = x.data
        #Prepare matrices
        X = np.c_[x, np.ones_like(x)]
        a = np.swapaxes(np.dot(X.T, (X[None, :, :] * mask.T[:, :, None])), 0, 1)
        b = np.dot(X.T, (mask*y))
        #Solve for slope/intercept
        print("Solving for trend")
        r = np.linalg.solve(a, b.T)
        #Reshape to original dimensions
        #r = r.reshape(origshape[1], origshape[2], 2)

        #Create output grids with original dimensions
        slope = np.ma.masked_all_like(self.ma_stack[0])
        intercept = np.ma.masked_all_like(self.ma_stack[0])
        detrended_std = np.ma.masked_all_like(self.ma_stack[0])

        #Fill in the valid indices
        slope[valid_idx] = r[:,0]
        intercept[valid_idx] = r[:,1]
        y_fit = r[:,0]*np.ma.array(x[:,None]*mask, mask=y_orig.mask) + r[:,1]
        resid = y_orig - y_fit
        #Note: Should be able to compute std of resid relative to resid mean, as lsq resid mean will be 0
        resid_std = resid.std(axis=0).data
        #resid_std = np.sqrt(np.sum(resid**2, axis=0)/resid.count(axis=0))
        detrended_std[valid_idx] = resid_std

        if rsq:
            rsquared = np.ma.masked_all_like(self.ma_stack[0])
            SStot = np.sum((y_orig - y_orig.mean(axis=0))**2, axis=0).data
            SSres = np.sum(resid**2, axis=0).data
            count = y_orig.count(axis=0)
            r2 = 1 - (SSres/SStot)
            rsquared[valid_idx] = r2
            #rmse = np.ma.masked_all_like(self.ma_stack[0])
            #Rrmse = np.sqrt(SSres/count)
            #rmse[valid_idx] = Rrmse

        if conf_test:
            SE = np.sqrt(SSres/(count - 2)/np.sum((x - x_mean)**2, axis=0))
            T0 = r[:,0]/SE
            alpha = 0.05
            ta = np.zeros_like(r2)
            from scipy.stats import t
            for c in np.unique(count):
                t1 = abs(t.ppf(alpha/2.0,c-2))
                ta[(count == c)] = t1
            sig = np.logical_and((T0 > -ta), (T0 < ta))
            sigmask = np.zeros_like(valid_idx, dtype=bool)
            sigmask[valid_idx] = ~sig
            #SSerr = SStot - SSres
            #F0 = SSres/(SSerr/(count - 2))
            #from scipy.stats import f
            #    f.cdf(sig, 1, c-2)
            slope = np.ma.array(slope, mask=~sigmask)
            intercept = np.ma.array(intercept, mask=~sigmask)
            detrended_std = np.ma.array(detrended_std, mask=~sigmask)
            rsquared = np.ma.array(rsquared, mask=~sigmask)
        
        #slope is in units of m/day since x is ordinal date
        slope *= 365.25
        #Filter out clearly bogus values here?
        self.stack_trend = np.ma.array(slope, dtype=self.dtype)
        self.stack_intercept = np.ma.array(intercept, dtype=self.dtype)
        self.stack_detrended_std = np.ma.array(detrended_std, dtype=self.dtype)
        #self.stack_rsquared = np.ma.array(rsquared, dtype=self.dtype)
    
    def mean_hillshade(self):
        if hasattr(self, 'stack_med'):
            in_fn = os.path.splitext(self.stack_fn)[0]+'_med.tif'
        elif hasattr(self, 'stack_mean'):
            in_fn = os.path.splitext(self.stack_fn)[0]+'_mean.tif'
        else:
            self.compute_stats()
            in_fn = os.path.splitext(self.stack_fn)[0]+'_mean.tif'
        if not os.path.exists(in_fn):
            self.write_stats()
        hs_fn = os.path.splitext(in_fn)[0]+'_hs.tif'
        if not os.path.exists(hs_fn):
            from pygeotools.lib import geolib
            print("Generate shaded relief from mean")
            self.stack_mean_hs = geolib.gdaldem_wrapper(in_fn, 'hs')
        else:
            self.stack_mean_hs = iolib.fn_getma(hs_fn)

def stack_smooth(s_orig, size=7, save=False):
    from copy import deepcopy
    from pygeotools.lib import filtlib
    print("Copying original DEMStack")
    s = deepcopy(s_orig)
    s.stack_fn = os.path.splitext(s_orig.stack_fn)[0]+'_smooth%ipx.npz' % size

    #Loop through each array and smooth
    print("Smoothing all arrays in stack with %i px gaussian filter" % size)
    for i in range(s.ma_stack.shape[0]):
        print('%i of %i' % (i+1, s.ma_stack.shape[0]))
        s.ma_stack[i] = filtlib.gauss_fltr_astropy(s.ma_stack[i], size=size)

    if s.stats:
        s.compute_stats()
        if save:
            s.write_stats()
    #Update datestack
    if s.datestack and s.date_list_o.count() > 1:
        s.compute_dt_stats()
        if save:
            s.write_datestack()
    #Update trend 
    if s.trend:
        s.compute_trend()
        if save:
            s.write_trend()
    if save:
        s.savestack()
    return s

#This will reduce stack extent 
#Extent should be list [xmin, ymin, xmax, ymax]
#extent = [-1692221, -365223, -1551556, -245479]
#Should also take pixel indices
def stack_clip(s_orig, extent, out_stack_fn=None, copy=True, save=False):
    #Should check for valid extent

    #This is not memory efficient, but is much simpler
    #To be safe, if we are saving out, create a copy to avoid overwriting
    if copy or save:
        from copy import deepcopy
        print("Copying original DEMStack")
        s = deepcopy(s_orig)
    else:
        #Want to be very careful here, as we could overwrite the original file
        s = s_orig

    from pygeotools.lib import geolib
    gt = s.gt
    s_shape = s.ma_stack.shape[1:3]

    #Compute pixel bounds for input extent
    min_x_px, max_y_px = geolib.mapToPixel(extent[0], extent[1], gt)
    max_x_px, min_y_px = geolib.mapToPixel(extent[2], extent[3], gt)

    #Clip to stack extent and round to whole integers
    min_x_px = int(max(0, min_x_px)+0.5)
    max_x_px = int(min(s_shape[1], max_x_px)+0.5)
    min_y_px = int(max(0, min_y_px)+0.5)
    max_y_px = int(min(s_shape[0], max_y_px)+0.5)
  
    #Clip the stack
    x_slice = slice(min_x_px,max_x_px)
    y_slice = slice(min_y_px,max_y_px)
    s.ma_stack = s.ma_stack[:, y_slice, x_slice]

    #Now update geospatial info
    #This returns the pixel center in map coordinates
    #Want to remove 0.5 px offset for upper left corner in gt
    out_ul = geolib.pixelToMap(min_x_px - 0.5, min_y_px - 0.5, gt)
   
    #Update stack geotransform
    s.gt[0] = out_ul[0]
    s.gt[3] = out_ul[1]
    #Update new stack extent
    s.get_extent()

    #Check for and discard emtpy arrays 
    #Might be faster to reshape then np.ma.count(s.ma_stack, axis=1)
    count_list = np.array([i.count() for i in s.ma_stack])
    idx = count_list > 0 
    
    #Output subset with valid data in next extent
    #fn_list, source, error, error_dict_list, date_list, date_list_o
    #Note: no need to copy again
    s_sub = get_stack_subset(s, idx, out_stack_fn=out_stack_fn, copy=False, save=False) 

    print("Orig filename:", s_orig.stack_fn)
    print("Orig extent:", s_orig.extent)
    print("Orig dimensions:", s_orig.ma_stack.shape)
    print("Input extent:", extent)
    print("New filename:", s_sub.stack_fn)
    print("New extent:", s_sub.extent)
    print("New dimensions:", s_sub.ma_stack.shape)
    
    if save:
        if os.path.abspath(s_orig.stack_fn) == os.path.abspath(s_sub.stack_fn):
            print("Original stack would be overwritten!")
            print("Skipping save")
        else:
            s_sub.save = True
            s_sub.savestack()

    #The following should be unchanged by clip - it is more efficient to clip thes, but easier to regenerate
    #if s.stats:
    #stack_count, stack_mean, stack_min, stack_max, stack_std
    #s.stack_min = s.stack_min[y_slice, x_slice]
    #if s.datestack:
    #dt_ptp, dt_min, dt_max, dt_center
    #if s.med:
    #stack_med
    #if s.trend:
    #trend, intercept, detrended_std
    #Recompute stats/etc

    return s_sub

#Note: See stack_filter.py for examples of generating idx for subset of stack
#This will pull out a subset of an existing stack
def get_stack_subset(s_orig, idx, out_stack_fn=None, copy=True, save=False):
    #This must be a numpy boolean array
    idx = np.array(idx)
    if np.any(idx):
        #This is not memory efficient, but is much simpler
        #To be safe, if we are saving out, create a copy to avoid overwriting
        if copy or save:
            from copy import deepcopy
            print("Copying original DEMStack")
            s = deepcopy(s_orig)
        else:
            #Want to be very careful here, as we could overwrite the original file
            s = s_orig
        #Update fn_list
        #Note: need to change fn_list to np.array - object array, allows longer strings
        #s.fn_list = s.fn_list[idx]
        print("Original stack: %i" % len(s_orig.fn_list))
        s.fn_list = (np.array(s.fn_list)[idx]).tolist()
        print("Filtered stack: %i" % len(s.fn_list))
        #Update date_lists
        s.date_list = s.date_list[idx]
        s.date_list_o = s.date_list_o[idx]
        #Update ma
        s.ma_stack = s.ma_stack[idx]
        #Update source/error
        #s.source = s.source[idx]
        s.source = (np.array(s.source)[idx]).tolist()
        s.error = s.error[idx]
        s.error_dict_list = np.array(s.error_dict_list)[idx]
        #Update stack_fn
        #out_stack_fn should be full path, with npz
        if out_stack_fn is None:
            s.stack_fn = None
            s.get_stack_fn()
        else:
            s.stack_fn = out_stack_fn
        #Check to make sure we are not going to overwrite
        if os.path.abspath(s_orig.stack_fn) == os.path.abspath(s.stack_fn):
            print("Warning: new stack has identical filename: %s" % s.stack_fn)
            print("As a precaution, new stack will not be saved")
            save = False
        s.save = save
        #Update stats
        if s.stats:
            s.compute_stats()
            if save:
                s.write_stats()
        #Update datestack
        if s.datestack and s.date_list_o.count() > 1:
            s.compute_dt_stats()
            if save:
                s.write_datestack()
        #Update trend 
        if s.trend:
            s.compute_trend()
            if save:
                s.write_trend()
        if save:
            s.savestack()
    else:
        print("No valid entries for input index array")
        s = None
    return s

#First stack should be "master" - preserve stats, etc
def stack_merge(s1, s2, out_stack_fn=None, sort=True, save=False):
    from pygeotools.lib import geolib
    from copy import deepcopy
    #Assumes input stacks have identical extent, resolution, and projection
    if s1.ma_stack.shape[1:3] != s2.ma_stack.shape[1:3]:
        print(s1.ma_stack.shape)
        print(s2.ma_stack.shape)
        sys.exit('Input stacks must have identical array dimensions')
    if not geolib.extent_compare(s1.extent, s2.extent):
        print(s1.extent)
        print(s2.extent)
        sys.exit('Input stacks must have identical extent')
    if not geolib.res_compare(s1.res, s2.res):
        print(s1.res)
        print(s2.res)
        sys.exit('Input stacks must have identical res')

    print("\nCombining fn_list and ma_stack")
    fn_list = np.array(s1.fn_list + s2.fn_list)

    if sort:
        #Sort based on filenames (should be datesort)
        sort_idx = np.argsort([os.path.split(x)[-1] for x in fn_list])
    else:
        sort_idx = Ellipsis

    #Now pull out final, sorted order
    fn_list = fn_list[sort_idx]
    ma_stack = np.ma.vstack((s1.ma_stack, s2.ma_stack))[sort_idx]
    #date_list = np.ma.dstack(s1.date_list, s2.date_list)
    #date_list_o = np.ma.dstack(s1.date_list_o, s2.date_list_o)
    source = np.array(s1.source + s2.source)[sort_idx]
    error = np.ma.concatenate([s1.error, s2.error])[sort_idx]
    #These are object arrays
    error_dict_list = np.concatenate([s1.error_dict_list, s2.error_dict_list])[sort_idx]

    print("Creating copy for new stack")
    s = deepcopy(s1)
    s.fn_list = list(fn_list)
    s.ma_stack = ma_stack
    s.source = list(source)
    s.error = error
    s.error_dict_list = error_dict_list
    #This will use original stack outdir
    if not out_stack_fn:
        s.get_stack_fn()
    else:
        s.stack_fn = out_stack_fn
    s.get_date_list()

    #These will preserve trend from one stack if present in only one stack
    #Useful when combining surface topo and bed topo
    if s1.datestack and s2.datestack:
        s.compute_dt_stats()
    if save and s1.datestack:
        s.write_datestack()

    if s1.stats and s2.stats:
        s.compute_stats()
    if save and s1.stats:
        s.write_stats()

    if s1.trend and s2.trend:
        s.compute_trend()
    if save and s1.trend:
        s.write_trend()

    if save:
        s.savestack()
    return s

#Compute linear regression for every pixel in stack
def ma_linreg(ma_stack, dt_list, n_thresh=2, min_dt_ptp=None, rsq=False, conf_test=False):
    from pygeotools.lib import timelib
    date_list_o = timelib.np_dt2o(dt_list)
    date_list_o.set_fill_value(0.0)
    #Only compute where we have n_thresh unmasked values in time
    count = np.ma.masked_equal(ma_stack.count(axis=0), 0).astype(np.uint16)
    print("Excluding pixels with count < %i" % n_thresh)
    valid_idx = (count.data >= n_thresh)
    #Want to avoid computing trend where dt is small
    #Note, actual minimum depends on magnitude of trend
    if min_dt_ptp is None:
        max_dt_ptp = np.ptp(calcperc(date_list_o, (4, 96)))
        min_dt_ptp = 0.10 * max_dt_ptp
    y_orig = ma_stack[:, valid_idx]
    #Extract mask for axis 0 - invert, True where data is available
    mask = ~y_orig.mask
    #Remove masks, fills with fill_value
    y = y_orig.data
    #Independent variable is time ordinal
    x = date_list_o
    x_mean = x.mean()
    x = x.data
    #Prepare matrices
    X = np.c_[x, np.ones_like(x)]
    a = np.swapaxes(np.dot(X.T, (X[None, :, :] * mask.T[:, :, None])), 0, 1)
    b = np.dot(X.T, (mask*y))
    #Solve for slope/intercept
    print("Solving for trend")
    r = np.linalg.solve(a, b.T)
    #Reshape to original dimensions
    #r = r.reshape(origshape[1], origshape[2], 2)

    #Create output grids with original dimensions
    slope = np.ma.masked_all_like(ma_stack[0])
    intercept = np.ma.masked_all_like(ma_stack[0])
    detrended_std = np.ma.masked_all_like(ma_stack[0])

    #Fill in the valid indices
    slope[valid_idx] = r[:,0]
    intercept[valid_idx] = r[:,1]
    y_fit = r[:,0]*np.ma.array(x[:,None]*mask, mask=y_orig.mask) + r[:,1]
    resid = y_orig - y_fit
    #Note: Should be able to compute std of resid relative to resid mean, as lsq resid mean will be 0
    resid_std = resid.std(axis=0).data
    #resid_std = np.sqrt(np.sum(resid**2, axis=0)/resid.count(axis=0))
    detrended_std[valid_idx] = resid_std

    if rsq:
        rsquared = np.ma.masked_all_like(ma_stack[0])
        SStot = np.sum((y_orig - y_orig.mean(axis=0))**2, axis=0).data
        SSres = np.sum(resid**2, axis=0).data
        count = y_orig.count(axis=0)
        r2 = 1 - (SSres/SStot)
        rsquared[valid_idx] = r2

    if conf_test:
        SE = np.sqrt(SSres/(count - 2)/np.sum((x - x_mean)**2, axis=0))
        T0 = r[:,0]/SE
        alpha = 0.05
        ta = np.zeros_like(r2)
        from scipy.stats import t
        for c in np.unique(count):
            t1 = abs(t.ppf(alpha/2.0,c-2))
            ta[(count == c)] = t1
        sig = np.logical_and((T0 > -ta), (T0 < ta))
        sigmask = np.zeros_like(valid_idx, dtype=bool)
        sigmask[valid_idx] = ~sig
        #SSerr = SStot - SSres
        #F0 = SSres/(SSerr/(count - 2))
        #from scipy.stats import f
        #    f.cdf(sig, 1, c-2)
        slope = np.ma.array(slope, mask=~sigmask)
        intercept = np.ma.array(intercept, mask=~sigmask)
        detrended_std = np.ma.array(detrended_std, mask=~sigmask)
        rsquared = np.ma.array(rsquared, mask=~sigmask)
    
    #slope is in units of m/day since x is ordinal date
    slope *= 365.25
    return slope, intercept, detrended_std

#=======================
#Masked array edge find 
#=======================

def get_edges(a, convex=False):
    a = checkma(a)
    #Need to deal with RGB images here
    #Need to be careful, probably want to take minimum value from masks
    if a.ndim == 3:
        #Assume that the same mask applies to each band
        #Only create edges for one band
        b = a[:,:,0]
        #Could convert to HSL and do edges for L channel
        #b = a[:,:,0].mask + a[:,:,1].mask + a[:,:,2].mask
    else:
        b = a

    #Compute edges along both axes, need both to handle undercuts
    #These are inclusive, indices indicate position of unmasked data
    edges0 = np.ma.notmasked_edges(b, axis=0)
    edges1 = np.ma.notmasked_edges(b, axis=1)
    edges = np.array([np.concatenate([edges0[0][0], edges0[1][0], edges1[1][0], edges1[0][0]]), np.concatenate([edges0[0][1], edges0[1][1], edges1[1][1], edges1[0][1]])])

    #This is a rough outline - needs testing
    if convex:
        from scipy.spatial import ConvexHull
        #hull = ConvexHull(edges.T)
        #edges = edges.T[hull.simplices]
        #This is in scipy v0.14
        #edges0 = edges1 = hull.vertices

    return edges0, edges1, edges

def get_edgemask(a, edge_env=False, convex=False, dilate=False):
    a = checkma(a)
    #Need to deal with RGB images here
    #Need to be careful, probably want to take minimum value from masks
    if a.ndim == 3:
        #Assume that the same mask applies to each band
        #Only create edges for one band
        b = a[:,:,0]
        #Could convert to HSL and do edges for L channel
        #b = a[:,:,0].mask + a[:,:,1].mask + a[:,:,2].mask
    else:
        b = a
    
    #Get pixel locations of edges
    edges0, edges1, edges = get_edges(b, convex)

    #Compute min/max indices
    #minrow, maxrow, mincol, maxcol
    #edge_bbox = [edges0[0][0].min(), edges0[1][0].max() + 1, edges0[0][1].min(), edges0[1][1].max() + 1]
    edge_bbox = [edges[0].min(), edges[0].max() + 1, edges[1].min(), edges[1].max() + 1]

    #Initialize new mask arrays
    #Need both to deal with undercuts
    colmask = np.empty_like(b.mask)
    colmask[:] = True
    rowmask = np.empty_like(b.mask)
    rowmask[:] = True

    #Loop through each item in the edge list
    #i is index, col is the column number listed at index i
    #unmask pixels between specified row numbers at index i
    for i,col in enumerate(edges0[0][1]):
        colmask[edges0[0][0][i]:edges0[1][0][i], col] = False

    for j,row in enumerate(edges1[0][0]):
        rowmask[row, edges1[0][1][j]:edges1[1][1][j]] = False

    #Combine the two masks with or operator
    newmask = np.logical_or(colmask, rowmask)

    if dilate:
        print("Dilating edgemask")
        import scipy.ndimage as ndimage 
        n = 3
        #Note: this results in unmasked elements near image corners
        #This will erode True values, which correspond to masked elements
        #Effect is to expand the unmasked area
        #newmask = ndimage.morphology.binary_erosion(newmask, iterations=n)
        #Now dilate to return to original size
        #newmask = ndimage.morphology.binary_dilation(newmask, iterations=n)
        #This is a more logical approach, dilating unmasked areas
        newmask = ~ndimage.morphology.binary_dilation(~newmask, iterations=n)
        newmask = ~ndimage.morphology.binary_erosion(~newmask, iterations=n)

    if edge_env:
        return newmask, edge_bbox
    else:
        return newmask

#This will update the mask to remove interior holes from unmasked region
def apply_edgemask(a, trim=False):
    newmask, edge_bbox = get_edgemask(a, edge_env=True)

    if a.ndim > 2:
        for i in range(a.ndim):
            a.mask[:,:,i] = newmask
    else:
        a.mask[:] = newmask
    
    #Option to return a trimmed array
    if trim:
        return a[edge_bbox[0]:edge_bbox[1], edge_bbox[2]:edge_bbox[3]]
    else:
        return a

#This will trim an array to the unmasked region
#Want to update gt here as well - see ndvtrim.py
def masktrim(a):
    a = checkma(a)
    idx = np.ma.notmasked_edges(a, axis=0)
    minrow = idx[0][0].min()
    maxrow = idx[1][0].max()
    mincol = idx[0][1].min()
    maxcol = idx[1][1].max()
    #print minrow, maxrow, mincol, maxcol
    #Want to make sure these are inclusive
    return a[minrow:maxrow, mincol:maxcol]

#Return a common mask for a set of input ma
def common_mask(ma_list, apply=False):
    if type(ma_list) is not list:
        print("Input must be list of masked arrays")
        return None
    #Note: a.mask will return single False if all elements are False
    #np.ma.getmaskarray(a) will return full array of False
    #ma_list = [np.ma.array(a, mask=np.ma.getmaskarray(a), shrink=False) for a in ma_list]
    a = np.ma.array(ma_list, shrink=False)
    #Check array dimensions
    #Check dtype = bool
    #Masked values are listed as true, so want to return any()
    #a+b+c - OR (any)
    mask = a.mask.any(axis=0)
    #a*b*c - AND (all)
    #return a.all(axis=0)
    if apply:
        return [np.ma.array(b, mask=mask) for b in ma_list] 
    else:
        return mask

#This will attempt to remove islands
#Not sure about what happens if the dilation hits the edge of the array
#Should look into binary_closing 
#Also, ndimage now has greyscale support for closing holes
#http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.morphology.grey_closing.html
def mask_islands(a, iterations=3):
    import scipy.ndimage as ndimage 
    a = checkma(a)
    #newmask = a.mask
    newmask = np.ma.getmaskarray(a)
    newmask = ndimage.morphology.binary_dilation(newmask, iterations=iterations)
    newmask = ndimage.morphology.binary_erosion(newmask, iterations=iterations)
    return np.ma.array(a, mask=newmask)

def mask_dilate(a, iterations=1, erode=False):
    import scipy.ndimage as ndimage 
    a = checkma(a)
    if erode:
        a = mask_islands(a, iterations)
    newmask = (~np.ma.getmaskarray(a))
    newmask = ndimage.morphology.binary_dilation(newmask, iterations=iterations)
    return ~newmask

#This has not been tested thoroughly
def mask_erode(a, iterations=1, erode=False):
    import scipy.ndimage as ndimage 
    a = checkma(a)
    if erode:
        a = mask_islands(a, iterations)
    newmask = (np.ma.getmaskarray(a))
    newmask = ndimage.morphology.binary_dilation(newmask, iterations=iterations)
    return newmask

#This will fill internal holes in the mask
#This should be used to mask outer edges after inpainting or gdal_fillnodata
def maskfill(a, iterations=1, erode=False):
    import scipy.ndimage as ndimage 
    a = checkma(a)
    if erode:
        a = mask_islands(a, iterations)
    bmask = (~np.ma.getmaskarray(a))
    bmask_filled = ndimage.morphology.binary_fill_holes(bmask)
    #This will create valid values with a.filled in the original ma
    #a_erode.mask[:] = ~bmask_filled
    #return a_erode
    return ~bmask_filled

def maskfill_edgeinclude(a, iterations=1, erode=False):
    import scipy.ndimage as ndimage
    a = checkma(a)
    if erode: 
        a = mask_islands(a, iterations=1)
    #This is the dilation version
    #newmask = ~np.ma.getmaskarray(a)
    #newmask = ndimage.morphology.binary_dilation(newmask, iterations=iterations)
    #newmask = ndimage.morphology.binary_dilation(~newmask, iterations=iterations)
    #And the erosion version
    newmask = np.ma.getmaskarray(a)
    newmask = ndimage.morphology.binary_erosion(newmask, iterations=iterations)
    newmask = ndimage.morphology.binary_dilation(newmask, iterations=iterations)
    return newmask

#This is an alternative to the ma.notmasked_edges
#Note: probably faster/simpler to contour the mask
def contour_edges(a):
    import matplotlib.pyplot as plt
    a = checkma(a)
    #Contour nodata value
    levels = [a.fill_value]
    #kw = {'antialiased':True, 'colors':'r', 'linestyles':'-'}
    kw = {'antialiased':True}
    #Generate contours around nodata
    cs = plt.contour(a.filled(), levels, **kw)
    #This returns a list of numpy arrays
    #allpts = np.vstack(cs.allsegs[0])
    #Extract paths
    p = cs.collections[0].get_paths()
    #Sort by number of vertices in each path
    p_len = [i.vertices.shape[0] for i in p]
    p_sort = [x for (y,x) in sorted(zip(p_len,p), reverse=True)]    
    #cp = p[0].make_compound_path(*p)
    return p_sort

#Brute force search for edges of valid data
def edgefind_loop(a):
    #Row
    i=0
    edges = np.zeros(4)
    print("Top")
    while i < a.shape[0]:
        if a[i].count() > 0:
            edges[0] = i
            break
        i += 1
    i = a.shape[0] - 1
    print("Bottom")
    while i > edges[0]:
        if a[i].count() > 0:
            edges[1] = i
            break
        i -= 1
    #Col
    j=0
    print("Left")
    while j < a.shape[1]:
        if a[:,j].count() > 0:
            edges[2] = j
            break
        j += 1
    j = a.shape[1] - 1
    print("Right")
    while j > edges[2]:
        if a[:,j].count() > 0:
            edges[3] = j
            break
        j -= 1
    print(edges)
    #Default
    #minrow, maxrow, mincol, maxcol
    #minx,miny,maxx,maxy
    #edges=[edges[2],edges[1],edges[3],edges[0]
    return edges

#This is fast, best option for ndv trim
def edgefind2(a):
    edges = np.zeros(4)
    mask = ~np.ma.getmaskarray(a)
    rowmask = mask.any(axis=1)
    colmask = mask.any(axis=0)
    mask = None
    rowmask_nonzero = rowmask.nonzero()[0]
    rowmask = None
    colmask_nonzero = colmask.nonzero()[0]
    colmask = None
    edges[0:2] = [rowmask_nonzero.min(), rowmask_nonzero.max()]
    edges[2:4] = [colmask_nonzero.min(), colmask_nonzero.max()]
    return edges

def ndv_trim(a):
    #edge_bbox = edgefind_loop(a)
    #This is faster
    edge_bbox = edgefind2(a)
    #minrow, maxrow, mincol, maxcol
    #return a[edge_bbox[1]:edge_bbox[3], edge_bbox[0]:edge_bbox[2]]
    return a[edge_bbox[0]:edge_bbox[1], edge_bbox[2]:edge_bbox[3]]

#Fill masked areas with random noise
#This is needed for any fft-based operations
def randomfill(a):
    a = checkma(a)
    #For data that have already been normalized,
    #This provides a proper normal distribution with mean=0 and std=1
    #a = (a - a.mean()) / a.std()
    #noise = a.mask * (np.random.randn(*a.shape))
    noise = a.mask * np.random.normal(a.mean(), a.std(), a.shape)
    #Add the noise
    b = a.filled(0) + noise
    return b

#Wrapper for functions that can't handle ma (e.g. scipy.ndimage)
#Substitutes np.nan
#This will force filters to ignore nan, but causes adjacent pixels to be set to nan as well
#http://projects.scipy.org/scipy/ticket/1155 
def nanfill(a, f_a, *args, **kwargs):
    a = checkma(a)
    ndv = a.fill_value  
    #Note: The following fails for arrays that are not float (np.nan is float)
    b = f_a(a.filled(np.nan), *args, **kwargs)
    #the fix_invalid fill_value parameter doesn't seem to work
    out = np.ma.fix_invalid(b, copy=False)
    out.set_fill_value(ndv)
    return out

#=======================
#Masked array stats
#=======================

def fast_median(a):
    a = checkma(a)
    #return scoreatpercentile(a.compressed(), 50)
    if a.count() > 0:
        out = np.percentile(a.compressed(), 50)
    else:
        out = np.ma.masked
    return out

#Compute median absolute difference
#This is NMAD
#Note: 1.4826 = 1/c
#def mad(a, c=0.6745):
def mad(a, c=1.4826):
    a = checkma(a)
    #return np.ma.median(np.fabs(a - np.ma.median(a))) / c
    if a.count() > 0:
        out = fast_median(np.fabs(a - fast_median(a))) * c
    else:
        out = np.ma.masked
    return out

def mad_ax0(a, c=1.4826):
    out = np.ma.median(np.ma.fabs(a - np.ma.median(a, axis=0)), axis=0)
    return out

#Percentile values
def calcperc(b, perc=(0.1,99.9)):
    b = checkma(b)
    if b.count() > 0:
        #low = scoreatpercentile(b.compressed(), perc[0])
        #high = scoreatpercentile(b.compressed(), perc[1])
        low = np.percentile(b.compressed(), perc[0])
        high = np.percentile(b.compressed(), perc[1])
    else:
        low = 0
        high = 0

    #low = scipy.stats.mstats.scoreatpercentile(b, perc[0])
    #high = scipy.stats.mstats.scoreatpercentile(b, perc[1])

    #This approach can be used for unmasked array, but values less than 0 are problematic
    #bma_low = b.min()
    #bma_high = b.max()
    #low = scipy.stats.scoreatpercentile(b.data.flatten(), perc[0], (bma_low, bma_high))
    #high = scipy.stats.scoreatpercentile(b.data.flatten(), perc[1], (bma_low, bma_high))
    return low, high

def iqr(b, perc=(25, 75)):
    b = checkma(b)
    low, high = calcperc(b, perc)
    return low, high, high - low

def robust_spread(b, perc=(16,84)):
    p16, p84 = calcperc(b, perc)
    spread = np.abs((p84 - p16)/2)
    return p16, p84, spread

def robust_spread_idx(b, sigma=3):
    b = checkma(b)
    med = fast_median(b)
    p16, p84, spread = robust_spread(b)
    min = med - sigma*spread
    max = med + sigma*spread
    b_idx = (b > min) & (b < max)
    return b_idx

def robust_spread_fltr(b, sigma=3):
    #Should compare runtime w/ the ma solution
    b_idx = robust_spread_idx(b, sigma)
    newmask = np.ones_like(b, dtype=bool)
    newmask[b_idx] = False 
    return np.ma.array(b, mask=newmask)

#Should create stats object with these parameters
#Dictionary could work

#Note: need to specify dtype for mean/std calculations 
#full and fast give very different results
#Want to add 25/75 IQR - scipy.stats.mstats.scoreatpercentile
#Convert output to dictionary
#Need to convert stats to float before json.dumps
#The a.mean(dtype='float64') is needed for accuracte calculation
#names = ['count', 'min', 'max', 'mean', 'std', 'med', 'mad', 'q1', 'q2', 'iqr', 'mode', 'p16', 'p84', 'spread']
def print_stats(a, full=False):
    from scipy.stats.mstats import mode 
    a = checkma(a)
    thresh = 4E6
    if full or a.count() < thresh:
        q = (iqr(a))
        p16, p84, spread = robust_spread(a)
        #There has to be a better way to compute the mode for a ma
        #mstats.mode returns tuple of (array[mode], array[count])
        a_mode = float(mode(a, axis=None)[0])
        stats = (a.count(), a.min(), a.max(), a.mean(dtype='float64'), a.std(dtype='float64'), fast_median(a), mad(a), q[0], q[1], q[2], a_mode, p16, p84, spread) 
    else:
        ac = a.compressed()
        stride = int(np.around(ac.size / thresh))
        ac = np.ma.array(ac[::stride])
        #idx = np.random.permutation(ac.size)
        #Note: need the ma cast here b/c of a.count() below
        #ac = np.ma.array(ac[idx[::stride]])
        q = (iqr(ac))
        p16, p84, spread = robust_spread(ac)
        ac_mode = float(mode(ac, axis=None)[0])
        stats = (a.count(), a.min(), a.max(), a.mean(dtype='float64'), a.std(dtype='float64'), fast_median(ac), mad(ac), q[0], q[1], q[2], ac_mode, p16, p84, spread) 
    print("count: %i min: %0.2f max: %0.2f mean: %0.2f std: %0.2f med: %0.2f mad: %0.2f q1: %0.2f q2: %0.2f iqr: %0.2f mode: %0.2f p16: %0.2f p84: %0.2f spread: %0.2f" % stats)
    return stats

def rmse(a):
    ac = checkma(a).compressed()
    rmse = np.sqrt(np.sum(ac**2)/ac.size)
    return rmse

#Check that input is a masked array
def checkma(a, fix=True):
    #isinstance(a, np.ma.MaskedArray)
    if np.ma.is_masked(a):
        out=a
    else:
        out=np.ma.array(a)
    #Fix invalid values
    #Note: this is not necessarily desirable for writing
    if fix:
        #Note: this fails for datetime arrays! Treated as objects.
        #Note: datetime ma returns '?' for fill value
        from datetime import datetime
        if isinstance(a[0], datetime):
            print("Input array appears to be datetime.  Skipping fix")
        else:
            out=np.ma.fix_invalid(out, copy=False)
    return out

#Image view of ma
#Should probably move this to imview.py
def iv(b, **kwargs):
    import matplotlib.pyplot as plt
    import imview 
    b = checkma(b)
    #if hasattr(kwargs,'imshow_kwargs'):
    #    kwargs['imshow_kwargs']['interpolation'] = 'bicubic'
    #else:
    #    kwargs['imshow_kwargs'] = {'interpolation': 'bicubic'}
    #bma_fig(fig, bma, cmap='gist_rainbow_r', clim=None, bg=None, n_subplt=1, subplt=1, label=None, **imshow_kwargs)
    fig = plt.figure()
    imview.bma_fig(fig, b, **kwargs)
    plt.show()
    return fig

#=======================
#Gridding 
#=======================

def ma_interp(a, method='cubic', newmask=None):
    import scipy.interpolate 
    x, y, z = get_xyz(a)
    xi, yi = np.indices(a.shape[::-1])
    zi = scipy.interpolate.griddata((x,y), z, (xi,yi), method=method).T
    #f = scipy.interpolate.SmoothBivariateSpline(x, y, z)
    #zi = f(xi, yi)
    zi = np.ma.fix_invalid(zi, fill_value=a.fill_value)
    if newmask is not None:
        zi = np.ma.array(zi, mask=newmask)
    return zi

def get_xyz(a):
    a = checkma(a)
    y, x = np.indices(a.shape)
    y = np.ma.array(y, mask=np.ma.getmaskarray(a))
    x = np.ma.array(x, mask=np.ma.getmaskarray(a))
    return np.array([x.compressed(), y.compressed(), a.compressed()])

#Efficient solution for 1D
#Note, doesn't include edges
#http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

#This provides non-overlapping blocks of given size
#Could probably be modified for a stride of (1,1) to overlap
#From http://stackoverflow.com/questions/5073767/how-can-i-efficiently-process-a-numpy-array-in-blocks-similar-to-matlabs-blkpro
def block_view(A, block=(3, 3)):
    from numpy.lib.stride_tricks import as_strided as ast
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape= (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)

def sliding_window_padded(a, ws, ss=(1,1), flatten=True):
    colpad = ws[0]/2
    col_a = np.empty((a.shape[0],colpad))
    col_a[:] = np.nan
    a = np.column_stack([col_a, a, col_a])
    rowpad = ws[1]/2
    row_a = np.empty((rowpad, a.shape[1]))
    row_a[:] = np.nan
    a = np.row_stack([row_a, a, row_a])
    return sliding_window(a, ws, ss, flatten) 

#From http://www.johnvinyard.com/blog/?p=268 
def sliding_window(a, ws, ss=None, flatten=True):
    from numpy.lib.stride_tricks import as_strided as ast
    '''
    Return a sliding window over a in any number of dimensions
     
    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.
     
    Returns
        an array containing each n-dimensional window from a
    '''
     
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
     
    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)
     
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))
     
    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
        a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
     
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided
     
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = [i for i in dim if i != 1]
    return strided.reshape(dim)

def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.
     
    Parameters
        shape - an int, or a tuple of ints
     
    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass
 
    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass
     
    raise TypeError('shape must be an int, or a tuple of ints')
