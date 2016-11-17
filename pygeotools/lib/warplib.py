#! /usr/bin/env python

"""
David Shean
dshean@gmail.com

To do:
Much better type checking
Implement multiprocessing!
Proper vdatum tags in geotiff header
filename preservation
Instead of doing the 'first' stuff, check actual values before writing out original dataset
"""

import sys
import os 
import math

from osgeo import gdal, osr

from pygeotools.lib import geolib
from pygeotools.lib import iolib

mem_drv = iolib.mem_drv
gtif_drv = iolib.gtif_drv

#Note: can run into filesystem limits for number of open files
#http://superuser.com/questions/433746/is-there-a-fix-for-the-too-many-open-files-in-system-error-on-os-x-10-7-1
gdal.SetConfigOption('GDAL_MAX_DATASET_POOL_SIZE', '2048')
#Need to set in the shell
#ulimit -S -n 2048
import resource
resource.setrlimit(resource.RLIMIT_CORE,(resource.RLIM_INFINITY, resource.RLIM_INFINITY))

def warp(src_ds, res=None, extent=None, t_srs=None, r='cubic', driver=mem_drv, dst_fn=None, dst_ndv=None):
    src_srs = geolib.get_ds_srs(src_ds)
    
    if t_srs is None:
        t_srs = geolib.get_ds_srs(src_ds)
    
    src_gt = src_ds.GetGeoTransform()
    #Note: get_res returns [x_res, y_res]
    #Could just use gt here and average x_res and y_res
    src_res = geolib.get_res(src_ds, t_srs=t_srs, square=True)[0]

    if res is None:
        res = src_res

    if extent is None:
        extent = geolib.ds_geom_extent(src_ds, t_srs=t_srs)
    
    #Note: GDAL Lanczos creates block artifacts
    #Wait for gdalwarp to support gaussian resampling
    #Want to use Lanczos for downsampling
    #if src_res < res:
    #    gra = gdal.GRA_Lanczos
    #See http://blog.codinghorror.com/better-image-resizing/
    # Suggests cubic for downsampling, bilinear for upsampling
    #    gra = gdal.GRA_Cubic
    #Cubic for upsampling
    #elif src_res >= res:
    #    gra = gdal.GRA_Bilinear

    #At this point, the resolution and extent values must be float
    #Extent must be list
    res = float(res)
    extent = [float(i) for i in extent]

    #Might want to move this to memwarp_multi, keep memwarp basic w/ gdal.GRA types

    #Note:GRA_CubicSpline created huge block artifacts for the St. Helen's compute_dh WV cases
    #Stick with CubicSpline for both upsampling/downsampling for now
    if r == 'near':
        #Note: Nearest respects nodata when downsampling
        gra = gdal.GRA_NearestNeighbour
    elif r == 'bilinear':
        gra = gdal.GRA_Bilinear
    elif r == 'cubic':
        gra = gdal.GRA_Cubic
    elif r == 'cubicspline':
        gra = gdal.GRA_CubicSpline
    elif r == 'average':
        gra = gdal.GRA_Average
    elif r == 'lanczos':
        gra = gdal.GRA_Lanczos
    elif r == 'mode':
        #Note: Mode respects nodata when downsampling, but very slow
        gra = gdal.GRA_Mode
    else:
        sys.exit("Invalid resampling method")

    #Create progress function
    prog_func = gdal.TermProgress
    
    if dst_fn is None:
        #This is a dummy fn if only in mem, but can be accessed later via GetFileList()
        #Actually, no, doesn't look like the filename survivies
        dst_fn = ''
    
    #Compute output image dimensions
    dst_nl = int(round((extent[3] - extent[1])/res))
    dst_ns = int(round((extent[2] - extent[0])/res))
    #dst_nl = int(math.ceil((extent[3] - extent[1])/res))
    #dst_ns = int(math.ceil((extent[2] - extent[0])/res))
    #dst_nl = int(math.floor((extent[3] - extent[1])/res))
    #dst_ns = int(math.floor((extent[2] - extent[0])/res))
    print('nl: %i ns: %i res: %0.3f' % (dst_nl, dst_ns, res))
    #Create output dataset
    src_b = src_ds.GetRasterBand(1)
    src_dt = src_b.DataType
    src_nl = src_ds.RasterYSize
    src_ns = src_ds.RasterXSize

    dst_ds = driver.Create(dst_fn, dst_ns, dst_nl, src_ds.RasterCount, src_dt) 

    dst_ds.SetProjection(t_srs.ExportToWkt())
    #Might be an issue to use src_gt rotation terms here with arbitrary extent/res
    dst_gt = [extent[0], res, src_gt[2], extent[3], src_gt[4], -res]
    dst_ds.SetGeoTransform(dst_gt)
   
    #This will smooth the input before downsampling to prevent aliasing, fill gaps
    #Pretty inefficent, as we need to create another intermediate dataset
    gauss = False 

    for n in range(1, src_ds.RasterCount+1):
        if dst_ndv is None:
            src_b = src_ds.GetRasterBand(n)
            src_ndv = iolib.get_ndv_b(src_b)
            dst_ndv = src_ndv
        b = dst_ds.GetRasterBand(n)
        b.SetNoDataValue(dst_ndv)
        b.Fill(dst_ndv)

        if gauss:
            from pygeotools.lib import filtlib
            #src_a = src_b.GetVirtualMemArray()
            #Compute resampling ratio to determine filter window size
            res_ratio = float(res)/src_res
            print("Resampling factor: %0.3f" % res_ratio)
            #Might be more efficient to do iterative gauss filter with size 3, rather than larger windows
            f_size = math.floor(res_ratio/2.)*2+1
            #This is conservative to avoid filling holes with noise
            #f_size = math.floor(res_ratio/2.)*2-1
            if f_size <= 1:
                continue

            print("Smoothing window size: %i" % f_size)
            #Create temp dataset to store filtered array - avoid overwriting original
            temp_ds = driver.Create('', src_ns, src_nl, src_ds.RasterCount, src_dt) 
            temp_ds.SetProjection(src_srs.ExportToWkt())
            temp_ds.SetGeoTransform(src_gt)
            temp_b = temp_ds.GetRasterBand(n)
            temp_b.SetNoDataValue(dst_ndv)
            temp_b.Fill(dst_ndv)

            src_a = iolib.b_getma(src_b)
            src_a = filtlib.gauss_fltr_astropy(src_a, size=f_size)
            #Want to run with maskfill, so only fills gaps, without expanding isolated points
            temp_b.WriteArray(src_a)
            src_ds = temp_ds
            
            #In theory, NN should be fine since we already smoothed.  In practice, cubic still provides slightly better results
            #gra = gdal.GRA_NearestNeighbour
    
    #Note: default maxerror=0.0
    #Shouldn't neet to specify srs?
    #result = gdal.ReprojectImage(src_ds, dst_ds, gra)
    gdal.ReprojectImage(src_ds, dst_ds, src_srs.ExportToWkt(), t_srs.ExportToWkt(), gra, 0.0, 0.0, prog_func)

    #Note: this is now done in diskwarp
    #Write out to disk
    #if driver != mem_drv:
    #    dst_ds.FlushCache()

    #Return GDAL dataset object in memory
    return dst_ds

#Use this to warp to mem ds
def memwarp(src_ds, res=None, extent=None, t_srs=None, r=None, oudir=None, dst_ndv=None):
    driver = iolib.mem_drv
    return warp(src_ds, res, extent, t_srs, r, driver, dst_ndv=dst_ndv)

#Use this to warp directly to output file - no need to write to memory then CreateCopy 
def diskwarp(src_ds, res=None, extent=None, t_srs=None, r='cubic', outdir=None, dst_fn=None, dst_ndv=None):
    if dst_fn is None:
        dst_fn = os.path.splitext(src_ds.GetFileList()[0])[0]+'_warp.tif'
    if outdir is not None:
        dst_fn = os.path.join(outdir, os.path.basename(dst_fn))  
    driver = iolib.gtif_drv
    dst_ds = warp(src_ds, res, extent, t_srs, r, driver, dst_fn, dst_ndv=dst_ndv)
    #Write out
    dst_ds = None
    #Now reopen ds from disk
    dst_ds = gdal.Open(dst_fn)
    return dst_ds

def warp_multi(src_ds_list, res='first', extent='intersection', t_srs='first', r='cubic', warptype=memwarp, outdir=None, dst_ndv=None):
    #Type cast arguments as str for evaluation
    #Avoid path errors
    #res = str(res)
    #extent = str(extent)
    #t_srs = str(t_srs)

    if t_srs == 'first':
        t_srs = geolib.get_ds_srs(src_ds_list[0])
    elif t_srs == 'last':
        t_srs = geolib.get_ds_srs(src_ds_list[-1])
    #elif t_srs == 'source':
    #    t_srs = None 
    elif isinstance(t_srs, osr.SpatialReference): 
        pass
    elif isinstance(t_srs, gdal.Dataset):
        t_srs = geolib.get_ds_srs(t_srs)
    elif isinstance(t_srs, str) and os.path.exists(t_srs): 
        t_srs = geolib.get_ds_srs(gdal.Open(t_srs))
    else:
        temp = osr.SpatialReference()
        temp.ImportFromProj4(t_srs)
        t_srs = temp

    #Compute output resolution in t_srs
    #Returns min, max, mean, med
    res_stats = geolib.get_res_stats(src_ds_list, t_srs=t_srs)
    if res == 'first':
        res = geolib.get_res(src_ds_list[0], t_srs=t_srs, square=True)[0]
    elif res == 'last':
        res = geolib.get_res(src_ds_list[-1], t_srs=t_srs, square=True)[0]
    elif res == 'min':
        res = res_stats[0]
    elif res == 'max':
        res = res_stats[1]
    elif res == 'mean':
        res = res_stats[2]
    elif res == 'med':
        res = res_stats[3]
    elif res == 'source':
        res = None
    elif isinstance(res, gdal.Dataset):
        res = geolib.get_res(res, t_srs=t_srs, square=True)[0]
    elif isinstance(res, str) and os.path.exists(res): 
        res = geolib.get_res(gdal.Open(res), t_srs=t_srs, square=True)[0]
    else:
        res = float(res)

    #Extent returned is xmin, ymin, xmax, ymax
    if len(src_ds_list) == 1 and (extent == 'intersection' or extent == 'union'):
        extent = None
    elif extent == 'first':
        extent = geolib.ds_geom_extent(src_ds_list[0], t_srs=t_srs)
    elif extent == 'last':
        extent = geolib.ds_geom_extent(src_ds_list[-1], t_srs=t_srs)
    elif extent == 'intersection':
        #By default, compute_intersection takes ref_srs from ref_ds
        extent = geolib.ds_geom_intersection_extent(src_ds_list, t_srs=t_srs)
        if len(src_ds_list) > 1 and extent is None:
            #print "Input images do not intersect"
            sys.exit("Input images do not intersect")
    elif extent == 'union':
        #Need to clean up union t_srs handling
        extent = geolib.ds_geom_union_extent(src_ds_list, t_srs=t_srs)
    elif extent == 'source':
        extent = None
    elif isinstance(extent, gdal.Dataset):
        extent = geolib.ds_geom_extent(extent, t_srs=t_srs)
    elif isinstance(extent, str) and os.path.exists(extent): 
        extent = geolib.ds_geom_extent(gdal.Open(extent), t_srs=t_srs)
    elif isinstance(extent, (list, tuple)):
        extent = list(extent)
    else:
        extent = [float(i) for i in extent.split(' ')]

    print("\nWarping all inputs to the following:")
    print("Resolution: %s" % res)
    print("Extent: %s" % str(extent))
    print("Projection: '%s'" % t_srs.ExportToProj4())
    print("Resampling alg: %s\n" % r)  

    out_ds_list = []
    for i, ds in enumerate(src_ds_list):
        fn_list = ds.GetFileList()
        fn = '[memory]'
        if fn_list is not None:
            fn = fn_list[0]
        print("%i of %i: %s" % (i+1, len(src_ds_list), fn))
        #Check each ds to see if warp is necessary
        ds_res = geolib.get_res(ds, square=True)[0]
        #Note: this rounds to %0.5f, extent_compare uses %0.6f
        #ds_extent = geolib.ds_geom_extent(ds)
        ds_extent = geolib.ds_extent(ds)
        ds_t_srs = geolib.get_ds_srs(ds)

        #Debug
        #print ds_res, ds_extent
        #print res, extent

        #Note: these checks necessary to deal with rounding errors
        rescheck=False
        if res is None or geolib.res_compare(res, ds_res):
            rescheck=True
        extentcheck=False
        if extent is None or geolib.extent_compare(extent, ds_extent):
            extentcheck=True
        srscheck=False
        if (t_srs.IsSame(ds_t_srs)):
            srscheck=True
            
        if rescheck and extentcheck and srscheck:
            out_ds_list.append(ds)
        else:
            dst_ds = warptype(ds, res, extent, t_srs, r, outdir, dst_ndv=dst_ndv)
            out_ds_list.append(dst_ds)
    return out_ds_list

def memwarp_multi(src_ds_list, res='first', extent='intersection', t_srs='first', r='cubic'):
    return warp_multi(src_ds_list, res, extent, t_srs, r, warptype=memwarp)

def memwarp_multi_fn(src_fn_list, res='first', extent='intersection', t_srs='first', r='cubic'):
    #Should implement proper error handling here
    if not iolib.fn_list_check(src_fn_list):
        sys.exit('Missing input file(s)')
    src_ds_list = [gdal.Open(fn, gdal.GA_ReadOnly) for fn in src_fn_list]
    return memwarp_multi(src_ds_list, res, extent, t_srs, r)

def diskwarp_multi(src_ds_list, res='first', extent='intersection', t_srs='first', r='cubic', outdir=None, dst_ndv=None):
    return warp_multi(src_ds_list, res, extent, t_srs, r, warptype=diskwarp, outdir=outdir, dst_ndv=dst_ndv)

def diskwarp_multi_fn(src_fn_list, res='first', extent='intersection', t_srs='first', r='cubic', outdir=None, dst_ndv=None):
    #Should implement proper error handling here
    if not iolib.fn_list_check(src_fn_list):
        sys.exit('Missing input file(s)')
    src_ds_list = [gdal.Open(fn, gdal.GA_ReadOnly) for fn in src_fn_list]
    return diskwarp_multi(src_ds_list, res, extent, t_srs, r, outdir=outdir, dst_ndv=dst_ndv)

#Depreciated function - use diskwarp functions when writing to disk, avoids unnecessary CreateCopy
def writeout(ds, outfn):
    print("Writing out %s" % outfn) 
    #Use outfn extension to get driver
    #This may have issues if outfn already exists and the mem ds has different dimensions/res
    out_ds = iolib.gtif_drv.CreateCopy(outfn, ds, 0, options=iolib.gdal_opt)
    out_ds = None
