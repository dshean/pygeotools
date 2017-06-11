#! /usr/bin/env python
"""
Library of functions required for raster warping in memory or on disk

The guts of warptool.py

"""

"""
todo:
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
    """Warp an input dataset with predetermined arguments specifying output res/extent/srs

    This is the function that actually calls gdal.ReprojectImage
    
    Parameters
    ----------
    src_ds : gdal.Dataset object
        Dataset to be warped
    res : float
        Desired output resolution
    extent : list of float
        Desired output extent in t_srs coordinate system
    t_srs : osr.SpatialReference()
        Desired output spatial reference
    r : str
        Desired resampling algorithm
    driver : GDAL Driver to use for warp 
        Either MEM or GTiff
    dst_fn : str
        Output filename (for disk warp)
    dst_ndv : float
        Desired output NoData Value

    Returns
    -------
    dst_ds : gdal.Dataset object
        Warped dataset (either in memory or on disk)

    """
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

    gra = parse_rs_alg(r)

    #At this point, the resolution and extent values must be float
    #Extent must be list
    res = float(res)
    extent = [float(i) for i in extent]

    #Might want to move this to memwarp_multi, keep memwarp basic w/ gdal.GRA types

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
#NOTE: there is a bug in GDAL that mem datasets are initialized with ndv of 0, even if ds has different ndv set
#http://gis.stackexchange.com/questions/158503/9999-no-data-value-becomes-0-when-writing-array-to-gdal-memory-file
#Hardcode 0 here
def memwarp(src_ds, res=None, extent=None, t_srs=None, r=None, oudir=None, dst_ndv=0):
    """Helper function that calls warp for single input Dataset with output to memory (GDAL Memory Driver)
    """
    driver = iolib.mem_drv
    return warp(src_ds, res, extent, t_srs, r, driver=driver, dst_ndv=dst_ndv)

#Use this to warp directly to output file - no need to write to memory then CreateCopy 
def diskwarp(src_ds, res=None, extent=None, t_srs=None, r='cubic', outdir=None, dst_fn=None, dst_ndv=None):
    """Helper function that calls warp for single input Dataset with output to disk (GDAL GeoTiff Driver)
    """
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

def parse_rs_alg(r):
    """Parse resampling algorithm
    """
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
        gra = None
        sys.exit("Invalid resampling method")
    return gra

def parse_srs(t_srs, src_ds_list=None):
    """Parse arbitrary input t_srs

    Parameters
    ----------
    t_srs : str or gdal.Dataset or filename
        Arbitrary input t_srs 
    src_ds_list : list of gdal.Dataset objects, optional
        Needed if specifying 'first' or 'last'

    Returns
    -------
    t_srs : osr.SpatialReference() object
        Output spatial reference system
    """
    if t_srs is None and src_ds_list is None:
        print("Input t_srs and src_ds_list are both None")
    else:
        if t_srs is None:
            t_srs = 'first'
        if t_srs == 'first' and src_ds_list is not None:
            t_srs = geolib.get_ds_srs(src_ds_list[0])
        elif t_srs == 'last' and src_ds_list is not None:
            t_srs = geolib.get_ds_srs(src_ds_list[-1])
        #elif t_srs == 'source':
        #    t_srs = None 
        elif isinstance(t_srs, osr.SpatialReference): 
            pass
        elif isinstance(t_srs, gdal.Dataset):
            t_srs = geolib.get_ds_srs(t_srs)
        elif isinstance(t_srs, str) and os.path.exists(t_srs): 
            t_srs = geolib.get_ds_srs(gdal.Open(t_srs))
        elif isinstance(t_srs, str):
            temp = osr.SpatialReference()
            if 'EPSG' in t_srs.upper():
                epsgcode = int(t_srs.split(':')[-1])
                temp.ImportFromEPSG(epsgcode)
            elif 'proj' in t_srs:
                temp.ImportFromProj4(t_srs)
            else:
                #Assume the user knows what they are doing
                temp.ImportFromWkt(t_srs)
            t_srs = temp
        else:
            t_srs = None
    return t_srs

def parse_res(res, src_ds_list=None, t_srs=None):
    """Parse arbitrary input res 

    Parameters
    ----------
    res : str or gdal.Dataset or filename or float
        Arbitrary input res 
    src_ds_list : list of gdal.Dataset objects, optional
        Needed if specifying 'first' or 'last'
    t_srs : osr.SpatialReference() object 
        Projection for res calculations, optional

    Returns
    -------
    res : float 
        Output resolution
        None if source resolution should be preserved
    """
    #Default to using first t_srs for res calculations
    #Assumes src_ds_list is not None
    t_srs = parse_srs(t_srs, src_ds_list)

    #Valid strings
    res_str_list = ['first', 'last', 'min', 'max', 'mean', 'med']

    #Compute output resolution in t_srs
    if res in res_str_list and src_ds_list is not None:
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
    return res

def parse_extent(extent, src_ds_list, t_srs=None):
    """Parse arbitrary input extent

    Parameters
    ----------
    extent : str or gdal.Dataset or filename or list of float
        Arbitrary input extent
    src_ds_list : list of gdal.Dataset objects, optional
        Needed if specifying 'first', 'last', 'intersection', or 'union'
    t_srs : osr.SpatialReference() object, optional 
        Projection for res calculations

    Returns
    -------
    extent : list of float 
        Output extent [xmin, ymin, xmax, ymax] 
        None if source extent should be preserved
    """
    #Default to using first t_srs for extent calculations
    #Assumes src_ds_list is not None
    t_srs = parse_srs(t_srs, src_ds_list)

    #Valid strings
    extent_str_list = ['first', 'last', 'intersection', 'union']

    if extent in extent_str_list and src_ds_list is not None:
        if len(src_ds_list) == 1 and (extent == 'intersection' or extent == 'union'):
            extent = None
        elif extent == 'first':
            extent = geolib.ds_geom_extent(src_ds_list[0], t_srs=t_srs)
            #extent = geolib.ds_extent(src_ds_list[0], t_srs=t_srs)
        elif extent == 'last':
            extent = geolib.ds_geom_extent(src_ds_list[-1], t_srs=t_srs)
            #extent = geolib.ds_extent(src_ds_list[-1], t_srs=t_srs)
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
    return extent

def warp_multi(src_ds_list, res='first', extent='intersection', t_srs='first', r='cubic', warptype=memwarp, outdir=None, dst_ndv=None, verbose=False):
    """This parses and checks inputs, then calls desired warp function with appropriate arguments for each input ds
    
    Parameters
    ----------
    src_ds_list : list of gdal.Dataset objects
        List of original datasets to be warped
    res : arbitrary type
        Desired output resolution
    extent : arbitrary type
        Desired output extent
    t_srs : arbitrary type
        Desired output spatial reference
    r : str
        Desired resampling algorithm
    warptype : function
        Desired warp type (write to memory or disk)
    outdir : str
        Desired output directory (for disk warp)
    dst_ndv : float
        Desired output NoData Value
    verbose : bool 
        Print extra information for debugging purposes

    Returns
    -------
    out_ds_list : list of gdal.Dataset objects
        List of warped datasets (either in memory or on disk)
    """
    #Type cast arguments as str for evaluation
    #Avoid path errors
    #res = str(res)
    #extent = str(extent)
    #t_srs = str(t_srs)

    #Parse the input
    t_srs = parse_srs(t_srs, src_ds_list)
    res = parse_res(res, src_ds_list, t_srs)
    extent = parse_extent(extent, src_ds_list, t_srs)

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

        #If input srs are different, must warp
        ds_t_srs = geolib.get_ds_srs(ds)
        srscheck = bool(t_srs.IsSame(ds_t_srs))
       
        if verbose:
            print('\n%s' % ds_t_srs.ExportToWkt())
            print('%s\n' % t_srs.ExportToWkt())
            print('srscheck: %s\n' % srscheck)

        rescheck = False
        extentcheck = False

        #if srscheck:
        #Extract info from ds to see if warp is necessary
        ds_res = geolib.get_res(ds, square=True)[0]
        ds_extent = geolib.ds_extent(ds)

        #Note: these checks necessary to handle rounding and precision issues
        #Round extent and res to nearest mm
        precision = 1E-3
        #Or if t_srs has units of degrees
        if ds_t_srs.IsGeographic():
            precision = 1E-8

        rescheck = (res is None) or geolib.res_compare(res, ds_res, precision=precision)
        extentcheck = (extent is None) or geolib.extent_compare(extent, ds_extent, precision=precision)

        if verbose:
            print('\n%s, %s\n' % (ds_res, res)) 
            print('%s' % ds_extent)
            print('%s\n' % extent) 
            print('rescheck: %s' % rescheck)
            print('extentcheck: %s\n' % extentcheck)

        #If the ds passes all three, it is identical to desired output, short circuit
        if rescheck and extentcheck and srscheck:
            out_ds_list.append(ds)
        else:
            dst_ds = warptype(ds, res, extent, t_srs, r, outdir, dst_ndv=dst_ndv)
            out_ds_list.append(dst_ds)

    return out_ds_list

def memwarp_multi(src_ds_list, res='first', extent='intersection', t_srs='first', r='cubic', verbose=False, dst_ndv=0):
    """Helper function for memwarp of multiple input GDAL Datasets
    """
    return warp_multi(src_ds_list, res, extent, t_srs, r, verbose=verbose, warptype=memwarp, dst_ndv=dst_ndv)

def memwarp_multi_fn(src_fn_list, res='first', extent='intersection', t_srs='first', r='cubic', verbose=False, dst_ndv=0):
    """Helper function for memwarp of multiple input filenames
    """
    #Should implement proper error handling here
    if not iolib.fn_list_check(src_fn_list):
        sys.exit('Missing input file(s)')
    src_ds_list = [gdal.Open(fn, gdal.GA_ReadOnly) for fn in src_fn_list]
    return memwarp_multi(src_ds_list, res, extent, t_srs, r, verbose=verbose, dst_ndv=dst_ndv)

def diskwarp_multi(src_ds_list, res='first', extent='intersection', t_srs='first', r='cubic', verbose=False, outdir=None, dst_ndv=None):
    """Helper function for diskwarp of multiple input GDAL Datasets
    """
    return warp_multi(src_ds_list, res, extent, t_srs, r, verbose=verbose, warptype=diskwarp, outdir=outdir, dst_ndv=dst_ndv)

def diskwarp_multi_fn(src_fn_list, res='first', extent='intersection', t_srs='first', r='cubic', verbose=False, outdir=None, dst_ndv=None):
    """Helper function for diskwarp of multiple input filenames
    """
    #Should implement proper error handling here
    if not iolib.fn_list_check(src_fn_list):
        sys.exit('Missing input file(s)')
    src_ds_list = [gdal.Open(fn, gdal.GA_ReadOnly) for fn in src_fn_list]
    return diskwarp_multi(src_ds_list, res, extent, t_srs, r, verbose=verbose, outdir=outdir, dst_ndv=dst_ndv)

def writeout(ds, outfn):
    """Write ds to disk

    Note: Depreciated function - use diskwarp functions when writing to disk to avoid unnecessary CreateCopy

    """
    print("Writing out %s" % outfn) 
    #Use outfn extension to get driver
    #This may have issues if outfn already exists and the mem ds has different dimensions/res
    out_ds = iolib.gtif_drv.CreateCopy(outfn, ds, 0, options=iolib.gdal_opt)
    out_ds = None
