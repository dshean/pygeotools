#! /usr/bin/env python
"""
Functions for IO, mostly wrapped around GDAL

Note: This was all written before RasterIO existed, which might be a better choice. 
"""

import os

import numpy as np
from osgeo import gdal, gdal_array, osr

#Define drivers
mem_drv = gdal.GetDriverByName('MEM')
gtif_drv = gdal.GetDriverByName('GTiff')
vrt_drv = gdal.GetDriverByName("VRT")

#Default GDAL creation options
gdal_opt = ['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
#gdal_opt += ['BLOCKXSIZE=1024', 'BLOCKYSIZE=1024']
#List that can be used for building commands
gdal_opt_co = []
[gdal_opt_co.extend(('-co', i)) for i in gdal_opt]

#Add methods to load ma from OpenCV, PIL, etc.
#These formats should be directly readable as np arrays

#Note: want to modify to import all bands as separate arrays in ndarray
#Unless the user requests a single band, or range of bands

#Check for file existence
def fn_check(fn):
    """Wrapper to check for file existence
    
    Parameters
    ----------
    fn : str
        Input filename string.
    
    Returns
    -------
    bool
        True if file exists, False otherwise.
    """
    return os.path.exists(fn)

def fn_check_full(fn):
    """Check for file existence

    Avoids race condition, but slower than os.path.exists.
    
    Parameters
    ----------
    fn : str
        Input filename string.
    
    Returns
    -------
    status 
        True if file exists, False otherwise.
    """
    status = True 
    if not os.path.isfile(fn): 
        status = False
    else:
        try: 
            open(fn) 
        except IOError:
            status = False
    return status

def fn_list_check(fn_list):
    status = True
    for fn in fn_list:
        if not fn_check(fn):
            print('Unable to find: %s' % fn)
            status = False
    return status

#Wrapper around gdal.Open
def fn_getds(fn):
    """Wrapper around gdal.Open()
    """
    ds = None
    if fn_check(fn):
        ds = gdal.Open(fn, gdal.GA_ReadOnly)
    else:
        print("Unable to find %s" % fn)
    return ds

def fn_getma(fn, bnum=1):
    """Get masked array from input filename

    Parameters
    ----------
    fn : str
        Input filename string
    bnum : int, optional
        Band number
    
    Returns
    -------
    np.ma.array    
        Masked array containing raster values
    """
    #Add check for filename existence
    ds = fn_getds(fn)
    return ds_getma(ds, bnum=bnum)

#Given input dataset, return a masked array for the input band
def ds_getma(ds, bnum=1):
    """Get masked array from input GDAL Dataset

    Parameters
    ----------
    ds : gdal.Dataset 
        Input GDAL Datset
    bnum : int, optional
        Band number
    
    Returns
    -------
    np.ma.array    
        Masked array containing raster values
    """
    b = ds.GetRasterBand(bnum)
    return b_getma(b)

#Given input band, return a masked array
def b_getma(b):
    """Get masked array from input GDAL Band

    Parameters
    ----------
    b : gdal.Band 
        Input GDAL Band 
    
    Returns
    -------
    np.ma.array    
        Masked array containing raster values
    """
    b_ndv = get_ndv_b(b)
    #bma = np.ma.masked_equal(b.ReadAsArray(), b_ndv)
    #This is more appropriate for float, handles precision issues
    bma = np.ma.masked_values(b.ReadAsArray(), b_ndv)
    return bma

def get_sub_dim(src_ds, scale=None, maxdim=1024):
    """Compute dimensions of subsampled dataset 

    Parameters
    ----------
    ds : gdal.Dataset 
        Input GDAL Datset
    scale : int, optional
        Scaling factor
    maxdim : int, optional 
        Maximum dimension along either axis, in pixels
    
    Returns
    -------
    ns
        Numper of samples in subsampled output
    nl
        Numper of lines in subsampled output
    """
    ns = src_ds.RasterXSize
    nl = src_ds.RasterYSize
    maxdim = float(maxdim)
    if scale is None:
        scale_ns = ns/maxdim
        scale_nl = nl/maxdim
        scale = max(scale_ns, scale_nl)
    #Need to check to make sure scale is positive real 
    if scale > 1:
        ns = int(round(ns/scale))
        nl = int(round(nl/scale))
    return ns, nl

#Load a subsampled array
#Can specify scale factor or max dimension
#No need to load the entire dataset for stats computation
def gdal_getma_sub(src_ds, bnum=1, scale=None, maxdim=1024.):    
    """Load a subsampled array, rather than full resolution

    This is useful when working with large rasters

    Uses buf_xsize and buf_ysize options from GDAL ReadAsArray method.

    Parameters
    ----------
    ds : gdal.Dataset 
        Input GDAL Datset
    bnum : int, optional
        Band number
    scale : int, optional
        Scaling factor
    maxdim : int, optional 
        Maximum dimension along either axis, in pixels
    
    Returns
    -------
    np.ma.array    
        Masked array containing raster values
    """
    #print src_ds.GetFileList()[0]
    b = src_ds.GetRasterBand(bnum)
    b_ndv = get_ndv_b(b)
    ns, nl = get_sub_dim(src_ds, scale, maxdim)
    #The buf_size parameters determine the final array dimensions
    b_array = b.ReadAsArray(buf_xsize=ns, buf_ysize=nl)
    bma = np.ma.masked_values(b_array, b_ndv)
    return bma

#Note: need to consolidate with warplib.writeout (takes ds, not ma)
#Add option to build overviews when writing GTiff
#Input proj must be WKT
def writeGTiff(a, dst_fn, src_ds=None, bnum=1, ndv=None, gt=None, proj=None, create=False, sparse=False):
    """Write input array to disk as GeoTiff

    Parameters
    ----------
    a : np.array or np.ma.array
        Input array
    dst_fn : str
        Output filename
    src_ds: GDAL Dataset, optional
        Source Dataset to use for creating copy
    bnum : int, optional 
        Output band
    ndv : float, optional 
        Output NoData Value
    gt : list, optional
        Output GeoTransform
    proj : str, optional
        Output Projection (OGC WKT or PROJ.4 format)
    create : bool, optional
        Create new dataset
    sparse : bool, optional
        Output should be created with sparse options
    """
    #If input is not np.ma, this creates a new ma, which has default filL_value of 1E20
    #Must manually override with ndv
    #Also consumes a lot of memory
    #Should bypass if input is bool
    from pygeotools.lib.malib import checkma 
    a = checkma(a, fix=False)
    #Want to preserve fill_value if already specified
    if ndv is not None:
        a.set_fill_value(ndv)
    driver = gtif_drv
    #Currently only support writing singleband rasters
    #if a.ndim > 2:
    #   np_nbands = a.shape[2]
    #   if src_ds.RasterCount np_nbands: 
    #      for bnum in np_nbands:
    nbands = 1
    np_dt = a.dtype.name
    if src_ds is not None:
        #If this is a fn, get a ds
        #Note: this saves a lot of unnecessary iolib.fn_getds calls
        if isinstance(src_ds, str):
            src_ds = fn_getds(src_ds)
        #if isinstance(src_ds, gdal.Dataset):
        src_dt = gdal.GetDataTypeName(src_ds.GetRasterBand(bnum).DataType)
        src_gt = src_ds.GetGeoTransform()
        #This is WKT
        src_proj = src_ds.GetProjection()
        #src_srs = osr.SpatialReference()  
        #src_srs.ImportFromWkt(src_ds.GetProjectionRef())

    #Probably a cleaner way to handle this
    if gt is None:
        gt = src_gt
    if proj is None:
        proj = src_proj

    #Need to create a new copy of the default options
    opt = list(gdal_opt)
    
    #Note: packbits is better for sparse data
    if sparse:
        opt.remove('COMPRESS=LZW')
        opt.append('COMPRESS=PACKBITS')
        #Not sure if VW can handle sparse tif
        #opt.append('SPARSE_OK=TRUE')

    #Use predictor=3 for floating point data
    if 'float' in np_dt.lower() and 'COMPRESS=LZW' in opt: 
        opt.append('PREDICTOR=3')

    #If input ma is same as src_ds, write out array using CreateCopy from existing dataset
    #if not create and (src_ds is not None) and ((a.shape[0] == src_ds.RasterYSize) and (a.shape[1] == src_ds.RasterXSize) and (np_dt.lower() == src_dt.lower())): 
    #Should compare srs.IsSame(src_srs)
    if not create and (src_ds is not None) and ((a.shape[0] == src_ds.RasterYSize) and (a.shape[1] == src_ds.RasterXSize) and (np_dt.lower() == src_dt.lower())) and (src_gt == gt) and (src_proj == proj):
        #Note: third option is strict flag, set to false
        dst_ds = driver.CreateCopy(dst_fn, src_ds, 0, options=opt)
    #Otherwise, use Create
    else:
        a_dtype = a.dtype
        gdal_dtype = np2gdal_dtype(a_dtype)
        if a_dtype.name == 'bool':
            #Set ndv to 0
            a.fill_value = False
            opt.remove('COMPRESS=LZW')
            opt.append('COMPRESS=DEFLATE')
            #opt.append('NBITS=1')
        #Create(fn, nx, ny, nbands, dtype, opt)
        dst_ds = driver.Create(dst_fn, a.shape[1], a.shape[0], nbands, gdal_dtype, options=opt)
        #Note: Need GeoMA here to make this work, or accept gt as argument
        #Could also do ds creation in calling script
        if gt is not None:
            dst_ds.SetGeoTransform(gt)
        if proj is not None:
            dst_ds.SetProjection(proj)
    
    dst_ds.GetRasterBand(bnum).WriteArray(a.filled())
    dst_ds.GetRasterBand(bnum).SetNoDataValue(float(a.fill_value))
    dst_ds = None

#Move to geolib?
#Look up equivalent GDAL data type
def np2gdal_dtype(d):
    """
    Get GDAL RasterBand datatype that corresponds with NumPy datatype
    Input should be numpy array or numpy dtype
    """
    dt_dict = gdal_array.codes        
    if isinstance(d, (np.ndarray, np.generic)):
        d = d.dtype
    #This creates dtype from another built-in type
    #d = np.dtype(d)
    if isinstance(d, np.dtype):
        if d.name == 'int8':
            gdal_dt = 1
        elif d.name == 'bool':
            #Write out as Byte
            gdal_dt = 1 
        else:
            gdal_dt = list(dt_dict.keys())[list(dt_dict.values()).index(d)]
    else:
        print("Input must be NumPy array or NumPy dtype")
        gdal_dt = None
    return gdal_dt

def gdal2np_dtype(b):
    """
    Get NumPy datatype that corresponds with GDAL RasterBand datatype
    Input can be filename, GDAL Dataset, GDAL RasterBand, or GDAL integer dtype
    """
    dt_dict = gdal_array.codes
    if isinstance(b, str):
        b = gdal.Open(b)
    if isinstance(b, gdal.Dataset):
        b = b.GetRasterBand(1)
    if isinstance(b, gdal.Band):
        b = b.DataType
    if isinstance(b, int):
        np_dtype = dt_dict[b]
    else:
        np_dtype = None
        print("Input must be GDAL Dataset or RasterBand object")
    return np_dtype

#Replace nodata value in GDAL band
def replace_ndv(b, new_ndv):
    b_ndv = get_ndv_b(b)    
    bma = np.ma.masked_values(b.ReadAsArray(), b_ndv)
    bma.set_fill_value(new_ndv)
    b.WriteArray(bma.filled())
    b.SetNoDataValue(new_ndv)
    return b

def set_ndv(dst_fn, ndv):
    dst_ds = gdal.Open(dst_fn, gdal.GA_Update)
    for n in range(1, dst_ds.RasterCount+1):
        b = dst_ds.GetRasterBand(1)
        b.SetNoDataValue(ndv)
    dst_ds = None

#Should overload these functions to handle fn, ds, or b
#Perhaps abstract, as many functions will need this functionality
def get_ndv_fn(fn):
    ds = gdal.Open(fn, gdal.GA_ReadOnly)
    return get_ndv_ds(ds)

#Want to modify to handle multi-band images and return list of ndv
def get_ndv_ds(ds, bnum=1):
    b = ds.GetRasterBand(bnum)
    return get_ndv_b(b)

#Return nodata value for GDAL band
def get_ndv_b(b):
    """Get NoData value for GDAL band.

    If NoDataValue is not set in the band, 
    extract upper left and lower right pixel values.
    Otherwise assume NoDataValue is 0.
 
    Parameters
    ----------
    b : GDALRasterBand object 
        This is the input band.
 
    Returns
    -------
    b_ndv : float 
        NoData value 
    """

    b_ndv = b.GetNoDataValue()
    if b_ndv is None:
        #Check ul pixel for ndv
        ns = b.XSize
        nl = b.YSize
        ul = float(b.ReadAsArray(0, 0, 1, 1))
        #ur = float(b.ReadAsArray(ns-1, 0, 1, 1))
        lr = float(b.ReadAsArray(ns-1, nl-1, 1, 1))
        #ll = float(b.ReadAsArray(0, nl-1, 1, 1))
        #Probably better to use 3/4 corner criterion
        #if ul == ur == lr == ll:
        if np.isnan(ul) or ul == lr:
            b_ndv = ul
        else:
            #Assume ndv is 0
            b_ndv = 0
    elif np.isnan(b_ndv):
        b_dt = gdal.GetDataTypeName(b.DataType)
        if 'Float' in b_dt:
            b_ndv = np.nan
        else:
            b_ndv = 0
    return b_ndv

#Write out a recarray as a csv
def write_recarray(outfn, ra):
    with open(outfn,'w') as f:
        f.write(','.join([str(item) for item in ra.dtype.names])+'\n')
        for row in ra:
            f.write(','.join([str(item) for item in row])+'\n')
 
#Check to make sure image doesn't contain errors
def image_check(fn):
    ds = gdal.Open(fn)
    status = True 
    for i in range(ds.RasterCount):
        ds.GetRasterBand(i+1).Checksum()
        if gdal.GetLastErrorType() != 0:
            status = False 
    return status

def cpu_count():
    """Return system CPU count
    """
    from multiprocessing import cpu_count
    return cpu_count()

#This is a shared directory for files like LULC, used by multiple tools 
#Default location is $HOME/data
#Can specify in ~/.bashrc or ~/.profile
#export DATADIR=$HOME/data
def get_datadir():
    default_datadir = os.path.join(os.path.expanduser('~'), 'data')
    datadir = os.environ.get('DATADIR', default_datadir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    return datadir

#Function to get files using urllib
#This works with ftp
def getfile(url, outdir=None):
    """Function to fetch files using urllib

    Works with ftp

    """
    fn = os.path.split(url)[-1]
    if outdir is not None:
        fn = os.path.join(outdir, fn)
    if not os.path.exists(fn):
        #Find appropriate urlretrieve for Python 2 and 3
        try:
            from urllib.request import urlretrieve
        except ImportError:
            from urllib import urlretrieve 
        print("Retrieving: %s" % url)
        #Add progress bar
        urlretrieve(url, fn)
    return fn

#Function to get files using requests
#Works with https authentication
def getfile2(url, auth=None, outdir=None):
    """Function to fetch files using requests

    Works with https authentication

    """
    import requests
    print("Retrieving: %s" % url)
    fn = os.path.split(url)[-1]
    if outdir is not None:
        fn = os.path.join(outdir, fn)
    if auth is not None:
        r = requests.get(url, stream=True, auth=auth)
    else:
        r = requests.get(url, stream=True)
    chunk_size = 1000000
    with open(fn, 'wb') as fd:
        for chunk in r.iter_content(chunk_size):
            fd.write(chunk)

#Get necessary credentials to access MODSCAG products - hopefully this will soon be archived with NSIDC 
def get_auth():
    """Get authorization token for https
    """
    import getpass
    from requests.auth import HTTPDigestAuth
    #This binds raw_input to input for Python 2
    try:
       input = raw_input
    except NameError:
       pass
    uname = input("MODSCAG Username:")
    pw = getpass.getpass("MODSCAG Password:")
    auth = HTTPDigestAuth(uname, pw)
    #wget -A'h8v4*snow_fraction.tif' --user=uname --password=pw
    return auth
