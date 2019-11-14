#! /usr/bin/env python
"""
Geospatial functions for rasters, vectors.

"""

#Need to make sure all geom have spatial reference included

import sys
import os
import requests

import numpy as np
from osgeo import gdal, ogr, osr

#Enable GDAL exceptions
gdal.UseExceptions()

#Below are many spatial reference system definitions

#Define WGS84 srs
#mpd = 111319.9
wgs_srs = osr.SpatialReference()
wgs_srs.SetWellKnownGeogCS('WGS84')
wgs_proj = '+proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs '

#Define ECEF srs
ecef_srs=osr.SpatialReference()
ecef_srs.ImportFromEPSG(4978)

#Define ITRF2008 srs
itrf_srs=osr.SpatialReference()
itrf_srs.ImportFromEPSG(5332)

#TOPEX ellipsoid
tp_srs = osr.SpatialReference()
#tp_proj = '+proj=latlong +a=6378136.300000 +rf=298.25700000 +no_defs'
tp_proj = '+proj=latlong +a=6378136.300000 +b=6356751.600563 +towgs84=0,0,0,0,0,0,0 +no_defs'
tp_srs.ImportFromProj4(tp_proj)

#Vertical CS setup
#See: http://lists.osgeo.org/pipermail/gdal-dev/2011-August/029856.html
#https://github.com/OSGeo/proj.4/wiki/VerticalDatums

#Note: must have gtx grid files in /usr/local/share/proj
#Should add a check for these
#cd /usr/local/share/proj
#wget http://download.osgeo.org/proj/vdatum/egm96_15/egm96_15.gtx
#wget http://download.osgeo.org/proj/vdatum/egm08_25/egm08_25.gtx
#NAD83 (ellipsoid) to NAVD88 (orthometric)
#wget http://download.osgeo.org/proj/vdatum/usa_geoid/g.tar.gz
#tar -xzvf g.tar.gz
#rm g.tar.gz
#wget http://download.osgeo.org/proj/vdatum/usa_geoid2012.zip
#unzip usa_geoid2012.zip
#rm usa_geoid2012.zip

egm96_srs=osr.SpatialReference()
egm96_srs.ImportFromProj4("+proj=longlat +datum=WGS84 +no_defs +geoidgrids=egm96_15.gtx")

#Define EGM2008 srs
egm08_srs=osr.SpatialReference()
egm08_srs.ImportFromProj4("+proj=longlat +datum=WGS84 +no_defs +geoidgrids=egm08_25.gtx")

#Define NAD83/NAVD88 srs for CONUS
navd88_conus_srs=osr.SpatialReference()
navd88_conus_srs.ImportFromProj4("+proj=longlat +datum=NAD83 +no_defs +geoidgrids=g2012a_conus.gtx")

#Define NAD83/NAVD88 srs for Alaska
navd88_alaska_srs=osr.SpatialReference()
navd88_alaska_srs.ImportFromProj4("+proj=longlat +datum=NAD83 +no_defs +geoidgrids=g2012a_alaska.gtx")

#Define N Polar Stereographic srs
nps_srs=osr.SpatialReference()
#Note: this doesn't stick!
#nps_srs.ImportFromEPSG(3413)
nps_proj = '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs '
nps_srs.ImportFromProj4(nps_proj)

nps_egm08_srs=osr.SpatialReference()
nps_egm08_srs.ImportFromProj4('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +geoidgrids=egm08_25.gtx +no_defs')

#Define N Polar Stereographic srs
sps_srs=osr.SpatialReference()
#Note: this doesn't stick!
#sps_srs.ImportFromEPSG(3031)
sps_proj = '+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs '
sps_srs.ImportFromProj4(sps_proj)

sps_egm08_srs=osr.SpatialReference()
sps_egm08_srs.ImportFromProj4('+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +geoidgrids=egm08_25.gtx +no_defs')

aea_grs80_srs=osr.SpatialReference()
#aea_grs80_proj='+proj=aea +lat_1=55 +lat_2=65 +lat_0=50 +lon_0=-154 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '
aea_grs80_proj='+proj=aea +lat_1=55 +lat_2=65 +lat_0=50 +lon_0=-154 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs '
aea_grs80_srs.ImportFromEPSG(3338)

aea_navd88_srs=osr.SpatialReference()
#aea_navd88_proj='+proj=aea +lat_1=55 +lat_2=65 +lat_0=50 +lon_0=-154 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs +geoidgrids=g2012a_alaska.gtx'
aea_navd88_proj='+proj=aea +lat_1=55 +lat_2=65 +lat_0=50 +lon_0=-154 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs +towgs84=0,0,0,0,0,0,0 +geoidgrids=g2012a_conus.gtx,g2012a_alaska.gtx,g2012a_guam.gtx,g2012a_hawaii.gtx,g2012a_puertorico.gtx,g2012a_samoa.gtx +vunits=m +no_defs'
aea_navd88_srs.ImportFromProj4(aea_navd88_proj)

#HMA projection
hma_aea_srs = osr.SpatialReference()
#hma_aea_proj = '+proj=aea +lat_1=25 +lat_2=47 +lon_0=85 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs '
hma_aea_proj = '+proj=aea +lat_1=25 +lat_2=47 +lat_0=36 +lon_0=85 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs '
hma_aea_srs.ImportFromProj4(hma_aea_proj)

#CONUS projection
#CONUS bounds 36, 49, -105, -124
conus_aea_srs = osr.SpatialReference()
conus_aea_proj = '+proj=aea +lat_1=36 +lat_2=49 +lat_0=43 +lon_0=-115 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs '
conus_aea_srs.ImportFromProj4(conus_aea_proj)

#To do for transformations below:
#Check input order of lon, lat

#Helper for coordinate transformations
#Input for each coordinate can be float, ndarray, or masked ndarray
def cT_helper(x, y, z, in_srs, out_srs):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    if x.shape != y.shape:
        sys.exit("Inconsistent number of x and y points")
    valid_idx = Ellipsis 
    #Handle case where we have x array, y array, but a constant z (e.g., 0.0)
    if z.shape != x.shape:
        #If a constant elevation is provided
        if z.shape[0] == 1:
            orig_z = z
            z = np.zeros_like(x)
            z[:] = orig_z
            if np.ma.is_masked(x):
                z[np.ma.getmaskarray(x)] = np.ma.masked
        else:
            sys.exit("Inconsistent number of z and x/y points")
    #If any of the inputs is masked, only transform points with all three coordinates available
    if np.ma.is_masked(x) or np.ma.is_masked(y) or np.ma.is_masked(z):
        x = np.ma.array(x)
        y = np.ma.array(y)
        z = np.ma.array(z)
        from pygeotools.lib import malib
        valid_idx = ~(malib.common_mask([x,y,z]))
    #Prepare (x,y,z) tuples
    xyz = np.array([x[valid_idx], y[valid_idx], z[valid_idx]]).T
    #Define coordinate transformation
    cT = osr.CoordinateTransformation(in_srs, out_srs)
    #Loop through each point
    xyz2 = np.array([cT.TransformPoint(xi,yi,zi) for (xi,yi,zi) in xyz]).T
    #Fill in the masked array
    if xyz2.shape[1] == 1:
        xyz2 = xyz2.squeeze()
        x2, y2, z2 = xyz2[0], xyz2[1], xyz2[2] 
    else:
        x2 = np.zeros_like(x)
        y2 = np.zeros_like(y)
        z2 = np.zeros_like(z)
        x2[valid_idx] = xyz2[0]
        y2[valid_idx] = xyz2[1]
        z2[valid_idx] = xyz2[2]
    return x2, y2, z2

def ll2ecef(lon, lat, z=0.0):
    return cT_helper(lon, lat, z, wgs_srs, ecef_srs)
    
def ecef2ll(x, y, z):
    return cT_helper(x, y, z, ecef_srs, wgs_srs)

def ll2itrf(lon, lat, z=0.0):
    return cT_helper(lon, lat, z, wgs_srs, itrf_srs)

def itrf2ll(x, y, z):
    return cT_helper(x, y, z, itrf_srs, wgs_srs)

def tp2wgs(x, y, z):
    return cT_helper(x, y, z, tp_srs, wgs_srs)

def wgs2tp(x, y, z):
    return cT_helper(x, y, z, wgs_srs, tp_srs)

#Note: the lat/lon values returned here might be susceptible to rounding errors 
#Or are these true offsets due to dz?
#120.0 -> 119.99999999999999
#def geoid2ell(lon, lat, z=0.0, geoid=egm96_srs):
def geoid2ell(lon, lat, z=0.0, geoid=egm08_srs):
    llz = cT_helper(lon, lat, z, geoid, wgs_srs)
    return lon, lat, llz[2]

#def ell2geoid(lon, lat, z=0.0, geoid=egm96_srs):
def ell2geoid(lon, lat, z=0.0, geoid=egm08_srs):
    llz = cT_helper(lon, lat, z, wgs_srs, geoid)
    return lon, lat, llz[2]

def ll2nps(lon, lat, z=0.0):
    #Should throw error here
    if np.any(lat < 0.0):
        print("Warning: latitude out of range for output projection")
    return cT_helper(lon, lat, z, wgs_srs, nps_srs)

def nps2ll(x, y, z=0.0):
    return cT_helper(x, y, z, nps_srs, wgs_srs)

def ll2sps(lon, lat, z=0.0):
    if np.any(lat > 0.0):
        print("Warning: latitude out of range for output projection")
    return cT_helper(lon, lat, z, wgs_srs, sps_srs)

def sps2ll(x, y, z=0.0):
    return cT_helper(x, y, z, sps_srs, wgs_srs)

def scale_ps_ds(ds):
    clat, clon = get_center(ds)
    return scale_ps(clat)

def nps2geoid(x, y, z=0.0, geoid=nps_egm08_srs):
    return cT_helper(x, y, z, nps_srs, geoid)

def sps2geoid(x, y, z=0.0, geoid=sps_egm08_srs):
    return cT_helper(x, y, z, sps_srs, geoid)

def localtmerc_ds(ds):
    lon, lat = get_center(ds, t_srs=wgs_srs) 
    return localtmerc(lon, lat)

def localtmerc(lon, lat):
    local_srs = osr.SpatialReference()
    local_proj = '+proj=tmerc +lat_0=%0.7f +lon_0=%0.7f +datum=WGS84 +units=m +no_defs ' % (lat, lon)
    local_srs.ImportFromProj4(local_proj)
    return local_srs

def localortho_ds(ds):
    lon, lat = get_center(ds, t_srs=wgs_srs) 
    return localortho(lon, lat)

def localortho(lon, lat):
    """Create srs for local orthographic projection centered at lat, lon
    """
    local_srs = osr.SpatialReference()
    local_proj = '+proj=ortho +lat_0=%0.7f +lon_0=%0.7f +datum=WGS84 +units=m +no_defs ' % (lat, lon)
    local_srs.ImportFromProj4(local_proj)
    return local_srs

#Transform geometry to local orthographic projection, useful for width/height and area calc
def geom2localortho(geom):
    """Convert existing geom to local orthographic projection

    Useful for local cartesian distance/area calculations
    """
    cx, cy = geom.Centroid().GetPoint_2D()
    lon, lat, z = cT_helper(cx, cy, 0, geom.GetSpatialReference(), wgs_srs)
    local_srs = localortho(lon,lat)
    local_geom = geom_dup(geom)
    geom_transform(local_geom, local_srs)
    return local_geom 

def ll2local(lon, lat, z=0, local_srs=None):
    if local_srs is None: 
        lonm = lon.mean()
        latm = lat.mean()
        local_srs = localortho(lonm, latm)
    return cT_helper(lon, lat, z, wgs_srs, local_srs)

def sps2local(x, y, z=0, local_srs=None):
    if local_srs is None:
        xm = x.mean()
        ym = y.mean()
        lon, lat, z = sps2ll(xm, ym, z)
        local_srs = localortho(lon, lat)
    return cT_helper(x, y, z, sps_srs, local_srs)

def lldist(pt1, pt2):
    (lon1, lat1) = pt1 
    (lon2, lat2) = pt2 
    from vincenty import vincenty
    d = vincenty((lat1, lon1), (lat2, lon2))
    return d

#Scaling factor for area calculations in polar stereographic
#Should multiply the returned value by computed ps area to obtain true area
def scale_ps(lat):
    """
    This function calculates the scaling factor for a polar stereographic
    projection (ie. SSM/I grid) to correct area calculations. The scaling
    factor is defined (from Snyder, 1982, Map Projections used by the U.S.
    Geological Survey) as:

    k = (mc/m)*(t/tc), where:

    m = cos(lat)/sqrt(1 - e2*sin(lat)^2)
    t = tan(Pi/4 - lat/2)/((1 - e*sin(lat))/(1 + e*sin(lat)))^(e/2)
    e2 = 0.006693883 is the earth eccentricity (Hughes ellipsoid)
    e = sqrt(e2)
    mc = m at the reference latitude (70 degrees)
    tc = t at the reference latitude (70 degrees)

    The ratio mc/tc is precalculated and stored in the variable m70_t70.

    From Ben Smith PS scale m file (7/12/12)
    """
    lat = np.array(lat)
    if np.any(lat > 0):
        m70_t70 = 1.9332279 
        #Hack to deal with pole
        lat[lat>=90.0] = 89.999999999
    else:
        # for 71 deg, southern PS  -- checked BS 5/2012
        m70_t70 = 1.93903005  
        lat[lat<=-90.0] = -89.999999999

    #for WGS84, a=6378137, 1/f = 298.257223563 -> 1-sqrt(1-e^2) = f
    #-> 1-(1-f)^2 = e2 =    0.006694379990141
    #e2 = 0.006693883
    e2 = 0.006694379990141  # BS calculated from WGS84 parameters 5/2012
    e = np.sqrt(e2)

    lat = np.abs(np.deg2rad(lat))
    slat = np.sin(lat)
    clat = np.cos(lat)

    m = clat/np.sqrt(1. - e2*slat**2)
    t = np.tan(np.pi/4 - lat/2)/((1. - e*slat)/(1. + e*slat))**(e/2)
    k = m70_t70*t/m

    scale=(1./k)
    return scale

def wraplon(lon):
    lon = lon % 360.0
    return lon

def lon360to180(lon):
    """Convert longitude from (0, 360) to (-180, 180)
    """
    if np.any(lon > 360.0) or np.any(lon < 0.0):
        print("Warning: lon outside expected range")
        lon = wraplon(lon)
    #lon[lon > 180.0] -= 360.0
    #lon180 = (lon+180) - np.floor((lon+180)/360)*360 - 180
    lon = lon - (lon.astype(int)/180)*360.0
    return lon

def lon180to360(lon):
    """Convert longitude from (-180, 180) to (0, 360)
    """
    if np.any(lon > 180.0) or np.any(lon < -180.0):
        print("Warning: lon outside expected range")
        lon = lon360to180(lon)
    #lon[lon < 0.0] += 360.0
    lon = (lon + 360.0) % 360.0
    return lon

#Want to accept np arrays for these
def dd2dms(dd):
    """Convert decimal degrees to degrees, minutes, seconds
    """
    n = dd < 0
    dd = abs(dd)
    m,s = divmod(dd*3600,60)
    d,m = divmod(m,60)
    if n:
        d = -d
    return d,m,s

def dms2dd(d,m,s):
    """Convert degrees, minutes, seconds to decimal degrees
    """
    if d < 0:
        sign = -1
    else:
        sign = 1
    dd = sign * (int(abs(d)) + float(m) / 60 + float(s) / 3600)
    return dd

#Note: this needs some work, not sure what input str format was supposed to be
def dms2dd_str(dms_str, delim=' ', fmt=None):
    import re
    #dms_str = re.sub(r'\s', '', dms_str)
    if re.search('[swSW]', dms_str):
        sign = -1
    else:
        sign = 1
    #re.split('\s+', s)
    #(degree, minute, second, frac_seconds) = map(int, re.split('\D+', dms_str))
    #(degree, minute, second) = dms_str.split(delim)[0:3]
    #Remove consequtive delimiters (empty string records)
    (degree, minute, second) = [s for s in dms_str.split(delim) if s]
    #dd = sign * (int(degree) + float(minute) / 60 + float(second) / 3600 + float(frac_seconds) / 36000)
    dd = dms2dd(int(degree)*sign, int(minute), float(second)) 
    return dd

def dm2dd(d,m):
    """Convert degrees, decimal minutes to decimal degrees
    """
    dd = dms2dd(d,m,0)
    return dd

def dd2dm(dd):
    """Convert decimal to degrees, decimal minutes
    """
    d,m,s = dd2dms(dd)
    m = m + float(s)/3600
    return d,m,s 

def mapToPixel(mX, mY, geoTransform):
    """Convert map coordinates to pixel coordinates based on geotransform
    
    Accepts float or NumPy arrays

    GDAL model used here - upper left corner of upper left pixel for mX, mY (and in GeoTransform)
    """
    mX = np.asarray(mX)
    mY = np.asarray(mY)
    if geoTransform[2] + geoTransform[4] == 0:
        pX = ((mX - geoTransform[0]) / geoTransform[1]) - 0.5
        pY = ((mY - geoTransform[3]) / geoTransform[5]) - 0.5
    else:
        pX, pY = applyGeoTransform(mX, mY, invertGeoTransform(geoTransform))
    #return int(pX), int(pY)
    return pX, pY

#Add 0.5 px offset to account for GDAL model - gt 0,0 is UL corner, pixel 0,0 is center
def pixelToMap(pX, pY, geoTransform):
    """Convert pixel coordinates to map coordinates based on geotransform
    
    Accepts float or NumPy arrays

    GDAL model used here - upper left corner of upper left pixel for mX, mY (and in GeoTransform)
    """
    pX = np.asarray(pX, dtype=float)
    pY = np.asarray(pY, dtype=float)
    pX += 0.5
    pY += 0.5
    mX, mY = applyGeoTransform(pX, pY, geoTransform)
    return mX, mY

#Keep this clean and deal with 0.5 px offsets in pixelToMap
def applyGeoTransform(inX, inY, geoTransform):
    inX = np.asarray(inX)
    inY = np.asarray(inY)
    outX = geoTransform[0] + inX * geoTransform[1] + inY * geoTransform[2]
    outY = geoTransform[3] + inX * geoTransform[4] + inY * geoTransform[5]
    return outX, outY

def invertGeoTransform(geoTransform):
    # we assume a 3rd row that is [1 0 0]
    # compute determinate
    det = geoTransform[1] * geoTransform[5] - geoTransform[2] * geoTransform[4]
    if abs(det) < 0.000000000000001:
        return
    invDet = 1.0 / det
    # compute adjoint and divide by determinate
    outGeoTransform = [0, 0, 0, 0, 0, 0]
    outGeoTransform[1] = geoTransform[5] * invDet
    outGeoTransform[4] = -geoTransform[4] * invDet
    outGeoTransform[2] = -geoTransform[2] * invDet
    outGeoTransform[5] = geoTransform[1] * invDet
    outGeoTransform[0] = (geoTransform[2] * geoTransform[3] - geoTransform[0] * geoTransform[5]) * invDet
    outGeoTransform[3] = (-geoTransform[1] * geoTransform[3] + geoTransform[0] * geoTransform[4]) * invDet
    return outGeoTransform

def block_stats(x,y,z,ds,stat='median',bins=None):
    """Compute points on a regular grid (matching input GDAL Dataset) from scattered point data using specified statistic

    Wrapper for  scipy.stats.binned_statistic_2d
    
    Note: this is very fast for mean, std, count, but bignificantly slower for median
    """
    import scipy.stats as stats
    extent = ds_extent(ds)
    #[[xmin, xmax], [ymin, ymax]]
    range = [[extent[0], extent[2]], [extent[1], extent[3]]]
    if bins is None:
        bins = (ds.RasterXSize, ds.RasterYSize)
    if stat == 'max':
        stat = np.max
    elif stat == 'min':
        stat = np.min
    #block_count, xedges, yedges, bin = stats.binned_statistic_2d(x,y,z,'count',bins,range)
    block_stat, xedges, yedges, bin = stats.binned_statistic_2d(x,y,z,stat,bins,range)
    #Get valid blocks
    #if (stat == 'median') or (stat == 'mean'):
    if stat in ('median', 'mean', np.max, np.min):
        idx = ~np.isnan(block_stat)
    else:
        idx = (block_stat != 0)
    idx_idx = idx.nonzero()
    #Cell centers
    res = [(xedges[1] - xedges[0]), (yedges[1] - yedges[0])]
    out_x = xedges[:-1]+res[0]/2.0
    out_y = yedges[:-1]+res[1]/2.0
    out_x = out_x[idx_idx[0]]
    out_y = out_y[idx_idx[1]]
    out_z = block_stat[idx]
    return out_x, out_y, out_z

#Note: the above method returns block_stat, which is already a continuous grid
#Just need to account for ndv and the upper left x_edge and y_edge

def block_stats_grid(x,y,z,ds,stat='median'):
    """Fill regular grid (matching input GDAL Dataset) from scattered point data using specified statistic
    """
    mx, my, mz = block_stats(x,y,z,ds,stat)
    gt = ds.GetGeoTransform()
    pX, pY = mapToPixel(mx, my, gt)
    shape = (ds.RasterYSize, ds.RasterXSize)
    ndv = -9999.0
    a = np.full(shape, ndv)
    a[pY.astype('int'), pX.astype('int')] = mz
    return np.ma.masked_equal(a, ndv) 

#This was an abandoned attempt to split the 2D binning into smaller pieces
#Want to use crude spatial filter to chunk points, then run the median binning
#Instead, just export points and use point2dem
"""
def block_stats_grid_parallel(x,y,z,ds,stat='median'):
    extent = ds_extent(ds)
    bins = (ds.RasterXSize, ds.RasterYSize)
    res = get_res(ds)

    #Define block extents
    target_blocksize = 10000.
    blocksize = floor(target_blocksize/float(res)) * res

    xblocks = np.append(np.arange(extent[0], extent[2], blocksize), extent[2])
    yblocks = np.append(np.arange(extent[1], extent[3], blocksize), extent[3])
    for i in range(xblocks.size-1):
        for j in range(yblocks.size-1):
            extent = [xblocks[i], yblocks[j], xblocks[i+1], yblocks[j+1]]
            

    xmin, ymin, xmax, ymax = extent
    idx = ((x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax))
    x[idx], y[idx], z[idx]
    mx, my, mz = block_stats(x,y,z,ds,stat)

    import scipy.stats as stats
    range = [[extent[0], extent[2]], [extent[1], extent[3]]]
    block_stat, xedges, yedges, bin = stats.binned_statistic_2d(x,y,z,stat,bins,range)
    if stat in ('median', 'mean', np.max, np.min):
        idx = ~np.isnan(block_stat)
    else:
        idx = (block_stat != 0)
    idx_idx = idx.nonzero()
    #Cell centers
    #res = [(xedges[1] - xedges[0]), (yedges[1] - yedges[0])]
    out_x = xedges[:-1]+res[0]/2.0
    out_y = yedges[:-1]+res[1]/2.0
    out_x = out_x[idx_idx[0]]
    out_y = out_y[idx_idx[1]]
    out_z = block_stat[idx]
    return out_x, out_y, out_z

    gt = ds.GetGeoTransform()
    pX, pY = mapToPixel(mx, my, gt)
    shape = (ds.RasterYSize, ds.RasterXSize)
    ndv = -9999.0
    a = np.full(shape, ndv)
    a[pY.astype('int'), pX.astype('int')] = mz
    return np.ma.masked_equal(a, ndv) 
"""

def block_stats_grid_gen(x, y, z, res=None, srs=None, stat='median'):
    #extent = np.array([x.min(), x.max(), y.min(), y.max()])
    extent = np.array([x.min(), y.min(), x.max(), y.max()])
    if res is None:
        res = int((extent[2]-extent[0])/256.0)
    ds = mem_ds(res, extent, srs)
    return block_stats_grid(x,y,z,ds,stat), ds

#Create copy of ds in memory
#Should be able to use mem_drv CreateCopy method
#Alternative brute force implementation
#Should probably move to iolib
def mem_ds_copy(ds_orig):
    if True:
        from pygeotools.lib import iolib
        m_ds = iolib.mem_drv.CreateCopy('', ds_orig, 0)
    else:
        gt = ds_orig.GetGeoTransform()
        srs = ds_orig.GetProjection()
        dst_ns = ds_orig.RasterXSize
        dst_nl = ds_orig.RasterYSize
        nbands = ds_orig.RasterCount
        dtype = ds_orig.GetRasterBand(1).DataType
        m_ds = gdal.GetDriverByName('MEM').Create('', dst_ns, dst_nl, nbands, dtype)
        m_ds.SetGeoTransform(gt)
        m_ds.SetProjection(srs)
        for n in range(nbands):
            b = ds_orig.GetRasterBand(n+1)
            ndv = b.GetNoDataValue()
            #m_ds.AddBand()
            m_ds.GetRasterBand(n+1).WriteArray(b.ReadAsArray())
            m_ds.GetRasterBand(n+1).SetNoDataValue(ndv)
    return m_ds

def mem_ds(res, extent, srs=None, dtype=gdal.GDT_Float32):
    """Create a new GDAL Dataset in memory

    Useful for various applications that require a Dataset
    """
    #These round down to int
    #dst_ns = int((extent[2] - extent[0])/res)
    #dst_nl = int((extent[3] - extent[1])/res)
    #This should pad by 1 pixel, but not if extent and res were calculated together to give whole int
    dst_ns = int((extent[2] - extent[0])/res + 0.99)
    dst_nl = int((extent[3] - extent[1])/res + 0.99)
    m_ds = gdal.GetDriverByName('MEM').Create('', dst_ns, dst_nl, 1, dtype)
    m_gt = [extent[0], res, 0, extent[3], 0, -res]
    m_ds.SetGeoTransform(m_gt)
    if srs is not None:
        m_ds.SetProjection(srs.ExportToWkt())
    return m_ds

#Modify proj/gt of dst_fn in place
def copyproj(src_fn, dst_fn, gt=True):
    """Copy projection and geotransform from one raster file to another
    """
    src_ds = gdal.Open(src_fn, gdal.GA_ReadOnly)
    dst_ds = gdal.Open(dst_fn, gdal.GA_Update)
    dst_ds.SetProjection(src_ds.GetProjection())
    if gt:
        src_gt = np.array(src_ds.GetGeoTransform())
        src_dim = np.array([src_ds.RasterXSize, src_ds.RasterYSize])
        dst_dim = np.array([dst_ds.RasterXSize, dst_ds.RasterYSize])
        #This preserves dst_fn resolution
        if np.any(src_dim != dst_dim):
            res_factor = src_dim/dst_dim.astype(float)
            src_gt[[1, 5]] *= max(res_factor)
            #src_gt[[1, 5]] *= min(res_factor)
            #src_gt[[1, 5]] *= res_factor
        dst_ds.SetGeoTransform(src_gt)
    src_ds = None
    dst_ds = None

def geom_dup(geom):
    """Create duplicate geometry
    
    Needed to avoid segfault when passing geom around. See: http://trac.osgeo.org/gdal/wiki/PythonGotchas
    """
    g = ogr.CreateGeometryFromWkt(geom.ExportToWkt())
    g.AssignSpatialReference(geom.GetSpatialReference())
    return g 

#This should be a function of a new geom class
#Assumes geom has srs defined
#Modifies geom in place
def geom_transform(geom, t_srs):
    """Transform a geometry in place
    """
    s_srs = geom.GetSpatialReference()
    if not s_srs.IsSame(t_srs):
        ct = osr.CoordinateTransformation(s_srs, t_srs)
        geom.Transform(ct)
        geom.AssignSpatialReference(t_srs)

def shp_fieldnames(lyr):
    fdef = lyr.GetLayerDefn()
    f_list = []
    for i in range(fdef.GetFieldCount()):
        f_list.append(fdef.GetFieldDefn(i).GetName())
    return f_list

def shp_dict(shp_fn, fields=None, geom=True):
    """Get a dictionary for all features in a shapefile
    
    Optionally, specify fields
    """
    from pygeotools.lib import timelib
    ds = ogr.Open(shp_fn)
    lyr = ds.GetLayer()
    nfeat = lyr.GetFeatureCount()
    print('%i input features\n' % nfeat)
    if fields is None:
        fields = shp_fieldnames(lyr)
    d_list = []
    for n,feat in enumerate(lyr):
        d = {}
        if geom:
            geom = feat.GetGeometryRef()
            d['geom'] = geom
        for f_name in fields:
            i = str(feat.GetField(f_name))
            if 'date' in f_name:
                # date_f = f_name
                #If d is float, clear off decimal
                i = i.rsplit('.')[0]
                i = timelib.strptime_fuzzy(str(i))
            d[f_name] = i
        d_list.append(d)
    #d_list_sort = sorted(d_list, key=lambda k: k[date_f])
    return d_list

def lyr_proj(lyr, t_srs, preserve_fields=True):
    """Reproject an OGR layer
    """
    #Need to check t_srs
    s_srs = lyr.GetSpatialRef()
    cT = osr.CoordinateTransformation(s_srs, t_srs)

    #Do everything in memory
    drv = ogr.GetDriverByName('Memory')

    #Might want to save clipped, warped shp to disk?
    # create the output layer
    #drv = ogr.GetDriverByName('ESRI Shapefile')
    #out_fn = '/tmp/temp.shp'
    #if os.path.exists(out_fn):
    #    driver.DeleteDataSource(out_fn)
    #out_ds = driver.CreateDataSource(out_fn)
    
    out_ds = drv.CreateDataSource('out')
    outlyr = out_ds.CreateLayer('out', srs=t_srs, geom_type=lyr.GetGeomType())

    if preserve_fields:
        # add fields
        inLayerDefn = lyr.GetLayerDefn()
        for i in range(0, inLayerDefn.GetFieldCount()):
            fieldDefn = inLayerDefn.GetFieldDefn(i)
            outlyr.CreateField(fieldDefn)
        # get the output layer's feature definition
    outLayerDefn = outlyr.GetLayerDefn()

    # loop through the input features
    inFeature = lyr.GetNextFeature()
    while inFeature:
        # get the input geometry
        geom = inFeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(cT)
        # create a new feature
        outFeature = ogr.Feature(outLayerDefn)
        # set the geometry and attribute
        outFeature.SetGeometry(geom)
        if preserve_fields:
            for i in range(0, outLayerDefn.GetFieldCount()):
                outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
        # add the feature to the shapefile
        outlyr.CreateFeature(outFeature)
        # destroy the features and get the next input feature
        inFeature = lyr.GetNextFeature()
    #NOTE: have to operate on ds here rather than lyr, otherwise segfault
    return out_ds

#See https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html#convert-vector-layer-to-array
#Should check srs, as shp could be WGS84
def shp2array(shp_fn, r_ds=None, res=None, extent=None, t_srs=None):
    """Rasterize input shapefile to match existing raster Dataset (or specified res/extent/t_srs)
    """
    shp_ds = ogr.Open(shp_fn)
    lyr = shp_ds.GetLayer()
    #This returns xmin, ymin, xmax, ymax
    shp_extent = lyr_extent(lyr)
    shp_srs = lyr.GetSpatialRef()
    # dst_dt = gdal.GDT_Byte
    ndv = 0
    if r_ds is not None:
        r_extent = ds_extent(r_ds)
        res = get_res(r_ds, square=True)[0] 
        if extent is None:
            extent = r_extent
        r_srs = get_ds_srs(r_ds)
        r_geom = ds_geom(r_ds)
        # dst_ns = r_ds.RasterXSize
        # dst_nl = r_ds.RasterYSize
        #Convert raster extent to shp_srs
        cT = osr.CoordinateTransformation(r_srs, shp_srs)
        r_geom_reproj = geom_dup(r_geom)
        r_geom_reproj.Transform(cT)
        r_geom_reproj.AssignSpatialReference(t_srs)
        lyr.SetSpatialFilter(r_geom_reproj)
        #lyr.SetSpatialFilter(ogr.CreateGeometryFromWkt(wkt))
    else:
        #TODO: clean this up
        if res is None:
            sys.exit("Must specify input res")
        if extent is None:
            print("Using input shp extent")
            extent = shp_extent
    if t_srs is None:
        t_srs = r_srs
    if not shp_srs.IsSame(t_srs):
        print("Input shp srs: %s" % shp_srs.ExportToProj4())
        print("Specified output srs: %s" % t_srs.ExportToProj4())
        out_ds = lyr_proj(lyr, t_srs)
        outlyr = out_ds.GetLayer()
    else:
        outlyr = lyr
    #outlyr.SetSpatialFilter(r_geom)
    m_ds = mem_ds(res, extent, srs=t_srs, dtype=gdal.GDT_Byte)
    b = m_ds.GetRasterBand(1)
    b.SetNoDataValue(ndv)
    gdal.RasterizeLayer(m_ds, [1], outlyr, burn_values=[1])
    a = b.ReadAsArray()
    a = ~(a.astype('Bool'))
    return a

def raster_shpclip(r_fn, shp_fn, extent='raster', bbox=False, pad=None, invert=False, verbose=False):
    """Clip an input raster by input polygon shapefile for given extent

    """
    from pygeotools.lib import iolib, warplib

    r_ds = iolib.fn_getds(r_fn)
    r_srs = get_ds_srs(r_ds)
    r_extent = ds_extent(r_ds)
    r_extent_geom = bbox2geom(r_extent)

    #NOTE: want to add spatial filter here to avoid reprojeting global RGI polygons, for example
    
    shp_ds = ogr.Open(shp_fn)
    lyr = shp_ds.GetLayer()
    shp_srs = lyr.GetSpatialRef()
    if not r_srs.IsSame(shp_srs):
        shp_ds = lyr_proj(lyr, r_srs)
        lyr = shp_ds.GetLayer()
    #This returns xmin, ymin, xmax, ymax
    shp_extent = lyr_extent(lyr)
    shp_extent_geom = bbox2geom(shp_extent)

    #Define the output - can set to either raster or shp
    #Could accept as cl arg
    out_srs = r_srs

    if extent == 'raster':
        out_extent = r_extent 
    elif extent == 'shp':
        out_extent = shp_extent
    elif extent == 'intersection':
        out_extent = geom_intersection([r_extent_geom, shp_extent_geom])
    elif extent == 'union':
        out_extent = geom_union([r_extent_geom, shp_extent_geom])
    else:
        print("Unexpected extent specification, reverting to input raster extent")
        out_extent = 'raster'

    #Add padding around shp_extent
    #Should implement buffer here
    if pad is not None:
        out_extent = pad_extent(out_extent, width=pad)

    print("Raster to clip: %s\nShapefile used to clip: %s" % (r_fn, shp_fn))
    if verbose:
        print(shp_extent) 
        print(r_extent)
        print(out_extent)

    r_ds = warplib.memwarp(r_ds, extent=out_extent, t_srs=out_srs, r='cubic')
    r = iolib.ds_getma(r_ds)

    #If bbox, return without clipping, otherwise, clip to polygons
    if not bbox:
        #Create binary mask from shp
        mask = shp2array(shp_fn, r_ds)
        if invert:
            mask = ~(mask)
        #Now apply the mask
        r = np.ma.array(r, mask=mask)
    #Return both the array and the dataset, needed for writing out
    #Should probably just write r to r_ds and return r_ds 
    return r, r_ds

def shp2geom(shp_fn):
    """Extract geometries from input shapefile
    
    Need to handle multi-part geom: http://osgeo-org.1560.x6.nabble.com/Multipart-to-singlepart-td3746767.html
    """
    ds = ogr.Open(shp_fn)
    lyr = ds.GetLayer()
    srs = lyr.GetSpatialRef()
    lyr.ResetReading()
    geom_list = []
    for feat in lyr:
        geom = feat.GetGeometryRef()
        geom.AssignSpatialReference(srs)
        #Duplicate the geometry, or segfault
        #See: http://trac.osgeo.org/gdal/wiki/PythonGotchas
        #g = ogr.CreateGeometryFromWkt(geom.ExportToWkt())
        #g.AssignSpatialReference(srs)
        g = geom_dup(geom)
        geom_list.append(g)
    #geom = ogr.ForceToPolygon(' '.join(geom_list))    
    #Dissolve should convert multipolygon to single polygon 
    #return geom_list[0]
    ds = None
    return geom_list

def geom2shp(geom, out_fn, fields=False):
    """Write out a new shapefile for input geometry
    """
    from pygeotools.lib import timelib
    driverName = "ESRI Shapefile"
    drv = ogr.GetDriverByName(driverName)
    if os.path.exists(out_fn):
        drv.DeleteDataSource(out_fn)
    out_ds = drv.CreateDataSource(out_fn)
    out_lyrname = os.path.splitext(os.path.split(out_fn)[1])[0]
    geom_srs = geom.GetSpatialReference()
    geom_type = geom.GetGeometryType()
    out_lyr = out_ds.CreateLayer(out_lyrname, geom_srs, geom_type)
    if fields:
        field_defn = ogr.FieldDefn("name", ogr.OFTString)
        field_defn.SetWidth(128)
        out_lyr.CreateField(field_defn)
        field_defn = ogr.FieldDefn("path", ogr.OFTString)
        field_defn.SetWidth(254)
        out_lyr.CreateField(field_defn)
        #field_defn = ogr.FieldDefn("date", ogr.OFTString)
        #This allows sorting by date
        field_defn = ogr.FieldDefn("date", ogr.OFTInteger)
        field_defn.SetWidth(32)
        out_lyr.CreateField(field_defn)
        field_defn = ogr.FieldDefn("decyear", ogr.OFTReal)
        field_defn.SetPrecision(8)
        field_defn.SetWidth(64)
        out_lyr.CreateField(field_defn)
    out_feat = ogr.Feature(out_lyr.GetLayerDefn())
    out_feat.SetGeometry(geom)
    if fields:
        #Hack to force output extesion to tif, since out_fn is shp
        out_path = os.path.splitext(out_fn)[0] + '.tif'
        out_feat.SetField("name", os.path.split(out_path)[-1])
        out_feat.SetField("path", out_path)
        #Try to extract a date from input raster fn
        out_feat_date = timelib.fn_getdatetime(out_fn)
        if out_feat_date is not None:
            datestamp = int(out_feat_date.strftime('%Y%m%d'))
            #out_feat_date = int(out_feat_date.strftime('%Y%m%d%H%M'))
            out_feat.SetField("date", datestamp)
            decyear = timelib.dt2decyear(out_feat_date)
            out_feat.SetField("decyear", decyear)
    out_lyr.CreateFeature(out_feat)
    out_ds = None
    #return status?

def get_outline(ds, t_srs=None, scale=1.0, simplify=False, convex=False):
    """Generate outline of unmasked values in input raster

    get_outline is an attempt to reproduce the PostGIS Raster ST_MinConvexHull function

    Could potentially do the following: Extract random pts from unmasked elements, get indices, Run scipy convex hull, Convert hull indices to mapped coords

    See this: http://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask

    This generates a wkt polygon outline of valid data for the input raster
    
    Want to limit the dimensions of a, as notmasked_edges is slow: a = iolib.ds_getma_sub(ds, scale=scale)

    """
    gt = np.array(ds.GetGeoTransform())
    from pygeotools.lib import iolib
    a = iolib.ds_getma_sub(ds, scale=scale)
    #Create empty geometry
    geom = ogr.Geometry(ogr.wkbPolygon)
    #Check to make sure we have unmasked data
    if a.count() != 0:
        #Scale the gt for reduced resolution
        #The UL coords should remain the same, as any rounding will trim LR
        if (scale != 1.0):
            gt[1] *= scale
            gt[5] *= scale
        #Get srs
        ds_srs = get_ds_srs(ds)
        if t_srs is None:
            t_srs = ds_srs
        #Find the unmasked edges
        #Note: using only axis=0 from notmasked_edges will miss undercuts - see malib.get_edgemask
        #Better ways to do this - binary mask, sum (see numpy2stl)
        #edges0, edges1, edges = malib.get_edges(a)
        px = np.ma.notmasked_edges(a, axis=0)
        # coord = []
        #Combine edge arrays, reversing order and adding first point to complete polygon
        x = np.concatenate((px[0][1][::1], px[1][1][::-1], [px[0][1][0]]))
        #x = np.concatenate((edges[0][1][::1], edges[1][1][::-1], [edges[0][1][0]]))
        y = np.concatenate((px[0][0][::1], px[1][0][::-1], [px[0][0][0]]))
        #y = np.concatenate((edges[0][0][::1], edges[1][0][::-1], [edges[0][0][0]]))
        #Use np arrays for computing mapped coords
        mx, my = pixelToMap(x, y, gt)
        #Create wkt string
        geom_wkt = 'POLYGON(({0}))'.format(', '.join(['{0} {1}'.format(*a) for a in zip(mx,my)]))
        geom = ogr.CreateGeometryFromWkt(geom_wkt)
        if not ds_srs.IsSame(t_srs):
            ct = osr.CoordinateTransformation(ds_srs, t_srs)
            geom.Transform(ct)
        #Make sure geometry has correct srs assigned
        geom.AssignSpatialReference(t_srs)
        if not geom.IsValid():
            tol = gt[1] * 0.1
            geom = geom.Simplify(tol)
        #Need to get output units and extent for tolerance specification
        if simplify:
            #2 pixel tolerance
            tol = gt[1] * 2
            geom = geom.Simplify(tol)
        if convex:
            geom = geom.ConvexHull()
    else:
        print("No unmasked values found")
    return geom

#The following were originally in dem_coreg glas_proc, may require some additional cleanup
#Want to split into cascading levels, with lowest doing pixel-based sampling on array
#See demtools extract_profile.py

def ds_cT(ds, x, y, xy_srs=wgs_srs):
    """Convert input point coordinates to map coordinates that match input dataset
    """
    #Convert lat/lon to projected srs
    ds_srs = get_ds_srs(ds)
    #If xy_srs is undefined, assume it is the same as ds_srs
    mX = x
    mY = y
    if xy_srs is not None:
        if not ds_srs.IsSame(xy_srs):
            mX, mY, mZ = cT_helper(x, y, 0, xy_srs, ds_srs)
    return mX, mY

#Might be best to pass points as geom, with srs defined
def sample(ds, mX, mY, xy_srs=None, bn=1, pad=0, min_samp_perc=50, circ=False, count=False):
    """Sample input dataset at map coordinates
    
    This is a generic sampling function, and will return value derived from window (dimensions pad*2+1) around each point
   
    By default, assumes input map coords are identical to ds srs.  If different, specify xy_srs to enable conversion.
    """
    from pygeotools.lib import iolib, malib

    #Should offer option to fit plane to points and then sample values with sub-pixel precision

    shape = (ds.RasterYSize, ds.RasterXSize)
    gt = ds.GetGeoTransform()
    b = ds.GetRasterBand(bn)
    b_ndv = iolib.get_ndv_b(b)
    b_dtype = b.DataType
    np_dtype = iolib.gdal2np_dtype(b)

    #If necessary, convert input coordiantes to match ds srs
    mX, mY = ds_cT(ds, mX, mY, xy_srs=xy_srs)

    #This will sample an area corresponding to diameter of ICESat shot
    if pad == 'glas':
        spotsize = 70
        pad = int(np.ceil(((spotsize/gt[1])-1)/2))

    mX = np.atleast_1d(mX)
    mY = np.atleast_1d(mY)
    #Convert to pixel indices
    pX, pY = mapToPixel(mX, mY, gt)
    #Mask anything outside image dimensions
    pX = np.ma.masked_outside(pX, 0, shape[1]-1)
    pY = np.ma.masked_outside(pY, 0, shape[0]-1)
    common_mask = (~(np.logical_or(np.ma.getmaskarray(pX), np.ma.getmaskarray(pY)))).nonzero()[0]

    #Define x and y sample windows
    xwin=pad*2+1
    ywin=pad*2+1

    #This sets the minimum number of valid pixels, default 50%
    min_samp = int(np.ceil((min_samp_perc/100.)*xwin*ywin))

    #Create circular mask to simulate spot
    #This only makes sense for for xwin > 3 
    if circ:
        from pygeotools.lib import filtlib
        circ_mask = filtlib.circular_mask(xwin)
        min_samp = int(np.ceil((min_samp_perc/100.)*circ_mask.nonzero()[0].size))

    pX_int = pX[common_mask].data
    pY_int = pY[common_mask].data
    #Round to nearest integer indices
    pX_int = np.around(pX_int).astype(int)
    pY_int = np.around(pY_int).astype(int)
    #print("Valid extent: %i" % pX_int.size)

    #Create empty array to hold output
    #Added the valid pixel count quickly, should clean this up for more systematic stats return at each sample
    if count:
        stats = np.full((pX_int.size, 3), b_ndv, dtype=np_dtype)
    else:
        stats = np.full((pX_int.size, 2), b_ndv, dtype=np_dtype)

    r = gdal.GRA_NearestNeighbour
    #r = gdal.GRA_Cubic

    for i in range(pX_int.size):
        #Could have float offsets here with GDAL resampling
        samp = np.ma.masked_equal(b.ReadAsArray(xoff=pX_int[i]-pad, yoff=pY_int[i]-pad, win_xsize=xwin, win_ysize=ywin, resample_alg=r), b_ndv)
        if circ:
            samp = np.ma.array(samp, circ_mask)
        if samp.count() >= min_samp:
            if min_samp > 1:
                #Use mean and std
                #samp_med = samp.mean()
                #samp_mad = samp.std()
                #Use median and nmad (robust)
                samp_med = malib.fast_median(samp)
                samp_mad = malib.mad(samp)
                stats[i][0] = samp_med 
                stats[i][1] = samp_mad
                if count:
                    stats[i][2] = samp.count()
            else:
                stats[i][0] = samp[0]
                stats[i][1] = 0
                if count:
                    stats[i][2] = 1

            #vals, resid, coef = ma_fitplane(samp, gt=[0, gt[1], 0, 0, 0, gt[5]], perc=None)
            #Compute slope and aspect from plane
            #rmse = malib.rmse(resid)

    stats = np.ma.masked_equal(stats, b_ndv)
    #Create empty array with as input points
    if count:
        out = np.full((pX.size, 3), b_ndv, dtype=np_dtype)
    else:
        out = np.full((pX.size, 2), b_ndv, dtype=np_dtype)
    #Populate with valid samples
    out[common_mask, :] = stats
    out = np.ma.masked_equal(out, b_ndv)
    return out

def line2pts(geom, dl=None):
    """Given an input line geom, generate points at fixed interval
    
    Useful for extracting profile data from raster
    """
    #Extract list of (x,y) tuples at nodes
    nodes = geom.GetPoints()
    #print "%i nodes" % len(nodes)
   
    #Point spacing in map units
    if dl is None:
        nsteps=1000
        dl = geom.Length()/nsteps

    #This only works for equidistant projection!
    #l = np.arange(0, geom.Length(), dl)

    #Initialize empty lists
    l = []
    mX = []
    mY = []

    #Add first point to output lists
    l += [0]
    x = nodes[0][0]
    y = nodes[0][1]
    mX += [x]
    mY += [y]

    #Remainder
    rem_l = 0
    #Previous length (initially 0)
    last_l = l[-1]
    
    #Loop through each line segment in the feature
    for i in range(0,len(nodes)-1):
        x1, y1 = nodes[i]
        x2, y2 = nodes[i+1]
      
        #Total length of segment
        tl = np.sqrt((x2-x1)**2 + (y2-y1)**2)

        #Number of dl steps we can fit in this segment
        #This returns floor 
        steps = int((tl+rem_l)/dl)

        if steps > 0:
            dx = ((x2-x1)/tl)*dl
            dy = ((y2-y1)/tl)*dl
            rem_x = rem_l*(dx/dl)
            rem_y = rem_l*(dy/dl)
            
            #Loop through each step and append to lists
            for n in range(1, steps+1):
                l += [last_l + (dl*n)]
                #Remove the existing remainder
                x = x1 + (dx*n) - rem_x
                y = y1 + (dy*n) - rem_y
                mX += [x]
                mY += [y]

            #Note: could just build up arrays of pX, pY for entire line, then do single z extraction
            #Update the remainder
            rem_l += tl - (steps * dl)
            last_l = l[-1]
        else:
            rem_l += tl 

    return l, mX, mY 
    
#Started moving profile extraction code from extract_profile.py to here
#Need to clean this up
def extract_profile(ds, geom, dl=None, km=False):
    l, mX, mY = line2pts(geom, dl)
    #Need to convert to same srs
    pX, pY = mapToPixel(np.array(mX), np.array(mY), ds.GetGeoTransform())
    l = np.array(l)
    if km:
        l /= 1000.
    z = sample(ds, mX, mY)
    return (l, mX, mY, z)

def get_res_stats(ds_list, t_srs=None):
    """Return resolution stats for an input dataset list
    """
    if t_srs is None:
        t_srs = get_ds_srs(ds_list[0]) 
    res = np.array([get_res(ds, t_srs=t_srs) for ds in ds_list])
    #Check that all projections are identical
    #gt_array = np.array([ds.GetGeoTransform() for ds in args])
    #xres = gt_array[:,1]
    #yres = -gt_array[:,5]
    #if xres == yres:
    #res = np.concatenate((xres, yres))
    min = np.min(res)
    max = np.max(res)
    mean = np.mean(res)
    med = np.median(res)
    return (min, max, mean, med)

def get_res(ds, t_srs=None, square=False):
    """Get GDAL Dataset raster resolution
    """
    gt = ds.GetGeoTransform()
    ds_srs = get_ds_srs(ds)
    #This is Xres, Yres
    res = [gt[1], np.abs(gt[5])]
    if square:
        res = [np.mean(res), np.mean(res)]
    if t_srs is not None and not ds_srs.IsSame(t_srs):
        if True:
            #This diagonal approach is similar to the approach in gdaltransformer.cpp
            #Bad news for large extents near the poles
            #ullr = get_ullr(ds, t_srs)
            #diag = np.sqrt((ullr[0]-ullr[2])**2 + (ullr[1]-ullr[3])**2)
            extent = ds_extent(ds, t_srs)
            diag = np.sqrt((extent[2]-extent[0])**2 + (extent[3]-extent[1])**2)
            res = diag / np.sqrt(ds.RasterXSize**2 + ds.RasterYSize**2)
            res = [res, res]
        else:        
            #Compute from center pixel
            ct = osr.CoordinateTransformation(ds_srs, t_srs)
            pt = get_center(ds)
            #Transform center coordinates
            pt_ct = ct.TransformPoint(*pt)
            #Transform center + single pixel offset coordinates
            pt_ct_plus = ct.TransformPoint(pt[0] + gt[1], pt[1] + gt[5])
            #Compute resolution in new units 
            res = [pt_ct_plus[0] - pt_ct[0], np.abs(pt_ct_plus[1] - pt_ct[1])]
    return res

def get_center(ds, t_srs=None):
    """Get center coordinates of GDAL Dataset
    """
    gt = ds.GetGeoTransform()
    ds_srs = get_ds_srs(ds)
    #Note: this is center of center pixel, not ul corner of center pixel
    center = [gt[0] + (gt[1] * ds.RasterXSize/2.0), gt[3] + (gt[5] * ds.RasterYSize/2.0)]
    #include t_srs.Validate() and t_srs.Fixup()
    if t_srs is not None and not ds_srs.IsSame(t_srs):
        ct = osr.CoordinateTransformation(ds_srs, t_srs)
        center = list(ct.TransformPoint(*center)[0:2])
    return center

def get_ds_srs(ds):
    """Get srs object for GDAL Datset
    """
    ds_srs = osr.SpatialReference()
    ds_srs.ImportFromWkt(ds.GetProjectionRef())
    return ds_srs

def srs_check(ds):
    """Check validitiy of Dataset srs

    Return True if srs is properly defined
    """
    # ds_srs = get_ds_srs(ds)
    gt = np.array(ds.GetGeoTransform())
    gt_check = ~np.all(gt == np.array((0.0, 1.0, 0.0, 0.0, 0.0, 1.0)))
    proj_check = (ds.GetProjection() != '')
    #proj_check = ds_srs.IsProjected()
    out = False
    if gt_check and proj_check:  
        out = True
    return out

def ds_IsEmpty(ds):
    """Check to see if dataset is empty after warp
    """
    out = False
    b = ds.GetRasterBand(1)
    #Looks like this throws:
    #ERROR 1: Failed to compute min/max, no valid pixels found in sampling.
    #Should just catch this rater than bothering with logic below
    try: 
        mm = b.ComputeRasterMinMax()
        if (mm[0] == mm[1]): 
            ndv = b.GetNoDataValue()
            if ndv is None:
                out = True
            else:
                if (mm[0] == ndv):
                    out = True
    except Exception:
        out = True 
    #Check for std of nan
    #import math
    #stats = b.ComputeStatistics(1)
    #for x in stats:
    #    if math.isnan(x):
    #       out = True
    #       break
    return out

def gt_corners(gt, nx, ny):
    """Get corner coordinates based on input geotransform and raster dimensions
   """
    ul = [gt[0], gt[3]]
    ll = [gt[0], gt[3] + (gt[5] * ny)]
    ur = [gt[0] + (gt[1] * nx), gt[3]]
    lr = [gt[0] + (gt[1] * nx), gt[3] + (gt[5] * ny)]
    return ul, ll, ur, lr

"""
Notes on extent format:
gdalwarp uses '-te xmin ymin xmax ymax'
gdalbuildvrt uses '-te xmin ymin xmax ymax'
gdal_translate uses '-projwin ulx uly lrx lry' or '-projwin xmin ymax xmax ymin'

These functions should all use 'xmin ymin xmax ymax' for extent, unless otherwise specified
"""

def corner_extent(ul, ll, ur, lr): 
    """Get min/max extent based on corner coord
    """
    xmin = min(ul[0], ll[0], ur[0], lr[0])
    xmax = max(ul[0], ll[0], ur[0], lr[0])
    ymin = min(ul[1], ll[1], ur[1], lr[1])
    ymax = max(ul[1], ll[1], ur[1], lr[1])
    extent = [xmin, ymin, xmax, ymax]
    return extent
    
#This is called by malib.DEM_stack, where we don't necessarily have a ds
def gt_extent(gt, nx, ny):
    extent = corner_extent(*gt_corners(gt, nx, ny))
    return extent 

def ds_extent(ds, t_srs=None):
    """Return min/max extent of dataset based on corner coordinates

    xmin, ymin, xmax, ymax

    If t_srs is specified, output will be converted to specified srs
    """
    ul, ll, ur, lr = gt_corners(ds.GetGeoTransform(), ds.RasterXSize, ds.RasterYSize) 
    ds_srs = get_ds_srs(ds) 
    if t_srs is not None and not ds_srs.IsSame(t_srs):
        ct = osr.CoordinateTransformation(ds_srs, t_srs)
        #Check to see if ct creation failed
        #if ct == NULL:
        #Check to see if transform failed
        #if not ct.TransformPoint(extent[0], extent[1]):
        #Need to check that transformed coordinates fall within appropriate bounds
        ul = ct.TransformPoint(*ul)
        ll = ct.TransformPoint(*ll)
        ur = ct.TransformPoint(*ur)
        lr = ct.TransformPoint(*lr)
    extent = corner_extent(ul, ll, ur, lr)
    return extent 

#This rounds to nearest multiple of a
def round_nearest(x, a):
    return round(round(x / a) * a, -int(np.floor(np.log10(a))))

#Round extents to nearest pixel
#Should really pad these outward rather than round
def extent_round(extent, precision=1E-3):
    #Should force initial stack reation to multiples of res
    extround = [round_nearest(i, precision) for i in extent]
    #Check that bounds are still within original extent
    if False:
        extround[0] = max(extent[0], extround[0])
        extround[1] = max(extent[1], extround[1])
        extround[2] = min(extent[2], extround[2])
        extround[3] = min(extent[3], extround[3])
    return extround

def ds_geom(ds, t_srs=None):
    """Return dataset bbox envelope as geom
    """
    gt = ds.GetGeoTransform()
    ds_srs = get_ds_srs(ds)
    if t_srs is None:
        t_srs = ds_srs
    ns = ds.RasterXSize
    nl = ds.RasterYSize
    x = np.array([0, ns, ns, 0, 0], dtype=float)
    y = np.array([0, 0, nl, nl, 0], dtype=float)
    #Note: pixelToMap adds 0.5 to input coords, need to account for this here
    x -= 0.5
    y -= 0.5
    mx, my = pixelToMap(x, y, gt)
    geom_wkt = 'POLYGON(({0}))'.format(', '.join(['{0} {1}'.format(*a) for a in zip(mx,my)]))
    geom = ogr.CreateGeometryFromWkt(geom_wkt)
    geom.AssignSpatialReference(ds_srs)
    if not ds_srs.IsSame(t_srs):
        geom_transform(geom, t_srs)
    return geom

def geom_extent(geom):
    #Envelope is ul_x, ur_x, lr_y, ll_y (?)
    env = geom.GetEnvelope()
    #return xmin, ymin, xmax, ymax 
    return [env[0], env[2], env[1], env[3]]

def lyr_extent(lyr):
    #Envelope is ul_x, ur_x, lr_y, ll_y (?)
    env = lyr.GetExtent()
    #return xmin, ymin, xmax, ymax 
    return [env[0], env[2], env[1], env[3]]

#Compute dataset extent using geom
def ds_geom_extent(ds, t_srs=None):
    geom = ds_geom(ds, t_srs)
    return geom_extent(geom)

#Quick and dirty filter to check for points inside bbox
def pt_within_extent(x, y, extent):
    xmin, ymin, xmax, ymax = extent
    idx = ((x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)) 
    return idx

#Pad extent
#Want to rewrite to allow for user-specified map units in addition to percentage
def pad_extent(extent, perc=0.1, width=None, uniform=False):
    e = np.array(extent)
    if width is not None:
        dx = dy = width
        out = e + np.array([-dx, -dy, dx, dy])
    else:
        dx = e[2] - e[0]
        dy = e[3] - e[1]
        if uniform:
            dx = dy = np.mean([dx, dy])
        out = e + (perc * np.array([-dx, -dy, dx, dy]))
    return list(out)

#What happens if input geom have different t_srs???
#Add option to return envelope, don't need additional functions to do this
#Note: this can return multipolygon geometry!
def geom_union(geom_list, **kwargs):
    convex=False
    union = geom_list[0]
    for geom in geom_list[1:]:
        union = union.Union(geom)
    if convex:
        union = union.ConvexHull()
    return union

def ds_geom_union(ds_list, **kwargs):
    ref_srs = get_ds_srs(ds_list[0]) 
    if 't_srs' in kwargs: 
        if kwargs['t_srs'] is not None:
            if not ref_srs.IsSame(kwargs['t_srs']):
                ref_srs = kwargs['t_srs']
    geom_list = []
    for ds in ds_list:
        geom_list.append(ds_geom(ds, t_srs=ref_srs))
    union = geom_union(geom_list)
    return union

#Check to make sure we have at least 2 input ds
def ds_geom_union_extent(ds_list, **kwargs):
    union = ds_geom_union(ds_list, **kwargs)
    #Envelope is ul_x, ur_x, lr_y, lr_x
    #Define new geom class with better Envelope options?
    env = union.GetEnvelope()
    return [env[0], env[2], env[1], env[3]]

#Do we need to assign srs after the intersection here?
#intersect.AssignSpatialReference(srs)
def geom_intersection(geom_list, **kwargs):
    convex=False
    intsect = geom_list[0]
    valid = False
    for geom in geom_list[1:]:
        if intsect.Intersects(geom):
            valid = True
            intsect = intsect.Intersection(geom)
    if convex:
        intsect = intsect.ConvexHull()
    if not valid:
        intsect = None
    return intsect 

def geom_wh(geom):
    """Compute width and height of geometry in projected units
    """
    e = geom.GetEnvelope()
    h = e[1] - e[0]
    w = e[3] - e[2]
    return w, h

#Check to make sure we have at least 2 input ds
#Check to make sure intersection is valid
#***
#This means checking that stereographic projections don't extend beyond equator
#***
def ds_geom_intersection(ds_list, **kwargs):
    ref_srs = get_ds_srs(ds_list[0]) 
    if 't_srs' in kwargs: 
        if kwargs['t_srs'] is not None:
            if not ref_srs.IsSame(kwargs['t_srs']):
                ref_srs = kwargs['t_srs']
    geom_list = []
    for ds in ds_list:
        geom_list.append(ds_geom(ds, t_srs=ref_srs))
    intsect = geom_intersection(geom_list)
    return intsect 

def ds_geom_intersection_extent(ds_list, **kwargs):
    intsect = ds_geom_intersection(ds_list, **kwargs)
    if intsect is not None:
        #Envelope is ul_x, ur_x, lr_y, lr_x
        #Define new geom class with better Envelope options?
        env = intsect.GetEnvelope()
        intsect = [env[0], env[2], env[1], env[3]]
    return intsect

#This is necessary because extent precision is different
def extent_compare(e1, e2, precision=1E-3):
    #e1_f = '%0.6f %0.6f %0.6f %0.6f' % tuple(e1)
    #e2_f = '%0.6f %0.6f %0.6f %0.6f' % tuple(e2)
    e1_f = extent_round(e1, precision)
    e2_f = extent_round(e2, precision)
    return e1_f == e2_f

#This is necessary because extent precision is different
def res_compare(r1, r2, precision=1E-3):
    #r1_f = '%0.6f' % r1
    #r2_f = '%0.6f' % r2
    r1_f = round_nearest(r1, precision) 
    r2_f = round_nearest(r2, precision) 
    return r1_f == r2_f

#Clip raster by shape
#Note, this is a hack that uses gdalwarp command line util
#It is possible to do this with GDAL/OGR python API, but this works for now
#See: http://stackoverflow.com/questions/2220749/rasterizing-a-gdal-layer
def clip_raster_by_shp(dem_fn, shp_fn):
    import subprocess
    #This is ok when writing to outdir, but clip_raster_by_shp.sh writes to raster dir
    #try:
    #    with open(dem_fn) as f: pass
    #except IOError as e:
    cmd = ['clip_raster_by_shp.sh', dem_fn, shp_fn]
    print(cmd)
    subprocess.call(cmd, shell=False)
    dem_clip_fn = os.path.splitext(dem_fn)[0]+'_shpclip.tif'
    dem_clip_ds = gdal.Open(dem_clip_fn, gdal.GA_ReadOnly)
    return dem_clip_ds

#Hack
#extent is xmin ymin xmax ymax
def clip_shp(shp_fn, extent):
    import subprocess
    out_fn = os.path.splitext(shp_fn)[0]+'_clip.shp'
    #out_fn = os.path.splitext(shp_fn)[0]+'_clip'+os.path.splitext(shp_fn)[1]
    extent = [str(i) for i in extent]
    #cmd = ['ogr2ogr', '-f', 'ESRI Shapefile', out_fn, shp_fn, '-clipsrc']
    cmd = ['ogr2ogr', '-f', 'ESRI Shapefile', '-overwrite', '-t_srs', 'EPSG:3031', out_fn, shp_fn, '-clipdst']
    cmd.extend(extent)
    print(cmd)
    subprocess.call(cmd, shell=False)

#Need to combine these with shp2array
#Deal with different srs

#Rasterize shp to binary mask
def fn2mask(fn, r_ds):
    v_ds = org.Open(fn)
    mask = ds2mask(v_ds, r_ds)
    v_ds = None
    return mask

#Rasterize ogr dataset to binary mask
def ds2mask(v_ds, r_ds):
    lyr = v_ds.GetLayer()
    mask = lyr2mask(lyr, r_ds)
    lyr = None
    return mask

#Rasterize ogr layer to binary mask (core functionality for gdal.RasterizeLayer)
def lyr2mask(lyr, r_ds):
    #Create memory raster dataset and fill with 0s
    m_ds = gdal.GetDriverByName('MEM').CreateCopy('', r_ds, 1)
    b = m_ds.GetRasterBand(1)
    b.Fill(0)
    b.SetNoDataValue(0)
    #Not sure if gdal.RasterizeLayer can handle srs difference
    #r_srs = get_ds_srs(m_ds)
    #lyr_srs = lyr.GetSpatialReference()
    #if not r_srs.IsSame(lyr_srs):
    #   lyr = lyr_proj(lyr, r_srs)
    #Rasterize with values of 1
    gdal.RasterizeLayer(m_ds, [1], lyr, burn_values=[1])
    a = b.ReadAsArray()
    mask = a.astype('Bool')
    m_ds = None
    return ~mask

#Rasterize ogr geometry to binary mask (creates dummy layer) 
def geom2mask(geom, r_ds):
    geom_srs = geom.GetSpatialReference()
    geom2 = geom_dup(geom) 
    #Create memory vector dataset and add geometry as new feature
    ogr_ds = ogr.GetDriverByName('Memory').CreateDataSource('tmp_ds')
    m_lyr = ogr_ds.CreateLayer('tmp_lyr', srs=geom_srs)
    feat = ogr.Feature(m_lyr.GetLayerDefn())
    feat.SetGeometryDirectly(geom2)
    m_lyr.CreateFeature(feat)
    mask = lyr2mask(m_lyr, r_ds)
    m_lyr = None
    ogr_ds = None
    return mask

#Old function, does not work with inner rings or complex geometries
def geom2mask_PIL(geom, ds):
    from PIL import Image, ImageDraw
    #width = ma.shape[1]
    #height = ma.shape[0]
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()

    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    #Check to make sure we have polygon
    #3 is polygon, 6 is multipolygon
    #At present, this doesn't handle internal polygons
    #Want to set these to 0
    if (geom.GetGeometryType() == 3): 
        for ring in geom:
            pts = np.array(ring.GetPoints())
            px = np.array(mapToPixel(pts[:,0], pts[:,1], gt))
            px_poly = px.T.astype(int).ravel().tolist()
            draw.polygon(px_poly, outline=1, fill=1)
    elif (geom.GetGeometryType() == 6):
        for poly in geom:
            for ring in poly:
                pts = np.array(ring.GetPoints())
                px = np.array(mapToPixel(pts[:,0], pts[:,1], gt))
                px_poly = px.T.astype(int).ravel().tolist()
                draw.polygon(px_poly, outline=1, fill=1)
    # polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
    mask = np.array(img).astype(bool)
    return ~mask

def gdaldem_mem_ma(ma, ds=None, res=None, extent=None, srs=None, processing='hillshade', returnma=False, computeEdges=False):
    """
    Wrapper to allow gdaldem calculations for arbitrary NumPy masked array input
    Untested, work in progress placeholder
    Should only need to specify res, can caluclate local gt, cartesian srs
    """
    if ds is None:
        ds = mem_ds(res, extent, srs=None, dtype=gdal.GDT_Float32)
    else:
        ds = mem_ds_copy(ds)
    b = ds.GetRasterBand(1)
    b.WriteArray(ma)
    out = gdaldem_mem_ds(ds, processing=processing, returnma=returnma)
    return out 

#Should add gdal.DEMProcessingOptions support
def gdaldem_mem_ds(ds, processing='hillshade', returnma=False, computeEdges=False):
    """
    Wrapper for gdaldem functions

    Uses gdaldem API, requires GDAL v2.1+
    """
    choices = ["hillshade", "slope", "aspect", "color-relief", "TRI", "TPI", "Roughness"]
    out = None
    scale=1.0
    if not get_ds_srs(ds).IsProjected():
        scale=111120
    if processing in choices:
        out = gdal.DEMProcessing('', ds, processing, format='MEM', computeEdges=computeEdges, scale=scale)
    else:
        print("Invalid processing choice")
        print(choices)
    #This should be a separate function
    if returnma:
        from pygeotools.lib import iolib
        out = iolib.ds_getma(out)
    return out 

#Depreciated
def gdaldem_wrapper(fn, product='hs', returnma=True, verbose=True):
    """Wrapper for gdaldem functions

    Note: gdaldem is directly avaialable through API as of GDAL v2.1

    https://trac.osgeo.org/gdal/wiki/rfc59.1_utilities_as_a_library

    This function is no longer necessry, and will eventually be removed.
    """
    #These gdaldem functions should be able to ingest masked array
    #Just write out temporary file, or maybe mem vrt?
    valid_opt = ['hillshade', 'hs', 'slope', 'aspect', 'color-relief', 'TRI', 'TPI', 'roughness']
    try:
        open(fn)
    except IOError:
        print("Unable to open %s" %fn)

    if product not in valid_opt: 
        print("Invalid gdaldem option specified")

    import subprocess
    from pygeotools.lib import iolib
    bma = None
    opts = []
    if product == 'hs' or product == 'hillshade':
        product = 'hillshade'
        #opts = ['-compute_edges',]
        out_fn = os.path.splitext(fn)[0]+'_hs_az315.tif'
    else:
        out_fn = os.path.splitext(fn)[0]+'_%s.tif' % product
    if not os.path.exists(out_fn):
        cmd = ['gdaldem', product]
        cmd.extend(opts)
        cmd.extend(iolib.gdal_opt_co)
        cmd.extend([fn, out_fn])
        if verbose:
            print(' '.join(cmd))
            cmd_opt = {}
        else:
            fnull = open(os.devnull, 'w')
            cmd_opt = {'stdout':fnull, 'stderr':subprocess.STDOUT}
        subprocess.call(cmd, shell=False, **cmd_opt)

    if returnma:
        ds = gdal.Open(out_fn, gdal.GA_ReadOnly)
        bma = iolib.ds_getma(ds, 1)
        return bma 
    else:
        return out_fn

def gdaldem_slope(fn):
    return gdaldem_wrapper(fn, 'slope')

def gdaldem_aspect(fn):
    return gdaldem_wrapper(fn, 'aspect')

#Perhaps this should be generalized, and moved to malib
def bilinear(px, py, band_array, gt):
    """Bilinear interpolated point at(px, py) on band_array
    """
    #import malib
    #band_array = malib.checkma(band_array)
    ndv = band_array.fill_value
    ny, nx = band_array.shape
    # Half raster cell widths
    hx = gt[1]/2.0
    hy = gt[5]/2.0
    # Calculate raster lower bound indices from point
    fx =(px -(gt[0] + hx))/gt[1]
    fy =(py -(gt[3] + hy))/gt[5]
    ix1 = int(np.floor(fx))
    iy1 = int(np.floor(fy))
    # Special case where point is on upper bounds
    if fx == float(nx - 1):
        ix1 -= 1
    if fy == float(ny - 1):
        iy1 -= 1
    # Upper bound indices on raster
    ix2 = ix1 + 1
    iy2 = iy1 + 1
    # Test array bounds to ensure point is within raster midpoints
    if(ix1 < 0) or(iy1 < 0) or(ix2 > nx - 1) or(iy2 > ny - 1):
        return ndv
    # Calculate differences from point to bounding raster midpoints
    dx1 = px -(gt[0] + ix1*gt[1] + hx)
    dy1 = py -(gt[3] + iy1*gt[5] + hy)
    dx2 =(gt[0] + ix2*gt[1] + hx) - px
    dy2 =(gt[3] + iy2*gt[5] + hy) - py
    # Use the differences to weigh the four raster values
    div = gt[1]*gt[5]
    return(band_array[iy1,ix1]*dx2*dy2/div +
            band_array[iy1,ix2]*dx1*dy2/div +
            band_array[iy2,ix1]*dx2*dy1/div +
            band_array[iy2,ix2]*dx1*dy1/div)

#Jak values over fjord are ~30, offset is -29.99
def get_geoid_offset(ds, geoid_srs=egm08_srs):
    """Return offset for center of ds
    
    Offset is added to input (presumably WGS84 HAE) to get to geoid
    
    Note: requires vertical offset grids in proj share dir - see earlier note
    """
    ds_srs = get_ds_srs(ds)
    c = get_center(ds)
    x, y, z = cT_helper(c[0], c[1], 0.0, ds_srs, geoid_srs)
    return z

def get_geoid_offset_ll(lon, lat, geoid_srs=egm08_srs):
    x, y, z = cT_helper(lon, lat, 0.0, wgs_srs, geoid_srs)
    return z

#Note: the existing egm96-5 dataset has problematic extent
#warplib writes out correct res/extent, but egm96 is empty
#Eventually accept geoma
def wgs84_to_egm96(dem_ds, geoid_dir=None):
    from pygeotools.lib import warplib

    #Check input dem_ds - make sure WGS84

    geoid_dir = os.getenv('ASP_DATA')
    if geoid_dir is None:
        print("No geoid directory available")
        print("Set ASP_DATA or specify")
    
    egm96_fn = geoid_dir+'/geoids-1.1/egm96-5.tif' 
    try:
        open(egm96_fn)
    except IOError:
        sys.exit("Unable to find "+egm96_fn)
    egm96_ds = gdal.Open(egm96_fn)

    #Warp egm96 to match the input ma
    ds_list = warplib.memwarp_multi([dem_ds, egm96_ds], res='first', extent='first', t_srs='first') 

    #Try two-step with extent/res in wgs84
    #ds_list = warplib.memwarp_multi([dem_ds, egm96_ds], res='first', extent='intersection', t_srs='last') 
    #ds_list = warplib.memwarp_multi([dem_ds, ds_list[1]], res='first', extent='first', t_srs='first')

    print("Extracting ma from dem and egm96 ds")
    from pygeotools.lib import iolib
    dem = iolib.ds_getma(ds_list[0])
    egm96 = iolib.ds_getma(ds_list[1])

    print("Removing offset")
    dem_egm96 = dem - egm96
   
    return dem_egm96

#Run ASP dem_geoid adjustment utility
#Note: this is multithreaded
def dem_geoid(dem_fn):
    out_prefix = os.path.splitext(dem_fn)[0]
    adj_fn = out_prefix +'-adj.tif'
    if not os.path.exists(adj_fn):
        import subprocess
        cmd_args = ["-o", out_prefix, dem_fn]
        cmd = ['dem_geoid'] + cmd_args
        #cmd = 'dem_geoid -o %s %s' % (out_prefix, dem_fn)
        print(' '.join(cmd))
        subprocess.call(cmd, shell=False)
    adj_ds = gdal.Open(adj_fn, gdal.GA_ReadOnly)
    #from pygeotools.lib import iolib
    #return iolib.ds_getma(adj_ds, 1)
    return adj_ds

def dem_geoid_offsetgrid_ds(ds, out_fn=None):
    from pygeotools.lib import iolib
    a = iolib.ds_getma(ds)
    a[:] = 0.0
    if out_fn is None:
        out_fn = '/tmp/geoidoffset.tif'
    iolib.writeGTiff(a, out_fn, ds, ndv=-9999)
    import subprocess
    cmd_args = ["--geoid", "EGM2008", "-o", os.path.splitext(out_fn)[0], out_fn]
    cmd = ['dem_geoid'] + cmd_args
    print(' '.join(cmd))
    subprocess.call(cmd, shell=False)
    os.rename(os.path.splitext(out_fn)[0]+'-adj.tif', out_fn)
    o = iolib.fn_getma(out_fn)
    return o

def dem_geoid_offsetgrid(dem_fn):
    ds = gdal.Open(dem_fn)
    out_fn = os.path.splitext(dem_fn)[0]+'_EGM2008offset.tif'
    o = dem_geoid_offsetgrid_ds(ds, out_fn)
    return o

#Note: funcitonality with masking needs work
def map_interp(bma, gt, stride=1, full_array=True):
    import scipy.interpolate
    from pygeotools.lib import malib
    mx, my = get_xy_ma(bma, gt, stride, origmask=True)
    x, y, z = np.array([mx.compressed(), my.compressed(), bma.compressed()])
    #Define the domain for the interpolation
    if full_array:
        #Interpolate over entire array
        xi, yi = get_xy_ma(bma, gt, stride, origmask=False)
    else:
        #Interpolate over buffered area around points
        newmask = malib.maskfill(bma)
        newmask = malib.mask_dilate(bma, iterations=3)
        xi, yi = get_xy_ma(bma, gt, stride, newmask=newmask)
        xi = xi.compressed()
        yi = yi.compressed()
    #Do the interpolation
    zi = scipy.interpolate.griddata((x,y), z, (xi,yi), method='cubic')
    #f = scipy.interpolate.BivariateSpline(x, y, z)
    #zi = f(xi, yi, grid=False)
    #f = scipy.interpolate.interp2d(x, y, z, kind='cubic')
    #This is a 2D array, only need to specify 1D arrays of x and y for output grid
    #zi = f(xi, yi)
    if full_array:
        zia = np.ma.fix_invalid(zi, fill_value=bma.fill_value)
    else:
        pxi, pyi = mapToPixel(xi, yi, gt)
        pxi = np.clip(pxi.astype(int), 0, bma.shape[1])
        pyi = np.clip(pyi.astype(int), 0, bma.shape[0])
        zia = np.ma.masked_all_like(bma)
        zia.set_fill_value(bma.fill_value)
        zia[pyi, pxi] = zi
    return zia

def get_xy_ma(bma, gt, stride=1, origmask=True, newmask=None):
    """Return arrays of x and y map coordinates for input array and geotransform
    """
    pX = np.arange(0, bma.shape[1], stride)
    pY = np.arange(0, bma.shape[0], stride)
    psamp = np.meshgrid(pX, pY)
    #if origmask:
    #    psamp = np.ma.array(psamp, mask=np.ma.getmaskarray(bma), fill_value=0)
    mX, mY = pixelToMap(psamp[0], psamp[1], gt)
    mask = None
    if origmask:
        mask = np.ma.getmaskarray(bma)[::stride]
    if newmask is not None:
        mask = newmask[::stride]
    mX = np.ma.array(mX, mask=mask, fill_value=0)
    mY = np.ma.array(mY, mask=mask, fill_value=0)
    return mX, mY

def get_xy_1D(ds, stride=1, getval=False):
    """Return 1D arrays of x and y map coordinates for input GDAL Dataset 
    """
    gt = ds.GetGeoTransform()
    #stride = stride_m/gt[1]
    pX = np.arange(0, ds.RasterXSize, stride)
    pY = np.arange(0, ds.RasterYSize, stride)
    mX, dummy = pixelToMap(pX, pY[0], gt)
    dummy, mY = pixelToMap(pX[0], pY, gt)
    return mX, mY

def get_xy_grids(ds, stride=1, getval=False):
    """Return 2D arrays of x and y map coordinates for input GDAL Dataset 
    """
    gt = ds.GetGeoTransform()
    #stride = stride_m/gt[1]
    pX = np.arange(0, ds.RasterXSize, stride)
    pY = np.arange(0, ds.RasterYSize, stride)
    psamp = np.meshgrid(pX, pY)
    mX, mY = pixelToMap(psamp[0], psamp[1], gt)
    return mX, mY

def fitPlaneSVD(XYZ):
    """Fit a plane to input point data using SVD
    """
    [rows,cols] = XYZ.shape
    # Set up constraint equations of the form  AB = 0,
    # where B is a column vector of the plane coefficients
    # in the form b(1)*X + b(2)*Y +b(3)*Z + b(4) = 0.
    p = (np.ones((rows,1)))
    AB = np.hstack([XYZ,p])
    [u, d, v] = np.linalg.svd(AB,0)        
    # Solution is last column of v.
    B = np.array(v[3,:])
    coeff = -B[[0, 1, 3]]/B[2]
    return coeff 

def fitPlaneLSQ(XYZ):
    """Fit a plane to input point data using LSQ
    """
    [rows,cols] = XYZ.shape
    G = np.ones((rows,3))
    G[:,0] = XYZ[:,0]  #X
    G[:,1] = XYZ[:,1]  #Y
    Z = XYZ[:,2]
    coeff,resid,rank,s = np.linalg.lstsq(G,Z,rcond=None)
    return coeff

#Generic 2D polynomial fits
#https://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent
def polyfit2d(x, y, z, order=1):
    import itertools
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z, rcond=None)
    return m

def polyval2d(x, y, m):
    import itertools
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    #0 could be a valid value
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z 

#Should use numpy.polynomial.polynomial.polyvander2d
def ma_fitpoly(bma, order=1, gt=None, perc=(2,98), origmask=True):
    """Fit a plane to values in input array
    """
    if gt is None:
        gt = [0, 1, 0, 0, 0, -1] 
    #Filter, can be useful to remove outliers
    if perc is not None:
        from pygeotools.lib import filtlib
        bma_f = filtlib.perc_fltr(bma, perc)
    else:
        bma_f = bma
    #Get indices
    x, y = get_xy_ma(bma_f, gt, origmask=origmask)
    #Fit only where we have valid data
    bma_mask = np.ma.getmaskarray(bma)
    coeff = polyfit2d(x[~bma_mask].data, y[~bma_mask].data, bma[~bma_mask].data, order=order) 
    #For 1D, these are: c, y, x, xy
    print(coeff)
    #Compute values for all x and y, unless origmask=True
    vals = polyval2d(x, y, coeff) 
    resid = bma - vals
    return vals, resid, coeff

def ma_fitplane(bma, gt=None, perc=(2,98), origmask=True):
    """Fit a plane to values in input array
    """
    if gt is None:
        gt = [0, 1, 0, 0, 0, -1] 
    #Filter, can be useful to remove outliers
    if perc is not None:
        from pygeotools.lib import filtlib
        bma_f = filtlib.perc_fltr(bma, perc)
    else:
        bma_f = bma
    #Get indices
    x_f, y_f = get_xy_ma(bma_f, gt, origmask=origmask)
    #Regardless of desired output (origmask True or False), for fit, need to limit to valid pixels only 
    bma_f_mask = np.ma.getmaskarray(bma_f)
    #Create xyz stack, needed for SVD
    xyz = np.vstack((np.ma.array(x_f, mask=bma_f_mask).compressed(), \
            np.ma.array(y_f, mask=bma_f_mask).compressed(), bma_f.compressed())).T
    #coeff = fitPlaneSVD(xyz)
    coeff = fitPlaneLSQ(xyz)
    print(coeff)
    vals = coeff[0]*x_f + coeff[1]*y_f + coeff[2]
    resid = bma_f - vals
    return vals, resid, coeff

def ds_fitplane(ds):
    """Fit a plane to values in GDAL Dataset
    """
    from pygeotools.lib import iolib
    bma = iolib.ds_getma(ds)
    gt = ds.GetGeoTransform()
    return ma_fitplane(bma, gt)

#The following were moved from proj_select.py
def getUTMzone(geom):
    """Determine UTM Zone for input geometry
    """
    #If geom has srs properly defined, can do this
    #geom.TransformTo(wgs_srs)
    #Get centroid lat/lon
    lon, lat = geom.Centroid().GetPoint_2D()
    #Make sure we're -180 to 180
    lon180 = (lon+180) - np.floor((lon+180)/360)*360 - 180
    zonenum = int(np.floor((lon180 + 180)/6) + 1)
    #Determine N/S hemisphere
    if lat >= 0:
        zonehem = 'N'
    else:
        zonehem = 'S'
    #Deal with special cases
    if (lat >= 56.0 and lat < 64.0 and lon180 >= 3.0 and lon180 < 12.0):
        zonenum = 32
    if (lat >= 72.0 and lat < 84.0): 
        if (lon180 >= 0.0 and lon180 < 9.0): 
            zonenum = 31
        elif (lon180 >= 9.0 and lon180 < 21.0):
            zonenum = 33
        elif (lon180 >= 21.0 and lon180 < 33.0):
            zonenum = 35
        elif (lon180 >= 33.0 and lon180 < 42.0):
            zonenum = 37
    return str(zonenum)+zonehem

#Return UTM srs
def getUTMsrs(geom):
    utmzone = getUTMzone(geom)
    srs = osr.SpatialReference()    
    srs.SetUTM(int(utmzone[0:-1]), int(utmzone[-1] == 'N'))
    return srs

#Want to overload this to allow direct coordinate input, create geom internally
def get_proj(geom, proj_list=None):
    """Determine best projection for input geometry
    """
    out_srs = None
    if proj_list is None:
        proj_list = gen_proj_list()
    #Go through user-defined projeciton list
    for projbox in proj_list:
        if projbox.geom.Intersects(geom):
            out_srs = projbox.srs
            break
    #If geom doesn't fall in any of the user projection bbox, use UTM
    if out_srs is None:
        out_srs = getUTMsrs(geom)
    return out_srs

class ProjBox:
    """Object containing bbox geom and srs

    Used for automatic projection determination
    """
    def __init__(self, bbox, epsg):
        self.bbox = bbox
        self.geom = bbox2geom(bbox)
        self.srs = osr.SpatialReference()
        self.srs.ImportFromEPSG(epsg)

#This provides a preference order for projections
def gen_proj_list():
    """Create list of projections with cascading preference
    """
    #Eventually, just read this in from a text file
    proj_list = []
    #Alaska
    #Note, this spans -180/180
    proj_list.append(ProjBox([-180, -130, 51.35, 71.35], 3338))
    #proj_list.append(ProjBox([-130, 172.4, 51.35, 71.35], 3338))
    #Transantarctic Mountains
    proj_list.append(ProjBox([150, 175, -80, -70], 3294))
    #Greenland
    proj_list.append(ProjBox([-180, 180, 58, 82], 3413))
    #Antarctica
    proj_list.append(ProjBox([-180, 180, -90, -58], 3031))
    #Arctic
    proj_list.append(ProjBox([-180, 180, 60, 90], 3413))
    return proj_list

#This is first hack at centralizing site and projection definitions
class Site:
    def __init__(self, name, extent, srs=None, refdem_fn=None):
        #Site name: conus, hma
        self.name = name 
        #Spatial extent, list or tuple: [xmin, xmax, ymin, ymax]
        self.extent = extent 
        self.srs = srs
        self.refdem_fn = refdem_fn

site_dict = {}
#NED 1/3 arcsec (10 m)
ned13_dem_fn = '/nobackup/deshean/rpcdem/ned13/ned13_tiles_glac24k_115kmbuff.vrt'
#NED 1 arcsec (30 m)
ned1_dem_fn = '/nobackup/deshean/rpcdem/ned1/ned1_tiles_glac24k_115kmbuff.vrt'
site_dict['conus'] = Site(name='conus', extent=(-125, -104, 31, 50), srs=conus_aea_srs, refdem_fn=ned1_dem_fn)
#SRTM-GL1 1 arcsec (30 m)
srtm1_fn = '/nobackup/deshean/rpcdem/hma/srtm1/hma_srtm_gl1.vrt'
site_dict['hma'] = Site(name='hma', extent=(66, 106, 25, 47), srs=hma_aea_srs, refdem_fn=srtm1_fn)

#bbox should be [minlon, maxlon, minlat, maxlat]
#bbox should be [min_x, max_x, min_y, max_y]
def bbox2geom(bbox, t_srs=None):
    #Check bbox
    #bbox = numpy.array([-180, 180, 60, 90])
    x = [bbox[0], bbox[0], bbox[1], bbox[1], bbox[0]]
    y = [bbox[2], bbox[3], bbox[3], bbox[2], bbox[2]]
    geom_wkt = 'POLYGON(({0}))'.format(', '.join(['{0} {1}'.format(*a) for a in zip(x,y)]))
    geom = ogr.CreateGeometryFromWkt(geom_wkt)
    if t_srs is not None and not wgs_srs.IsSame(t_srs):
        ct = osr.CoordinateTransformation(t_srs, wgs_srs)
        geom.Transform(ct)
        geom.AssignSpatialReference(t_srs)
    return geom

def xy2geom(x, y, t_srs=None):
    """Convert x and y point coordinates to geom
    """
    geom_wkt = 'POINT({0} {1})'.format(x, y)
    geom = ogr.CreateGeometryFromWkt(geom_wkt)
    if t_srs is not None and not wgs_srs.IsSame(t_srs):
        ct = osr.CoordinateTransformation(t_srs, wgs_srs)
        geom.Transform(ct)
        geom.AssignSpatialReference(t_srs)
    return geom

def get_dem_mosaic_cmd(fn_list, o, fn_list_txt=None, tr=None, t_srs=None, t_projwin=None, georef_tile_size=None, threads=None, tile=None, stat=None):
    """
    Create ASP dem_mosaic command 
    Useful for spawning many single-threaded mosaicing processes
    """
    cmd = ['dem_mosaic',]
    if o is None:
        o = 'mos'
    cmd.extend(['-o', o])
    if threads is None:
        from pygeotools.lib import iolib
        threads = iolib.cpu_count()
    cmd.extend(['--threads', threads])
    if tr is not None:
        cmd.extend(['--tr', tr])
    if t_srs is not None:
        #cmd.extend(['--t_srs', t_srs.ExportToProj4()])
        cmd.extend(['--t_srs', '"%s"' % t_srs.ExportToProj4()])
        #cmd.extend(['--t_srs', "%s" % t_srs.ExportToProj4()])
    if t_projwin is not None:
        cmd.append('--t_projwin')
        cmd.extend(t_projwin)
        cmd.append('--force-projwin')
    if tile is not None:
        #Not yet implemented
        #cmd.extend(tile_list)
        cmd.append('--tile-index')
        cmd.append(tile)
    if georef_tile_size is not None:
        cmd.extend(['--georef-tile-size', georef_tile_size])
    if stat is not None:
        if stat == 'wmean':
            stat = None
        else:
            cmd.append('--%s' % stat.replace('index',''))
            if stat in ['lastindex', 'firstindex', 'medianindex']:
                #This will write out the index map to -last.tif by default
                cmd.append('--save-index-map')
                #Make sure we don't have ndv that conflicts with 0-based DEM indices
                cmd.extend(['--output-nodata-value','-9999'])
    #else:
    #    cmd.extend(['--save-dem-weight', o+'_weight'])
    #If user provided a file containing list of DEMs to mosaic (useful to avoid long bash command issues)
    if fn_list_txt is not None:
        if os.path.exists(fn_list_txt):
            cmd.append('-l')
            cmd.append(fn_list_txt)
        else: 
            print("Could not find input text file containing list of inputs")
    else:
        cmd.extend(fn_list)
    cmd = [str(i) for i in cmd]
    #print(cmd)
    #return subprocess.call(cmd)
    return cmd

#Formulas for CE90/LE90 here:
#http://www.fgdc.gov/standards/projects/FGDC-standards-projects/accuracy/part3/chapter3
def CE90(x_offset,y_offset):
    RMSE_x = np.sqrt(np.sum(x_offset**2)/x_offset.size) 
    RMSE_y = np.sqrt(np.sum(y_offset**2)/y_offset.size) 
    c95 = 2.4477
    c90 = 2.146
    RMSE_min = min(RMSE_x, RMSE_y)
    RMSE_max = max(RMSE_x, RMSE_y)
    ratio = RMSE_min/RMSE_max
    if ratio > 0.6 and ratio < 1.0:
        out = c90 * 0.5 * (RMSE_x + RMSE_y)
    else:
        out = c90 * np.sqrt(RMSE_x**2 + RMSE_y**2)
    return out

def LE90(z_offset):
    RMSE_z = np.sqrt(np.sum(z_offset**2)/z_offset.size)
    c95 = 1.9600
    c90 = 1.6449
    return c90 * RMSE_z

#Get approximate elevation MSL from USGS API using 10-m NED
#https://nationalmap.gov/epqs/
def get_NED(lon, lat):
    url = 'https://nationalmap.gov/epqs/pqs.php?x=%.8f&y=%.8f&units=Meters&output=json' % (lon, lat)
    r = requests.get(url)
    out = np.nan 
    ned_ndv = -1000000
    if r.status_code == 200:
        out = r.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation']
        out = float(out)
        if out == ned_ndv:
            out = np.nan 
        else:
            print("USGS elevation MSL: %0.2f" % out)
    return out

#USGS API can only handle one point at a time
get_NED_np = np.vectorize(get_NED)

#Get approximate elevation MSL from Open Elevation API (global)
#Note that this can periodically fail, likely throttled - need more robust request
#ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
#Can do multiple points at once:
#https://github.com/Jorl17/open-elevation/blob/master/docs/api.md
def get_OpenElevation(lon, lat):
    import time
    if isinstance(lon, (list, tuple, np.ndarray)):
        #https://api.open-elevation.com/api/v1/lookup\?locations\=10,10\|20,20\|41.161758,-8.583933
        locstr = '|'.join(['%0.8f,%0.8f' % i for i in zip(lat, lon)])
        url = 'https://api.open-elevation.com/api/v1/lookup?locations=%s' % locstr 
        out = np.full_like(lon, np.nan)
    else:
        out = np.nan 
        url = 'https://api.open-elevation.com/api/v1/lookup?locations=%0.8f,%0.8f' % (lat, lon)
    print(url)
    i = 0 
    while(i < 5):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                #out = float(r.json()['results'][0]['elevation'])
                out = [float(i['elevation']) for i in r.json()['results']]
                if len(out) == 1:
                    out = out[0]
                print("Open Elevation MSL: ", out)
                break
        except:
            time.sleep(3)
            i += 1
    return out

#Get geoid offset from NGS
#https://www.ngs.noaa.gov/web_services/geoid.shtml
def get_GeoidOffset(lon, lat):
    #Can specify model, 13 = GEOID12B
    url = 'https://geodesy.noaa.gov/api/geoid/ght?lat=%0.8f&lon=%0.8f' % (lat, lon)
    r = requests.get(url)
    out = np.nan 
    if r.status_code == 200:
        out = float(r.json()['geoidHeight'])
        print("NGS geoid offset: %0.2f" % out)
    return out

#NGS geoid can only handle one point at a time
get_GeoidOffset_np = np.vectorize(get_GeoidOffset)

#Get elevation (height above mean sea level - default for NED and Open Elevation)
def get_MSL(lon, lat):
    out = get_NED_np(lon, lat)
    #Check for missing elevations 
    idx = np.isnan(out)
    if np.any(idx):
        out[idx] = get_OpenElevation(lon[idx], lat[idx])
    return out 

#Get elevation (height above WGS84 ellipsoid)
def get_HAE(lon, lat):
    out = get_MSL(lon, lat) 
    idx = np.isnan(out)
    #If we have any valid values, remove geoid offset
    if np.any(~idx):
        offset = get_GeoidOffset_np(lon[~idx], lat[~idx])
        out[~idx] += offset
    return out 

def test_elev_api(lon, lat):
    lat = np.random.uniform(-90,90,10)
    lon = np.random.uniform(-180,180,10)
    return get_MSL(lon,lat)

#Create a raster heatmap from polygon features (geopandas GeoDataFrame)
#res is output grid cell size in meters
def heatmap(gdf, res=1000, out_fn='heatmap.tif', return_ma=False):
    from pygeotools.lib import iolib
    import subprocess
    gpkg_fn = os.path.splitext(out_fn)[0] + '.gpkg' 
    if not os.path.exists(out_fn):
	#Could probably do this in MEM, or maybe gdal_rasterize is now exposed in API
        gdf.to_file(gpkg_fn, driver='GPKG')
        cmd = ['gdal_rasterize', '-burn', '1', '-tr', str(res), str(res), '-ot', 'UInt32', '-a_nodata', '0', '-add', gpkg_fn, out_fn]
        #Add standard raster output options
        cmd.extend(iolib.gdal_opt_co)
        print(cmd)
        subprocess.call(cmd)
    if return_ma:
        #with rio.Open(out_fn) as src:
        #    out = src.read()
        out = iolib.fn_getma(out_fn)
    else:
        out = out_fn
    return out

#Create a raster heatmap from OGR-readable file (e.g., shp, gpkg) 
def heatmap_fn(fn, res=1000, return_ma=False):
    import geopandas as gpd
    gdf = gpd.read_file(fn)
    out_fn = os.path.splitext(fn)[0]+'_heatmap.tif'
    return heatmap(gdf, res=res, out_fn=out_fn, return_ma=return_ma)
