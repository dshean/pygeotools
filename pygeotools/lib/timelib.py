#! /usr/bin/env python
"""
Time conversion utilities
"""

import os
from datetime import datetime, timedelta
import time

import numpy as np
import matplotlib.dates

#Seconds per year
spy = 86400.*365.25

#lon,lat = geolib.get_center(ds, t_srs=geolib.wgs_srs)
def getTimeZone(lat, lon):
    """Get timezone for a given lat/lon
    """
    #Need to fix for Python 2.x and 3.X support
    import urllib.request, urllib.error, urllib.parse
    import xml.etree.ElementTree as ET
    #http://api.askgeo.com/v1/918/aa8292ec06199d1207ccc15be3180213c984832707f0cbf3d3859db279b4b324/query.xml?points=37.78%2C-122.42%3B40.71%2C-74.01&databases=Point%2CTimeZone%2CAstronomy%2CNaturalEarthCountry%2CUsState2010%2CUsCounty2010%2CUsCountySubdivision2010%2CUsTract2010%2CUsBlockGroup2010%2CUsPlace2010%2CUsZcta2010
    req = "http://api.askgeo.com/v1/918/aa8292ec06199d1207ccc15be3180213c984832707f0cbf3d3859db279b4b324/query.xml?points="+str(lat)+"%2C"+str(lon)+"&databases=TimeZone"
    opener = urllib.request.build_opener()
    f = opener.open(req)
    tree = ET.parse(f)
    root = tree.getroot()
    #Check response
    tzid = None
    if root.attrib['code'] == '0':
        tz = list(root.iter('TimeZone'))[0]
        #shortname = tz.attrib['ShortName']
        tzid = tz.attrib['TimeZoneId']
    return tzid 

def getLocalTime(utc_dt, tz):
    """Return local timezone time
    """
    import pytz
    local_tz = pytz.timezone(tz)
    local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(local_tz)
    return local_dt

def ul_time(utc_dt, lon):
    """Compute local time for input longitude
    """
    #return utc_dt + timedelta(hours=lon / np.pi * 12)
    offset = timedelta(hours=(lon*(24.0/360)))
    return utc_dt + offset

def solarTime(utc_dt, lat, lon):
    """Compute local solar time for given (lat, lon)
    """
    import ephem
    o = ephem.Observer()
    o.date = utc_dt
    o.lat = str(lat)
    o.lon = str(lon)
    sun = ephem.Sun()
    sun.compute(o)
    hour_angle = o.sidereal_time() - sun.ra
    rad = str(ephem.hours(hour_angle + ephem.hours('12:00')).norm)
    t = datetime.strptime(rad, '%H:%M:%S.%f')
    solar_dt = datetime.combine(utc_dt.date(), t.time()) 
    return solar_dt 

def strptime_fuzzy(s):
    """Fuzzy date string parsing

    Note: this returns current date if not found. If only year is provided, will return current month, day
    """
    import dateutil.parser
    dt = dateutil.parser.parse(str(s), fuzzy=True) 
    return dt 

def fn_getdatetime(fn):
    """Extract datetime from input filename
    """
    dt_list = fn_getdatetime_list(fn)
    if dt_list:
        return dt_list[0]
    else:
        return None

#Return datetime object extracted from arbitrary filename
def fn_getdatetime_list(fn):
    """Extract all datetime strings from input filename
    """
    #Want to split last component
    fn = os.path.split(os.path.splitext(fn)[0])[-1]
    import re
    #WV01_12JUN152223255-P1BS_R1C1-102001001B3B9800__WV01_12JUN152224050-P1BS_R1C1-102001001C555C00-DEM_4x.tif
    #Need to parse above with month name 
    #Note: made this more restrictive to avoid false matches:
    #'20130304_1510_1030010020770600_1030010020CEAB00-DEM_4x'
    #This is a problem, b/c 2015/17/00:
    #WV02_20130315_10300100207D5600_1030010020151700
    #This code should be obsolete before 2019 
    #Assume new filenames
    #fn = fn[0:13]
    #Use cascading re find to pull out timestamps
    #Note: Want to be less restrictive here - could have a mix of YYYYMMDD_HHMM, YYYYMMDD and YYYY in filename
    #Should probably search for all possibilities, then prune  
    #NOTE: these don't include seconds in the time
    dstr = None
    dstr = re.findall(r'(?:^|_)(?:19|20)[0-9][0-9](?:0[1-9]|1[012])(?:0[1-9]|[12][0-9]|3[01])[_T](?:0[0-9]|1[0-9]|2[0-3])[0-5][0-9]', fn)
    if not dstr:
        dstr = re.findall(r'(?:^|_)(?:19|20)[0-9][0-9](?:0[1-9]|1[012])(?:0[1-9]|[12][0-9]|3[01])(?:0[0-9]|1[0-9]|2[0-3])[0-5][0-9]', fn)
    if not dstr:
        dstr = re.findall(r'(?:^|_)(?:19|20)[0-9][0-9](?:0[1-9]|1[012])(?:0[1-9]|[12][0-9]|3[01])_', fn)
        #This should pick up dates separated by a dash
        #dstr = re.findall(r'(?:^|_|-)(?:19|20)[0-9][0-9](?:0[1-9]|1[012])(?:0[1-9]|[12][0-9]|3[01])', fn)
    if not dstr:
        dstr = re.findall(r'(?:^|_)(?:19|20)[0-9][0-9]_', fn)
    #This is for USGS archive filenames
    if not dstr:
        dstr = re.findall(r'[0-3][0-9][a-z][a-z][a-z][0-9][0-9]', fn)
    #if not dstr:
    #    dstr = re.findall(r'(?:^|_)(?:19|20)[0-9][0-9]', fn)
    #This is a hack to remove peripheral underscores and dashes
    dstr = [d.lstrip('_').rstrip('_') for d in dstr]
    dstr = [d.lstrip('-').rstrip('-') for d in dstr]
    #This returns an empty list of nothing is found
    out = [strptime_fuzzy(s) for s in dstr]
    #This is USGS archive format
    #out = [datetime.strptime(s, '%d%b%y') for s in dstr][0]
    return out

def get_t_factor(t1, t2):
    """Time difference between two datetimes, expressed as decimal year 
    """
    t_factor = None
    if t1 is not None and t2 is not None and t1 != t2:  
        dt = t2 - t1
        year = timedelta(days=365.25)
        t_factor = abs(dt.total_seconds() / year.total_seconds()) 
    return t_factor

def get_t_factor_fn(fn1, fn2, ds=None):
    t_factor = None
    #Extract timestamps from input filenames
    t1 = fn_getdatetime(fn1)
    t2 = fn_getdatetime(fn2)
    t_factor = get_t_factor(t1,t2)
    #Attempt to load timestamp arrays (for mosaics with variable timestamps)
    t1_fn = os.path.splitext(fn1)[0]+'_ts.tif'
    t2_fn = os.path.splitext(fn2)[0]+'_ts.tif'
    if os.path.exists(t1_fn) and os.path.exists(t2_fn) and ds is not None:
        print("Preparing timestamp arrays")
        from pygeotools.lib import warplib
        t1_ds, t2_ds = warplib.memwarp_multi_fn([t1_fn, t2_fn], extent=ds, res=ds)
        print("Loading timestamps into masked arrays")
        from pygeotools.lib import iolib
        t1 = iolib.ds_getma(t1_ds)
        t2 = iolib.ds_getma(t2_ds)
        #This is a new masked array
        t_factor = (t2 - t1) / 365.25
    return t_factor

def sort_fn_list(fn_list):
    """Sort input filename list by datetime
    """
    dt_list = get_dt_list(fn_list)
    fn_list_sort = [fn for (dt,fn) in sorted(zip(dt_list,fn_list))]
    return fn_list_sort

def get_dt_list(fn_list):
    """Get list of datetime objects, extracted from a filename
    """
    dt_list = np.array([fn_getdatetime(fn) for fn in fn_list])
    return dt_list

#Pad must be timedelta
#pad=timedelta(days=30)
def filter_fn_list(dt, fn_list, pad):
    dt_list = get_dt_list(fn_list)
    #These should be sorted by time
    #This pulls fixed number on either side of dt
    #idx = timelib.get_closest_dt_idx(cdt, v_dt_list)
    #idx = 'idx-2:idx+2' 
    #This pulls from a fixed time interval on either side of dt
    idx = get_closest_dt_padded_idx(dt, dt_list, pad)
    fn_list_sel = fn_list[idx]
    return fn_list_sel

def get_closest_dt_fn(fn, fn_list):
    dt = fn_getdatetime(fn)
    dt_list = np.array([fn_getdatetime(fn) for fn in fn_list])
    idx = get_closest_dt_idx(dt, dt_list) 
    return fn_list[idx]

def get_closest_dt_idx(dt, dt_list):
    """Find index of datetime in dt_list that is closest to input dt
    """
    from pygeotools.lib import malib
    dt_list = malib.checkma(dt_list, fix=False)
    dt_diff = np.abs(dt - dt_list)
    return dt_diff.argmin()

def get_closest_dt_padded_idx(dt, dt_list, pad=timedelta(days=30)):
    from pygeotools.lib import malib
    dt_list = malib.checkma(dt_list, fix=False)
    dt_diff = np.abs(dt - dt_list)
    valid_idx = (dt_diff.data < pad).nonzero()[0]
    return valid_idx

def get_unique_monthyear(dt_list):
    my = [(dt.year,dt.month) for dt in dt_list]
    return np.unique(my)

def get_dt_bounds_monthyear(dt_list):
    my_list = get_unique_monthyear(dt_list)
    out = []
    for my in my_list:
        dt1 = datetime(my[0], my[1], 1)
        dt2 = datetime(my[0], my[1], 1)
    out.append((dt1, dt2))
    return out

def get_unique_years(dt_list):
    years = [dt.year for dt in dt_list]
    return np.unique(years)

def dt_filter_rel_annual_idx(dt_list, min_rel_dt=(1,1), max_rel_dt=(12,31)):
    """Return dictionary containing indices of timestamps that fall within relative month/day bounds of each year
    """
    dt_list = np.array(dt_list)
    years = get_unique_years(dt_list)
    from collections import OrderedDict
    out = OrderedDict() 
    for year in years:
        #If within the same year
        if min_rel_dt[0] < max_rel_dt[1]:
            dt1 = datetime(year, min_rel_dt[0], min_rel_dt[1])
            dt2 = datetime(year, max_rel_dt[0], max_rel_dt[1])
        #Or if our relative values include Jan 1
        else:
            dt1 = datetime(year, min_rel_dt[0], min_rel_dt[1])
            dt2 = datetime(year+1, max_rel_dt[0], max_rel_dt[1])
        idx = np.logical_and((dt_list >= dt1), (dt_list <= dt2))
        if np.any(idx):
            out[year] = idx
    return out

#Use this to get datetime bounds for annual mosaics
def get_dt_bounds(dt_list, min_rel_dt=(1,1), max_rel_dt=(12,31)):
    years = get_unique_years(dt_list)
    out = []
    for year in years:
        #If within the same year
        if min_rel_dt[0] < max_rel_dt[1]:
            dt1 = datetime(year, min_rel_dt[0], min_rel_dt[1])
            dt2 = datetime(year, max_rel_dt[0], max_rel_dt[1])
        else:
            dt1 = datetime(year, min_rel_dt[0], min_rel_dt[1])
            dt2 = datetime(year+1, max_rel_dt[0], max_rel_dt[1])
        if dt2 > dt_list[0] and dt1 < dt_list[-1]:
            out.append((dt1, dt2))
    return out

def get_dt_bounds_fn(list_fn, min_rel_dt=(5,31), max_rel_dt=(6,1)):
    f = open(list_fn, 'r')
    fn_list = []
    for line in f:
        fn_list.append(line)
    fn_list = np.array(fn_list)
    fn_list.sort()
    dt_list = [fn_getdatetime(fn) for fn in fn_list]
    dt_list = np.array(dt_list)
    bounds = get_dt_bounds(dt_list, min_rel_dt, max_rel_dt)
    for b in bounds:
        print(b)
        c_date = center_date(b[0], b[1])
        #c_date = datetime(b[1].year,1,1)
        idx = (dt_list >= b[0]) & (dt_list < b[1])
        #out_fn = os.path.splitext(list_fn)[0]+'_%s_%s-%s_fn_list.txt' % \
        #        (c_date.strftime('%Y%m%d'), b[0].strftime('%Y%m%d'), b[1].strftime('%Y%m%d'))
        out_fn = '%s_%s-%s_fn_list.txt' % \
                (c_date.strftime('%Y%m%d'), b[0].strftime('%Y%m%d'), b[1].strftime('%Y%m%d'))
        out_f = open(out_fn, 'w')
        for fn in fn_list[idx]:
            out_f.write('%s\n' % fn)
        out_f = None

#parallel 'dem_mosaic -l {} --count -o {.}' ::: 2*fn_list.txt

#This checks to see if input dt is between the given relative month/day interval 
def rel_dt_test(dt, min_rel_dt=(1,1), max_rel_dt=(12,31)):
    if dt_check(dt): 
        min_dt = datetime(dt.year, min_rel_dt[0], min_rel_dt[1])
        max_dt = datetime(dt.year, max_rel_dt[0], max_rel_dt[1])
        out = (dt >= min_dt) & (dt <= max_dt)
    else:
        out = False
    return out

def rel_dt_list_idx(dt_list, min_rel_dt=(1,1), max_rel_dt=(12,31)):
    return [rel_dt_test(dt, min_rel_dt, max_rel_dt) for dt in dt_list]

def dt_check(dt):
    return isinstance(dt, datetime)

def seconds2timedelta(s):
    return timedelta(seconds=s)
    
def timedelta2decdays(d):
    return d.total_seconds()/86400.

def timedelta2decyear(d):
    return d.total_seconds()/spy

def timedelta_div(t, d):
    return timedelta(seconds=t.total_seconds()/float(d))

#Return center date between two datetime
#Useful for velocity maps
def center_date(dt1, dt2):
    #return dt1 + (dt2 - dt1)/2
    return mean_date([dt1, dt2])

def mean_date(dt_list):
    """Calcuate mean datetime from datetime list
    """
    dt_list_sort = sorted(dt_list)
    dt_list_sort_rel = [dt - dt_list_sort[0] for dt in dt_list_sort]
    avg_timedelta = sum(dt_list_sort_rel, timedelta())/len(dt_list_sort_rel)
    return dt_list_sort[0] + avg_timedelta

def median_date(dt_list):
    """Calcuate median datetime from datetime list
    """
    #dt_list_sort = sorted(dt_list)
    idx = len(dt_list)/2
    if len(dt_list) % 2 == 0:
        md = mean_date([dt_list[idx-1], dt_list[idx]])
    else:
        md = dt_list[idx]
    return md

def mid_date(dt_list):
    dt1 = min(dt_list)
    dt2 = max(dt_list)
    return dt1 + (dt2 - dt1)/2

def dt_ptp(dt_list):
    dt_list_sort = sorted(dt_list)
    ptp = dt_list_sort[-1] - dt_list_sort[0]
    ndays = ptp.total_seconds()/86400.0
    return ndays 

def uniq_days_dt(dt_list):
    o_list, idx = uniq_days_o(dt_list)
    return o2dt(o_list), idx

def uniq_days_o(dt_list):
    if not isinstance(dt_list[0], float):
        o_list = dt2o(dt_list)
    else:
        o_list = dt_list
    #o_list_sort = np.sort(o_list)
    #o_list_sort_idx = np.argsort(o_list)
    #Round down to nearest day
    o_list_uniq, idx = np.unique(np.floor(o_list), return_index=True)
    return o_list_uniq, idx

def round_dt(dt):
    dt += timedelta(seconds=86400/2.)
    dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return dt

def dt_range(dt1, dt2, interval):
    total_dt = dt2 - dt1
    nint = int((float(total_dt.total_seconds())/interval.total_seconds())+0.999)
    out = dt1 + np.arange(nint) * interval
    return out

def dt_cluster(dt_list, dt_thresh=16.0):
    """Find clusters of similar datetimes within datetime list
    """
    if not isinstance(dt_list[0], float):
        o_list = dt2o(dt_list)
    else:
        o_list = dt_list
    o_list_sort = np.sort(o_list)
    o_list_sort_idx = np.argsort(o_list)
    d = np.diff(o_list_sort)
    #These are indices of breaks
    #Add one so each b starts a cluster
    b = np.nonzero(d > dt_thresh)[0] + 1 
    #Add one to shape so we include final index
    b = np.hstack((0, b, d.shape[0] + 1))
    f_list = []
    for i in range(len(b)-1):
        #Need to subtract 1 here to give cluster bounds
        b_idx = [b[i], b[i+1]-1]
        b_dt = o_list_sort[b_idx]
        #These should be identical if input is already sorted
        b_idx_orig = o_list_sort_idx[b_idx]
        all_idx = np.arange(b_idx[0], b_idx[1])
        all_sort = o_list_sort[all_idx]
        #These should be identical if input is already sorted
        all_idx_orig = o_list_sort_idx[all_idx] 
        dict = {}
        dict['break_indices'] = b_idx_orig 
        dict['break_ts_o'] = b_dt 
        dict['break_ts_dt'] = o2dt(b_dt) 
        dict['all_indices'] = all_idx_orig
        dict['all_ts_o'] = all_sort
        dict['all_ts_dt'] = o2dt(all_sort)
        f_list.append(dict)
    return f_list

#Seconds since epoch
def sinceEpoch(dt):
    return time.mktime(dt.timetuple())

def dt2decyear(dt):
    """Convert datetime to decimal year
    """
    year = dt.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year+1, month=1, day=1)
    yearElapsed = sinceEpoch(dt) - sinceEpoch(startOfThisYear)
    yearDuration = sinceEpoch(startOfNextYear) - sinceEpoch(startOfThisYear)
    fraction = yearElapsed/yearDuration
    return year + fraction 

def decyear2dt(t):
    """Convert decimal year to datetime
    """
    year = int(t)
    rem = t - year 
    base = datetime(year, 1, 1)
    dt = base + timedelta(seconds=(base.replace(year=base.year+1) - base).total_seconds() * rem)
    #This works for np array input
    #year = t.astype(int)
    #rem = t - year 
    #base = np.array([datetime(y, 1, 1) for y in year])
    return dt

#Better to use astro libe or jdcal for julian to gregorian conversions 
#Source: http://code.activestate.com/recipes/117215/
def dt2jd(dt):
    """Convert datetime to julian date
    """
    a = (14 - dt.month)//12
    y = dt.year + 4800 - a
    m = dt.month + 12*a - 3
    return dt.day + ((153*m + 2)//5) + 365*y + y//4 - y//100 + y//400 - 32045

def jd2dt(jd):
    """Convert julian date to datetime
    """
    n = int(round(float(jd)))
    a = n + 32044
    b = (4*a + 3)//146097
    c = a - (146097*b)//4
    d = (4*c + 3)//1461
    e = c - (1461*d)//4
    m = (5*e + 2)//153
    day = e + 1 - (153*m + 2)//5
    month = m + 3 - 12*(m//10)
    year = 100*b + d - 4800 + m/10
    
    tfrac = 0.5 + float(jd) - n
    tfrac_s = 86400.0 * tfrac 
    minfrac, hours = np.modf(tfrac_s / 3600.)
    secfrac, minutes = np.modf(minfrac * 60.)
    microsec, seconds = np.modf(secfrac * 60.)

    return datetime(year, month, day, int(hours), int(minutes), int(seconds), int(microsec*1E6))

#This has not been tested 
def gps2dt(gps_week, gps_ms):
    """Convert GPS week and ms to a datetime
    """
    gps_epoch = datetime(1980,1,6,0,0,0)
    gps_week_s = timedelta(seconds=gps_week*7*24*60*60)
    gps_ms_s = timedelta(milliseconds=gps_ms) 
    return gps_epoch + gps_week_s + gps_ms_s

def mat2dt(o):
    """Convert Matlab ordinal to Python datetime

    Need to account for AD 0 and AD 1 discrepancy between the two: http://sociograph.blogspot.com/2011/04/how-to-avoid-gotcha-when-converting.html
    
    python_datetime = datetime.fromordinal(int(o)) + timedelta(days=o%1) - timedelta(days = 366)
    """
    return o2dt(o) - timedelta(days=366)

#Python datetime to matlab ordinal
def dt2mat(dt):
    """Convert Python datetime to Matlab ordinal
    """
    return dt2o(dt + timedelta(days=366))

#note
#If ma, need to set fill value to 0 when converting to ordinal

def dt2o(dt):
    """Convert datetime to Python ordinal
    """
    #return datetime.toordinal(dt)
    #This works for arrays of dt
    #return np.array(matplotlib.dates.date2num(dt))
    return matplotlib.dates.date2num(dt)

#Need to split ordinal into integer and decimal parts
def o2dt(o):
    """Convert Python ordinal to datetime
    """
    #omod = np.modf(o)
    #return datetime.fromordinal(int(omod[1])) + timedelta(days=omod[0])
    #Note: num2date returns dt or list of dt
    #This funciton should always return a list
    #return np.array(matplotlib.dates.num2date(o))
    return matplotlib.dates.num2date(o)

#Return integer DOY (julian)
def dt2j(dt):
    """Convert datetime to integer DOY (Julian)
    """
    #return int(dt.strftime('%j'))
    return int(dt.timetuple().tm_yday)

#Year and day of year to datetime
#Add comment to http://stackoverflow.com/questions/2427555/python-question-year-and-day-of-year-to-date
#ordinal allows for days>365 and decimal days
def j2dt(yr, j):
    """Convert year + integer DOY (Julian) to datetime
    """
    return o2dt(dt2o(datetime(int(yr), 1, 1))+j-1)
    #The solution below can't deal with jd>365
    #jmod = np.modf(j)
    #return datetime.strptime(str(yr)+str(int(jmod[1])), '%Y%j') + timedelta(days=jmod[0])

def print_dt(dt):
    return dt.strftime('%Y%m%d_%H%M')

#Generate a new files with time ordinal written to every pixel
#If dt_ref is provided, return time interval in decimal days
#Should add functionality to do relative doy
def gen_ts_fn(fn, dt_ref=None, ma=False):
    from osgeo import gdal
    from pygeotools.lib import iolib
    print("Generating timestamp for: %s" % fn)
    fn_ts = os.path.splitext(fn)[0]+'_ts.tif' 
    if not os.path.exists(fn_ts) or dt_ref is not None:
        ds = gdal.Open(fn)
        #Should be ok with float ordinals here
        a = iolib.ds_getma(ds)
        ts = fn_getdatetime(fn)
        #Want to check that dt_ref is valid datetime object
        if dt_ref is not None:
            t = ts - dt_ref 
            t = t.total_seconds()/86400.
            fn_ts = os.path.splitext(fn)[0]+'_ts_rel.tif'
        else:
            t = dt2o(ts)
        a[~np.ma.getmaskarray(a)] = t
        #Probably want to be careful about ndv here - could be 0 for rel
        #ndv = 1E20
        ndv = -9999.0 
        a.set_fill_value(ndv)
        iolib.writeGTiff(a, fn_ts, ds) 
    if ma:
        return a
    else:
        return fn_ts

#Convert date listed in .meta to timestamp
#'Central Julian Date (CE) for Pair'
def tsx_cdate(t1):
    #print jd2dt(t1).strftime('%Y%m%d')
    #print jd2dt(t1).strftime('%Y%m%d_%H%M')
    return jd2dt(t1)

def tsx_cdate_print(t1):
    print(tsx_cdate(t1).strftime('%Y%m%d_%H%M'))

#Matlab to python o = matlab - 366
#Launch was June 15, 2007 at 0214 UTC
#Repeat is 11 days, period is 95 minutes

#Orbit 8992, 2454870.406, Jan:27:2009, Feb:18:2009, 09:43:59
#Orbit 40703, 2456953.658, Oct:17:2014, Oct:28:2014, 03:47:50

#This isn't perfect
def tsx_orbit2dt(orbits):
    refdate = 733936.4245-365
    reforbit = 5516
    orbdates = refdate + (orbits-reforbit)*11./167. + 5.5
    return mat2dt(orbdates)

#np vectorize form of functions
#Should clean these up - most can probably be handled directly using np arrays and np.datetime64
np_mat2dt = np.vectorize(mat2dt)
np_dt2mat = np.vectorize(dt2mat)
np_dt2o = np.vectorize(dt2o)
np_o2dt = np.vectorize(o2dt)
np_j2dt = np.vectorize(j2dt)
np_dt2j = np.vectorize(dt2j)
np_decyear2dt = np.vectorize(decyear2dt)
np_dt2decyear = np.vectorize(dt2decyear)
np_utc2dt = np.vectorize(datetime.utcfromtimestamp)
np_print_dt = np.vectorize(print_dt)
