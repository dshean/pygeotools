#! /usr/bin/env python
"""
David Shean
dshean@gmail.com

Filter an input raster with various approaches

Could use some serious cleanup - lots of obsolute code
"""

import sys
import os
import shutil

import numpy as np

from pygeotools.lib import iolib
from pygeotools.lib import malib
from pygeotools.lib import geolib
from pygeotools.lib import warplib

#Note:
#Original main casts input as float32 so np.nan filling works
#Want function that checks and returns float32 if not
#dem = iolib.ds_getma(dem_ds, 1).astype(np.float32)
#Return original dtype?

#Absolute elevation range filter using an existing low-res DEM
def dz_fltr(dem_fn, refdem_fn, perc=None, abs_dz_lim=(0, 30), smooth=True):
    try:
        open(refdem_fn)
    except IOError:
        sys.exit('Unable to open reference DEM: %s' % refdem_fn)

    dem_ds, refdem_ds = warplib.memwarp_multi_fn([dem_fn, refdem_fn], res='first', extent='first', t_srs='first')
    dem = iolib.ds_getma(dem_ds)
    refdem = iolib.ds_getma(refdem_ds)
    out = dz_fltr_ma(dem, refdem, perc, abs_dz_lim, smooth)
    return out

def dz_fltr_ma(dem, refdem, perc=None, abs_dz_lim=(0,30), smooth=True):
    if smooth:
        refdem = gauss_fltr_astropy(refdem)
        dem = gauss_fltr_astropy(dem)

    dz = refdem - dem

    #This is True for invalid values in DEM, and should be masked
    demmask = np.ma.getmaskarray(dem)

    if perc:
        dz_perc = malib.calcperc(dz, perc)
        print("Applying dz percentile filter (%s%%, %s%%): (%0.1f, %0.1f)" % (perc[0], perc[1], dz_perc[0], dz_perc[1]))
        #This is True for invalid values
        perc_mask = ((dz < dz_perc[0]) | (dz > dz_perc[1])).filled(False)
        demmask = (demmask | perc_mask)

    if abs_dz_lim:
        #This is True for invalid values
        abs_dz_mask = ((np.abs(dz) < abs_dz_lim[0]) | (np.abs(dz) > abs_dz_lim[1])).filled(False)
        if True:
            cutoff = 150
            abs_dz_lim = (0, 80)
            low = (refdem < cutoff).data
            abs_dz_mask[low] = ((np.abs(dz) < abs_dz_lim[0]) | (np.abs(dz) > abs_dz_lim[1])).filled(False)[low]
        demmask = (demmask | abs_dz_mask)

    out = np.ma.array(dem, mask=demmask, fill_value=dem.fill_value)
    return out

#Absolute elevation range filter using an existing low-res DEM
def abs_range_fltr_lowresDEM(dem_fn, refdem_fn, pad=30):
    try:
        open(refdem_fn)
    except IOError:
        sys.exit('Unable to open reference DEM: %s' % refdem_fn)

    dem_ds, refdem_ds = warplib.memwarp_multi_fn([dem_fn, refdem_fn], res='first', extent='first', t_srs='first')
    dem = iolib.ds_getma(dem_ds)
    refdem = iolib.ds_getma(refdem_ds)

    rangelim = (refdem.min(), refdem.max())
    rangelim = (rangelim[0] - pad, rangelim[1] + pad)

    print('Excluding values outside of padded ({0:0.1f} m) lowres DEM range: {1:0.1f} to {2:0.1f} m'.format(pad, *rangelim))
    out = range_fltr(dem, rangelim)
    return out

#Check input range
def range_fltr(dem, rangelim):
    print('Excluding values outside of range: {0:0.1f} to {1:0.1f} m'.format(*rangelim))
    out = np.ma.masked_outside(dem, *rangelim)
    out.set_fill_value(dem.fill_value)
    return out

def perc_fltr(dem, perc=(1.0, 99.0)):
    rangelim = malib.calcperc(dem, perc)
    print('Excluding values outside of percentile ({0:0.2f}, {1:0.2f}) range: {2:0.1f} to {3:0.1f} m'.format(*(perc + rangelim)))
    out = range_fltr(dem, rangelim)
    return out

#These are percentile ranges for 1,2,3 sigma
#15.865, 84.135
#2.275, 97.725
#0.135, 99.865

def threesigma(dem):
    std = dem.std()
    u = dem.mean()
    rangelim = (u - 3*std, u + 3*std)
    out = range_fltr(dem, rangelim)
    return out

#This is used to clean up difference maps before alignment
def mad_fltr(dem, mad_sigma=2):
    med = np.ma.median(dem)
    mad = malib.mad(dem)
    rangelim = (med - mad_sigma * mad, med + mad_sigma * mad)
    print('Excluding values outside of range defined by {0} mad sigma: {1:0.1f} to {2:0.1f} m'.format(mad_sigma, *rangelim))
    out = range_fltr(dem, rangelim)
    return out

#Slope filter
#Would a simple gradient or diff operation here be sufficient to reveal bogus pixels
def slope_fltr(dem_fn, slopelim=(0.1, 40)):
    #Note, Noh and Howat set minimum slope of 20 deg for coregistration purposes
    #perc = (0.01, 99.99)
    #slopelim = malib.calcperc(dem_slope, perc)
    #dem_slope = np.gradient(dem)
    dem_slope = geolib.gdaldem_slope(dem_fn)
    dem = iolib.fn_getma(dem_fn)
    out = np.ma.array(dem, mask=np.ma.masked_outside(dem_slope, *slopelim).mask, keep_mask=True, fill_value=dem.fill_value)
    shutil.rm(os.path.splitext(dem_fn)[0]+'_slope.tif')
    return out

#Smooth with gaussian filter
def gauss_fltr(dem, sigma=1):
    print("Applying gaussian smoothing filter with sigma %s" % sigma)
    #Note, ndimage doesn't properly handle ma - convert to nan
    from scipy.ndimage.filters import gaussian_filter
    dem_filt_gauss = gaussian_filter(dem.filled(np.nan), sigma)
    #Now mask all nans
    #dem = np.ma.array(dem_filt_gauss, mask=dem.mask)
    out = np.ma.fix_invalid(dem_filt_gauss, copy=False, fill_value=dem.fill_value)
    out.set_fill_value(dem.fill_value)
    return out

#http://stackoverflow.com/questions/23832852/by-which-measures-should-i-set-the-size-of-my-gaussian-filter-in-matlab
#width1 = 3; sigma1 = (width1-1) / 6;
#Specify width for smallest feature of interest
#Determine sigma appropriately

#sigma is width of 1 std in pixels (not multiplier)
#scipy and astropy both use cutoff of 4*sigma on either side of kernel - 99.994%
#3*sigma on either side of kernel - 99.7%
#Filter width will be a multiple of 8 times sigma 
#Specify sigma with this in mind

#Alternatively, specify width of feature, compute appropriate sigma for kernel
#specify size, then compute sigma:
#sigma = (size - 1) / 8.

#If size is < the required width for 6-8 sigma
#Need to use different mode to create kernel to ensure 
#mode 'oversample' and 'center' are essentially identical for sigma 1, but very different for sigma 0.3

#The sigma/size calculations below should work for non-integer sigma

#Astropy gaussian filter properly handles convolution with NaN
def gauss_fltr_astropy(dem, size=None, sigma=None, origmask=False, fill_interior=False):
    #import astropy.nddata
    import astropy.convolution
    dem = malib.checkma(dem)
    #Generate 2D gaussian kernel for input sigma and size
    #Default size is 8*sigma in x and y directions
    #kernel = astropy.nddata.make_kernel([size, size], sigma, 'gaussian')
    #Size must be odd
    if size is not None:
        size = int(np.floor(size/2)*2 + 1)
        size = max(size, 3)
    #Truncate the filter at this many standard deviations. Default is 4.0
    truncate = 3.0
    if size is not None and sigma is None:
        sigma = (size - 1) / (2*truncate)
    elif size is None and sigma is not None:
        #Round up to nearest odd int
        size = int(np.ceil((sigma * (2*truncate) + 1)/2)*2 - 1)
    elif size is None and sigma is None:
        #Use default parameters
        sigma = 1
        size = int(np.ceil((sigma * (2*truncate) + 1)/2)*2 - 1)
    size = max(size, 3)
    kernel = astropy.convolution.Gaussian2DKernel(sigma, x_size=size, y_size=size, mode='oversample')

    print("Applying gaussian smoothing filter with size %i and sigma %0.3f (sum %0.3f)" % \
            (size, sigma, kernel.array.sum()))

    #This will fill holes
    #np.nan is float
    #dem_filt_gauss = astropy.nddata.convolve(dem.astype(float).filled(np.nan), kernel, boundary='fill', fill_value=np.nan)
    #dem_filt_gauss = astropy.convolution.convolve(dem.astype(float).filled(np.nan), kernel, boundary='fill', fill_value=np.nan)
    #Added normalization to ensure filtered values are not brightened/darkened if kernelsum != 1
    dem_filt_gauss = astropy.convolution.convolve(dem.astype(float).filled(np.nan), kernel, boundary='fill', fill_value=np.nan, normalize_kernel=True)
    #This will preserve original ndv pixels, applying original mask after filtering
    if origmask:
        print("Applying original mask")
        #Allow filling of interior holes, but use original outer edge
        if fill_interior:
            mask = malib.maskfill(dem)
        else:
            mask = dem.mask
        dem_filt_gauss = np.ma.array(dem_filt_gauss, mask=mask, fill_value=dem.fill_value)
    out = np.ma.fix_invalid(dem_filt_gauss, copy=False, fill_value=dem.fill_value)
    out.set_fill_value(dem.fill_value.astype(dem.dtype))
    return out.astype(dem.dtype)

#This uses a pyramidal downsampling approach for gaussian smoothing - avoids large kernels
#Very fast

def gauss_fltr_pyramid(dem, size=None, full=False, origmask=False):
    dem = malib.checkma(dem)
    levels = int(np.floor(np.log2(size)))
    #print levels
    dim = np.floor(np.array(dem.shape)/float(2**levels) + 1)*(2**levels)
    #print dem.shape
    #print dim
    #Can do something with np.pad here
    #np.pad(a_fp.filled(), 1, mode='constant', constant_values=(a_fp.fill_value,))
    dem2 = np.full(dim, dem.fill_value)
    offset = (dim - np.array(dem.shape))/2.0
    #print offset
    #dem2[0:dem.shape[0],0:dem.shape[1]] = dem.data 
    dem2[offset[0]:dem.shape[0]+offset[0],offset[1]:dem.shape[1]+offset[1]] = dem.data 
    dem2 = np.ma.masked_equal(dem2, dem.fill_value)
    #dem2 = dem
    for n in range(levels):
        print(dem2.shape)
        dim = (np.floor(np.array(dem2.shape)/2.0 + 1)*2).astype(int)
        #dem2 = gauss_fltr_astropy(dem2, size=5, origmask=origmask)
        #dem2 = gauss_fltr_astropy(dem2, size=5)
        dem2 = gauss_fltr_astropy(dem2, size=5)
        #Note: Should use zoom with same bilinear interpolation here for consistency
        #However, this doesn't respect nan
        #dem2 = zoom(dem2, 0.5, order=1, prefilter=False, cval=dem.fill_value)
        dem2 = dem2[::2,::2]
    if full:
        print("Resizing to original input dimensions")
        from scipy.ndimage import zoom
        for n in range(levels):
            print(dem2.shape)
            #Note: order 1 is bilinear
            dem2 = zoom(dem2, 2, order=1, prefilter=False, cval=dem.fill_value)
        #dem2 = zoom(dem2, 2**levels, order=1, prefilter=False, cval=dem2.fill_value)
        print(dem2.shape)
        #This was for power of 2 offset
        #offset = (2**levels)/2
        #print offset
        #dem2 = dem2[offset:dem.shape[0]+offset,offset:dem.shape[1]+offset]
        #Use original offset
        dem2 = dem2[offset[0]:dem.shape[0]+offset[0],offset[1]:dem.shape[1]+offset[1]]
        if origmask:
            print("Applying original mask")
            #Allow filling of interior holes, but use original outer edge
            maskfill = malib.maskfill(dem)
            #dem2 = np.ma.array(dem2, mask=np.ma.getmaskarray(dem))
            dem2 = np.ma.array(dem2, mask=maskfill, fill_value=dem.fill_value)
    return dem2

#Use the OpenCV gaussian filter - still propagates NaN
def gauss_fltr_opencv(dem, size=3, sigma=1):
    import cv2
    dem = malib.checkma(dem)
    dem_cv = cv2.GaussianBlur(dem.filled(np.nan), (size, size), sigma)
    out = np.ma.fix_invalid(dem_cv)
    out.set_fill_value(dem.fill_value)
    return out

def gaussfill(dem, size=3, newmask=None):
    smooth = gauss_fltr_astropy(dem, size=size)
    smooth[~dem.mask] = dem[~dem.mask]
    if newmask is not None:
        smooth = np.ma.array(smooth, mask=newmask)
    return smooth

def highpass(dem, size=None, sigma=None):
    dem_gauss = gauss_fltr_astropy(dem, size=size, sigma=sigma)
    return dem - dem_gauss 

def lowpass(dem, size=None, sigma=None):
    dem_gauss = gauss_fltr_astropy(dem, size=size, sigma=sigma)
    return dem_gauss 

#Note: size2 should be larger than size1
#Freq is 1/size
def bandpass(dem, size1=None, size2=None):
    return highpass(lowpass(dem, size1), size2)

#Smooth and remove noise with median filter 
def median_fltr(dem, fsize=7, origmask=False):
    print("Applying median filter with size %s" % fsize)
    #Note, ndimage doesn't properly handle ma - convert to nan
    from scipy.ndimage.filters import median_filter
    dem_filt_med = median_filter(dem.filled(np.nan), fsize)
    #Now mask all nans
    out = np.ma.fix_invalid(dem_filt_med, copy=False, fill_value=dem.fill_value)
    if origmask:
        out = np.ma.array(out, mask=dem.mask, fill_value=dem.fill_value)
    out.set_fill_value(dem.fill_value)
    return out

#Use the OpenCV gaussian filter - still propagates NaN
def median_fltr_opencv(dem, size=3, iterations=1):
    import cv2
    dem = malib.checkma(dem)
    if size > 5:
        print("Need to implement iteration")
    n = 0
    out = dem
    while n <= iterations:
        dem_cv = cv2.medianBlur(out.astype(np.float32).filled(np.nan), size)
        out = np.ma.fix_invalid(dem_cv)
        out.set_fill_value(dem.fill_value)
        n += 1
    return out

#This correctly handles nan, and is efficient for smaller arrays
def rolling_fltr(dem, f=np.nanmedian, size=3, circular=True):
    dem = malib.checkma(dem)
    newshp = (dem.size, size*size)
    #Force a step size of 1
    t = malib.sliding_window_padded(dem.filled(np.nan), (size, size), (1, 1))
    if circular:
        r = size/2
        c = (r,r)
        y,x = np.ogrid[-c[0]:size-c[0], -c[1]:size-c[1]]
        mask = x*x + y*y <= r*r
        t[:,~mask] = np.nan
    t = t.reshape(newshp)
    out = f(t, axis=1).reshape(dem.shape)
    out = np.ma.fix_invalid(out)
    out.set_fill_value(dem.fill_value)
    return out

#Smooth and remove noise with median filter 
#Note, the skimage median_filter offers a mask option
#This fills in nodata areas with median of valid pixels!
#Effectively an inpainting routine
def median_fltr_skimage(dem, radius=3, erode=1, origmask=False):
    #Note, ndimage doesn't properly handle ma - convert to nan
    dem = malib.checkma(dem)
    dem = dem.astype(np.float64)
    #Mask islands
    if erode > 0:
        print("Eroding islands smaller than %s pixels" % (erode * 2)) 
        dem = malib.mask_islands(dem, iterations=erode)
    print("Applying median filter with radius %s" % radius) 
    #Note: this funcitonality was present in scikit-image 0.9.3
    import skimage.filter
    dem_filt_med = skimage.filter.median_filter(dem, radius, mask=~dem.mask)
    #Starting in version 0.10.0, this is the new filter
    #This is the new filter, but only supports uint8 or unit16
    #import skimage.filters
    #import skimage.morphology 
    #dem_filt_med = skimage.filters.rank.median(dem, disk(radius), mask=~dem.mask)
    #dem_filt_med = skimage.filters.median(dem, skimage.morphology.disk(radius), mask=~dem.mask)
    #Now mask all nans
    #skimage assigns the minimum value as nodata
    #CHECK THIS, seems pretty hacky
    #Also, looks like some valid values are masked at this stage, even though they should be above min
    ndv = np.min(dem_filt_med)
    #ndv = dem_filt_med.min() + 0.001
    out = np.ma.masked_less_equal(dem_filt_med, ndv)
    #Should probably replace the ndv with original ndv
    out.set_fill_value(dem.fill_value)
    if origmask:
        print("Applying original mask")
        #Allow filling of interior holes, but use original outer edge
        #maskfill = malib.maskfill(dem, iterations=radius)
        maskfill = malib.maskfill(dem)
        #dem_filt_gauss = np.ma.array(dem_filt_gauss, mask=dem.mask, fill_value=dem.fill_value)
        out = np.ma.array(out, mask=maskfill, fill_value=dem.fill_value)
    return out

#Note: this has some nasty ringing
#Uniform (mean) filter 
def uniform_fltr(dem, fsize=7):
    print("Applying uniform filter with size %s" % fsize)
    #Note, ndimage doesn't properly handle ma - convert to nan
    from scipy.ndimage.filters import unifiform_filter
    dem_filt_med = uniform_filter(dem.filled(np.nan), fsize)
    #Now mask all nans
    out = np.ma.fix_invalid(dem_filt_med, copy=False, fill_value=dem.fill_value)
    out.set_fill_value(dem.fill_value)
    return out

def butter_low(dt_list, val, lowpass=1.0):
    import scipy.signal
    #val_mask = np.ma.getmaskarray(val)
    #dt is 300 s, 5 min
    dt_diff = np.diff(dt_list)
    if isinstance(dt_diff[0], float):
        dt_diff *= 86400.
    else:
        dt_diff = np.array([dt.total_seconds() for dt in dt_diff])
    dt = malib.fast_median(dt_diff)
    #f is 0.00333 Hz
    #288 samples/day
    fs = 1./dt
    nyq = fs/2.
    order = 3
    f_max = (1./(86400*lowpass)) / nyq
    b, a = scipy.signal.butter(order, f_max, btype='lowpass')
    #b, a = sp.signal.butter(order, (f_min, f_max), btype='bandstop')
    w, h = scipy.signal.freqz(b, a, worN=2000)
    # w_f = (nyq/np.pi)*w
    # w_f_days = 1/w_f/86400.
    #plt.plot(w_f_days, np.abs(h))

    val_f = scipy.signal.filtfilt(b, a, val)
    return val_f

#This is framework for a butterworth bandpass for 1D data
#Needs to be cleaned up and generalized
def butter(dt_list, val, lowpass=1.0):
    import scipy.signal
    import matplotlib.pyplot as plt
    #dt is 300 s, 5 min
    dt_diff = np.diff(dt_list)
    dt_diff = np.array([dt.total_seconds() for dt in dt_diff])
    dt = malib.fast_median(dt_diff)
    #f is 0.00333 Hz
    #288 samples/day
    fs = 1./dt
    nyq = fs/2.

    if False:
        #psd, f = psd(z_msl, fs) 
        sp_f, sp_psd = scipy.signal.periodogram(val, fs, detrend='linear')
        #sp_f, sp_psd = scipy.signal.welch(z_msl, fs, nperseg=2048)
        sp_f_days = 1./sp_f/86400.

        plt.figure()
        plt.plot(sp_f, sp_psd)
        plt.plot(sp_f_days, sp_psd)
        plt.semilogy(sp_f_days, sp_psd)
        plt.xlabel('Frequency')
        plt.ylabel('Power')

    print("Filtering tidal signal")
    #Define bandpass filter
    #f_min = dt/(86400*0.25)
    f_max = (1./(86400*0.1)) / nyq
    f_min = (1./(86400*1.8)) / nyq
    order = 6
    b, a = scipy.signal.butter(order, f_min, btype='highpass')
    #b, a = sp.signal.butter(order, (f_min, f_max), btype='bandpass')
    w, h = scipy.signal.freqz(b, a, worN=2000)
    w_f = (nyq/np.pi)*w
    w_f_days = 1/w_f/86400.
    #plt.figure()
    #plt.plot(w_f_days, np.abs(h))
    val_f_tide = scipy.signal.filtfilt(b, a, val)

    b, a = scipy.signal.butter(order, f_max, btype='lowpass')
    #b, a = sp.signal.butter(order, (f_min, f_max), btype='bandstop')
    w, h = scipy.signal.freqz(b, a, worN=2000)
    w_f = (nyq/np.pi)*w
    w_f_days = 1/w_f/86400.
    #plt.plot(w_f_days, np.abs(h))

    val_f_tide_denoise = scipy.signal.filtfilt(b, a, val_f_tide)
    #val_f_notide = sp.signal.filtfilt(b, a, val)
    val_f_notide = val - val_f_tide 
    # TODO Does this need to return something

#This is a framework for 2D FFT filtering
#It has not be tested or finished - Ben suggests this is a dead end
#See separate utility freq_analysis.py

#Want to fit linear function to artifact line in freq space
#Then mask everything near that line at distances of ~5-200 pixels
#Or whatever the maximum CCD artifact dimension happens to be
#This will depend on scaling - consult CCD map for interval

def freq_filt(bma):
    #Fill ndv with random data
    bf = malib.randomfill(bma)

    import scipy.fftpack
    f = scipy.fftpack.fft2(bf)
    ff = scipy.fftpack.fftshift(f)

    #Ben suggested a Hahn filter here, remove the low frequency, high amplitude information
    #Then do a second fft?

    #np.log(np.abs(ff))

    #perc = malib.calcperc(np.real(ff), perc=(80, 95))
    #malib.iv(numpy.real(ff), clim=perc)

    #See http://scipy-lectures.github.io/advanced/image_processing/

    #Starting at a,b, compute argmax along vertical axis for restricted range
    #Fit line to the x and y argmax values
    #Mask [argmax[y]-1:argmax[y]+1]

    #Create radial mask
    ff_dim = np.array(ff.shape)
    a,b = ff_dim/2
    n = ff_dim.max()
    y,x = np.ogrid[-a:n-a, -b:n-b]
    r1 = 40 
    r2 = 60 
    ff_mask = np.ma.make_mask(ff)
    radial_mask = (r1**2 <= x**2 + y**2) & (x**2 + y**2 < r2**2)
    #Note issues with rounding indices here
    #Hacked in +1 for testing
    ff_mask[:] = radial_mask[a-ff_dim[0]/2:a+ff_dim[0], b-ff_dim[1]/2:b+1+ff_dim[1]/2]

    #Combine radial and line mask

    #Convert mask to 0-1, then feather

    fm = ff * ff_mask

    #Inverse fft
    bf_filt = scipy.fftpack.ifft2(scipy.fftpack.ifftshift(fm))

    #Apply original mask
    bf_filt = np.ma.masked_array(bf_filt, bma.mask)

    #Abs will go from complex to real, but need to preserve sign
    #malib.iv(numpy.real(bf_filt))

#mdenoise call

#Neighborhood filter
#window = [-1, 0, 1]
#slope = np.convolve

#Want to fit plane to neighborhood
#Compute RMS
#If actual value above/below plane exceeds threshold, throw out
