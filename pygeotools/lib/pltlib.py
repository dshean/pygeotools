#! /usr/bin/env python

#David Shean
#dshean@gmail.com
#11/29/15

#Library of common matplotlib functions
import matplotlib
import matplotlib.pyplot as plt

import numpy as np

#import itertools
#color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
#colors = itertools.cycle(color_list)

def hide_ticks(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def add_cbar(ax, im, label=None, cbar_kwargs={'extend':'both', 'orientation':'vertical', 'shrink':0.7, 'fraction':0.12, 'pad':0.02}):
    #cbar_kwargs['format'] = '%i'
    cbar = plt.colorbar(im, ax=ax, **cbar_kwargs) 
    if label is not None:
        cbar.set_label(label)
    #Set colorbar to be opaque, even if image is transparent
    cbar.set_alpha(1)
    cbar.draw_all()
    return cbar

def minorticks_on(ax, x=True, y=True):
    if x:
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    if y:
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    #ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1000))

def pad_xaxis(ax, abspad=None, percpad=0.02):
    xmin, xmax = ax.get_xlim()
    x_ptp = xmax - xmin
    if abspad is not None:
        pad = abs(abspad)
    else:
        pad = abs(x_ptp) * percpad
    xmin -= pad
    xmax += pad
    ax.set_xlim(xmin, xmax)

def fmt_date_ax(ax, minor=3):
    minorticks_on(ax)
    #months = range(1,13,minor)
    months = [4, 7, 10]
    ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator(months))
    #ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=6))
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    #Rotate and align tick labels
    fig = ax.get_figure()
    fig.autofmt_xdate()
    #ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    #Update interactive display
    date_str = '%Y-%m-%d %H:%M'
    date_fmt = matplotlib.dates.DateFormatter(date_str)
    ax.fmt_xdata = date_fmt
    #trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    #ax.fill_between(x, 0, 1, facecolor='gray', alpha=0.5, transform=trans)
    ax.xaxis.grid(True, 'major')

#This overlays a shapefile
#Should probably isolate to geometry
#Currently needs the ds associated with the imshow ax
#Should probably set this in the ax transform - need to look into this
#def shp_overlay(ax, ds, shp_fn, gt=None, color='g'):
def shp_overlay(ax, ds, shp_fn, gt=None, color='w'):
    import ogr
    import geolib
    #ogr2ogr -f "ESRI Shapefile" output.shp input.shp -clipsrc xmin ymin xmax ymax
    shp_ds = ogr.Open(shp_fn)
    lyr = shp_ds.GetLayer()
    lyr_srs = lyr.GetSpatialRef()
    lyr.ResetReading()
    nfeat = lyr.GetFeatureCount()
    #Note: this is inefficient for large numbers of features
    #Should produce collections of points or lines, then have single plot call
    for n, feat in enumerate(lyr):
        geom = feat.GetGeometryRef()
        geom_type = geom.GetGeometryType()
        #Points
        if geom_type == 1:
            mX, mY, z = geom.GetPoint()
            attr = {'marker':'o', 'markersize':5, 'linestyle':'None'}
        #Line
        elif geom_type == 2:
            l, mX, mY = geolib.line2pts(geom)
            z = 0
            #attr = {'marker':None, 'linestyle':'-', 'linewidth':0.5, 'alpha':0.8}
            attr = {'marker':None, 'linestyle':'-', 'linewidth':1.0, 'alpha':0.8}
            #attr = {'marker':'.', 'markersize':0.5, 'linestyle':'None'}
        #Polygon, placeholder
        #Note: this should be done with the matplotlib patch functionality
        #http://matplotlib.org/users/path_tutorial.html
        elif geom_type == 3:
            print "Polygon support not yet implemented"
            l, mX, mY = geolib.line2pts(geom)
            z = 0
            attr = {'marker':None, 'linestyle':'-', 'facecolor':'w'}

        ds_srs = geolib.get_ds_srs(ds) 
        if gt is None:
            gt = ds.GetGeoTransform()
        if not lyr_srs.IsSame(ds_srs):
            mX, mY, z = geolib.cT_helper(mX, mY, z, lyr_srs, ds_srs)

        #ds_extent = geolib.ds_extent(ds)
        ds_extent = geolib.ds_geom_extent(ds)
      
        mX = np.ma.array(mX)
        mY = np.ma.array(mY)

        mX[mX < ds_extent[0]] = np.ma.masked
        mX[mX > ds_extent[2]] = np.ma.masked
        mY[mY < ds_extent[1]] = np.ma.masked
        mY[mY > ds_extent[3]] = np.ma.masked

        mask = np.ma.getmaskarray(mY) | np.ma.getmaskarray(mX)
        mX = mX[~mask]
        mY = mY[~mask]

        if mX.count() > 0:
            ax.set_autoscale_on(False)
            if geom_type == 1: 
                pX, pY = geolib.mapToPixel(np.array(mX), np.array(mY), gt)
                ax.plot(pX, pY, color=color, **attr)
            else:
                l = np.ma.array(l)
                l = l[~mask]

                lmed = np.ma.median(np.diff(l))
                lbreaks = (np.diff(l) > lmed*2).nonzero()[0]
                if lbreaks.size: 
                    a = 0
                    lbreaks = list(lbreaks)
                    lbreaks.append(l.size)
                    for b in lbreaks:
                        mmX = mX[a:b+1]
                        mmY = mY[a:b+1]
                        a = b+1
                        #import ipdb; ipdb.set_trace()
                        #pX, pY = geolib.mapToPixel(np.array(mX), np.array(mY), gt)
                        pX, pY = geolib.mapToPixel(mmX, mmY, gt)
                        #print n, np.diff(pX).max(), np.diff(pY).max()
                        #ax.plot(pX, pY, color='LimeGreen', **attr)
                        #ax.plot(pX, pY, color='LimeGreen', alpha=0.5, **attr)
                        #ax.plot(pX, pY, color='w', alpha=0.5, **attr)
                        ax.plot(pX, pY, color=color, **attr)
                else:
                    pX, pY = geolib.mapToPixel(np.array(mX), np.array(mY), gt)
                    ax.plot(pX, pY, color=color, **attr)
