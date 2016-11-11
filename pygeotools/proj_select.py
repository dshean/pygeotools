#! /usr/bin/env python

#David Shean
#dshean@gmail.com

#Script that automatically selects projection for input geometry
#Goes through user-defined bounding boxes w/ projections (defined in geolib)
#Default output proj is UTM zone containing centroid
#Should allow user to specify input coord srs

import os
import sys
import math

from osgeo import gdal, ogr, osr 

from pygeotools.lib import geolib

def main(argv=None):
    #Input is a filename
    if len(sys.argv[1:]) == 1:
        fn = sys.argv[1]
        ext = os.path.splitext(fn)[1]
        #Accept DigitalGlobe image metadata in xml
        if ext == '.xml':
            from dgtools.lib import dglib
            geom = dglib.xml2geom(fn)
        #Want to be better about handling arbitrary gdal formats here
        elif ext == '.tif' or ext == '.ras':
            ds = gdal.Open(fn)
            geom = geolib.ds_geom(ds, t_srs=geolib.wgs_srs)
            #geom = geolib.get_outline(ds, t_srs=geolib.wgs_srs)
    #Input is lat/lon
    elif len(sys.argv[1:]) == 2:
        y = float(sys.argv[1])
        x = float(sys.argv[2])
        #Force longitude -180 to 180 
        x = (x+180) - math.floor((x+180)/360)*360 - 180
        #Check that latitude is valid 
        if y > 90 or y < -90:
            sys.exit('Invalid latitude: %f' % y)
        geom = geolib.xy2geom(x, y)
    else:
        sys.exit("Usage: %s [lat lon]|[raster.tif]" % os.path.basename(sys.argv[0]))

    #Now determine the srs from geom
    srs = geolib.get_proj(geom)
    #And print to stdout
    print(srs.ExportToProj4().strip())

if __name__ == '__main__':
    main()
