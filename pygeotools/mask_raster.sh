#! /bin/bash

#David Shean
#dshean@gmail.com

#Utility to burn a shp mask into a raster
#Useful for masking blunders in DEMs

r_fn=$1
shp_fn=$2

echo
r_ndv=$(gdalinfo $r_fn | grep NoData | awk -F'=' '{print $2}')

echo "Copying input raster to new file"
cp -pv $r_fn ${r_fn%.*}_masked.tif
echo "Removing overviews"
gdaladdo -clean ${r_fn%.*}_masked.tif
echo "Burning mask into copy"
echo "gdal_rasterize -burn $r_ndv $shp_fn ${r_fn%.*}_masked.tif"
gdal_rasterize -burn $r_ndv $shp_fn ${r_fn%.*}_masked.tif
