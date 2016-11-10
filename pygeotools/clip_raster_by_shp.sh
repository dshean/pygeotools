#! /bin/bash

#David Shean
#dshean@gmail.com

#Clip raster by polygon(s) in shapefile

#Requires standard gdal command line utilities

#Potentially an issue with shp containing many features - dissolve should work

#Adapted from:
#http://linfiniti.com/2009/09/clipping-rasters-with-gdal-using-polygons/

gdal_opt='-co TILED=YES -co COMPRESS=LZW -co BIGTIFF=IF_SAFER'

if [ "$#" -ne 2 ] && [ "$#" -ne 3 ] ; then
    echo; echo "Usage is $0 raster_fn shape_fn [bbox]"; echo
    exit 1 
fi

#Specify resampling algorithm to use
rs_alg=cubic
#rs_alg=near
#rs_alg=mode

r_fn=$1
s_fn_in=$2
s_fn=$s_fn_in
out_fn=${r_fn%.*}_shpclip.tif

#Get nodata value from raster, if defined
r_ndv=$(gdalinfo $r_fn | grep NoData | awk -F'=' '{print $2}')
#Otherwise, assume 0
if [ -z "$r_ndv" ] ; then
    r_ndv=0
fi

#Extract raster resolution
r_res=$(gdalinfo $r_fn | grep 'Pixel Size' | awk -F '[,()]' '{print $2 " " $3}') 
#Extract raster projection
r_proj=$(gdalsrsinfo -o proj4 $r_fn)
#Extract shapefile projection
s_proj=$(gdalsrsinfo -o proj4 $s_fn)

#Reproject shapefile if necessary
cleanup=false
if [ "$r_proj" != "$s_proj" ] ; then
    echo "Reprojecting input shapefile"
    eval ogr2ogr -overwrite -s_srs $s_proj -t_srs $r_proj ${s_fn%%.*}_reproj_${$}.shp $s_fn
    s_fn=${s_fn%%.*}_reproj_${$}.shp
    cleanup=true
fi

#!!!
#Need to compute common extent here
#!!!
s_extent=($(ogrinfo -so -al $s_fn | grep Extent | sed 's/Extent: //g' | sed 's/(//g' | sed 's/)//g' | sed 's/ - /, /g'))
r_extent=($(gdalinfo $r_fn | egrep 'Upper Left|Lower Right' | awk -F'[(,)]' '{print $2 " " $3}'))
r_extent=(${r_extent[0]} ${r_extent[3]} ${r_extent[2]} ${r_extent[1]})

#Use this to specify desired extent, default preserves input raster extent
extent=${r_extent[@]}
#extent=${s_extent[@]}

echo "raster extent:" ${r_extent[@]}
echo "shp extent:" ${s_extent[@]}
echo "out extent:" $extent

#Clip first, unless input and output extents are identical, then shortcut
r_temp=${r_fn%.*}_bbclip_${$}.tif
if [ "$extent" == "${r_extent[*]}" ] ; then
    ln -sv $r_fn $r_temp
else
    echo "Clipping to $extent"
    echo gdalwarp $gdal_opt -overwrite -dstnodata $r_ndv -te $extent -tr $r_res -of GTiff -r $rs_alg $r_fn $r_temp 
    gdalwarp $gdal_opt -overwrite -dstnodata $r_ndv -te $extent -tr $r_res -of GTiff -r $rs_alg $r_fn $r_temp 
fi

if [ "$3" == "bbox" ]; then
    mv $r_temp $out_fn
else
    #Note, using the -crop_to_cutline parameter here apparently introduces a raster shift!  This is very bad.
    #http://trac.osgeo.org/gdal/ticket/3947
    #gdalwarp -dstnodata 0 -q -cutline $s_fn -crop_to_cutline -of GTiff $r_fn $out_fn
    echo gdalwarp -overwrite -dstnodata $r_ndv $gdal_opt -te $extent -tr $r_res -of GTiff -r $rs_alg -cutline $s_fn $r_temp $out_fn 
    gdalwarp -overwrite -dstnodata $r_ndv $gdal_opt -te $extent -tr $r_res -of GTiff -r $rs_alg -cutline $s_fn $r_temp $out_fn 
    rm $r_temp 
fi

if $cleanup ; then
    rm ${s_fn%.*}.{shx,shp,prj,dbf}
fi
