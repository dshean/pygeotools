#! /bin/bash

#pygeotools command-line interface examples
#This was prepared for the UW eScience Python seminar on Jan 10, 2017
#It needs to be cleaned up and packaged with some sample data

function pause(){
   read -p "$*"
}

gdal_opt='-co TILED=YES -co COMPRESS=LZW -co BIGTIFF=IF_SAFER'

topdir=/Volumes/SHEAN_SSD2/conus/rainier_stack
cd $topdir

dem=20150911_2051_1020010044385E00_1020010045BCAE00-DEM_8m_trans.tif

-----------
imviewer.py
-----------

imviewer.py -h
pause 'Press [Enter] key to continue...'

imviewer.py $dem

#Select a color map
imviewer.py -cmap inferno $dem 

#Generate shaded relief
gdaldem hillshade $gdal_opt -compute_edges $dem ${dem%.*}_hs_az315.tif

#Color overlay
imviewer.py $dem -overlay ${dem%.*}_hs_az315.tif

~/src/imview/imview/color_hs.py -hs_overlay $dem

#Note coords in lower right corner, click a few points, summit
#3.28084

imviewer.py -clim 1000 1600 $dem -overlay ${dem%.*}_hs_az315.tif

imviewer.py $dem -overlay ${dem%.*}_hs_az315.tif -label 'Elevation (m WGS84)' -scale x -ticks -full

imviewer.py $dem -overlay ${dem%.*}_hs_az315.tif -label 'Elevation (m WGS84)' -scale x -ticks -full -of png

imviewer.py $dem -overlay ${dem%.*}_hs_az315.tif -label 'Elevation (m WGS84)' -scale x -ticks -of png -dpi 300 -outsize 8 8 

----------------------
clip_raster_by_shp.py
----------------------

rgi_all=~/data/rgi50/regions/rgi50_merge.shp
conus_24k=~/Documents/UW/CONUS/24k_poly/24k_selection_32610.shp

clip_raster_by_shp.py -h
pause 'Press [Enter] key to continue...'

clip_raster_by_shp.py -extent raster $dem $conus_24k

imviewer.py ${dem%.*}_shpclip.tif -overlay ${dem%.*}_hs_az315.tif -label 'Elevation (m WGS84)' 

--------------
warptool.py
--------------

dem_list="20140827_1945_10300100360B6C00_103001003712A200-DEM_32m_trans.tif \
20150911_2051_1020010044385E00_1020010045BCAE00-DEM_32m_trans.tif \
20160606_2211_102001004F454400_1020010050E87000-DEM_32m_trans.tif"

dem_list_8m=$(echo $dem_list | sed 's/_32m/_8m/g')

imviewer.py $dem_list

imviewer.py -link $dem_list

parallel "gdaldem hillshade $gdal_opt -compute_edges {} {.}_hs_az315.tif" ::: $dem_list

#Show usage
warptool.py -h
pause 'Press [Enter] key to continue...'

warptool.py -te intersection $dem_list

dem_list_warp=$(echo $dem_list | sed 's/.tif/_warp.tif/g')

imviewer.py -link $dem_list_warp

------------------
compute_dz.py
------------------

#30-m NED from 1970
ned_1970=19700901_ned1_2003_adj_warp.tif
#1-m LiDAR (downsampled to 10m here)
lidar_2008=20080901_rainier_ned13_warp.tif
#Composite 8-m DEM from summer 2014 and 2015
summer_2015=20150818_rainier_summer-tile-0.tif

fn_list="$ned_1970 $lidar_2008 $summer_2015"

warptool.py -tr min -te $summer_2015 -t_srs $summer_2015 $fn_list

fn_list_warp=$(echo $fn_list | sed 's/.tif/_warp.tif/g')

compute_dz.py -h
pause 'Press [Enter] key to continue...'

#compute_dz.py ${ned_1970%.*}_warp.tif ${lidar_2008%.*}_warp.tif
compute_dz.py ${lidar_2008%.*}_warp.tif ${summer_2015%.*}_warp.tif
compute_dz.py ${ned_1970%.*}_warp.tif ${summer_2015%.*}_warp.tif

#parallel "clip_raster_by_shp.py -extent raster {} $conus_24k" ::: *eul.tif

dz=${lidar_2008%.*}_warp_${summer_2015%.*}_warp_dz_eul.tif
dz=${ned_1970%.*}_warp_${summer_2015%.*}_warp_dz_eul.tif

clip_raster_by_shp.py -extent raster $dz $conus_24k

imviewer.py -cmap RdYlBu -clim -20 20 -label 'Elev. Diff (m)' ${dz%.*}_shpclip.tif -overlay ${summer_2015%.*}_hs_az315.tif

#Compute volume and mass statistics
vol_stats.py ${dz%.*}_shpclip.tif

----------------
make_stack.py
----------------

ned_extent=$(get_extent.py $lidar_2008)

dem_list=$(ls *32m_trans.tif)
#dem_list_8m=$(echo $dem_list | sed 's/_32m/_8m/g')

make_stack.py -h
pause 'Press [Enter] key to continue...'

make_stack.py -te intersection $dem_list 

--------------
stack_view.py
--------------

stack_view.py -h
pause 'Press [Enter] key to continue...'

stack_view.py *DEM_32m_trans_stack_33.npz

---------------
dem_mask.py
---------------

