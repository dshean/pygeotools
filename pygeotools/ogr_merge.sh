#! /bin/bash

#Utility to merge multiple shapefiles

#To do:
#Better about handling command line arguments
#Check that all shapefiles have same projeciton/cs?
#layer names should be basename

outfile=$1
ext=${outfile#*.}

if [ -f "$outfile" ]; then
    while [ true ]
    do
        echo "Output file $outfile already exists. Overwrite? (y/n)"
        read -e ans
   	if [ "$ans" == "y" ]; then
	    echo "Overwriting $outfile"
	    break
        elif [ "$ans" == "n" ]; then
	    echo "You chose not to override. Exiting"
	    exit 1
	else
	    echo "Response not understood"
        fi
    done
fi

#Should pull out proj from first input file and use for all
#Can run into issues with append when inputs have diff UTM zones, for example
proj='EPSG:4326'

if [ "$ext" == "sqlite" ] ; then
    ogr2ogr -t_srs $proj -overwrite -f 'SQLite' -nln ${2%.*} -dsco SPATIALITE=YES $outfile $2
    shift; shift
    for i in $@ 
    do
        ogr2ogr -t_srs $proj -update -append -nln ${i%.*} $outfile $i
    done
else
    ogr2ogr -t_srs $proj -overwrite -f 'ESRI Shapefile' -nln ${outfile%.*} $outfile $2
    shift; shift
    for i in $@ 
    do
        ogr2ogr -t_srs $proj -update -append $outfile $i -nln ${outfile%.*} 
    done
fi
