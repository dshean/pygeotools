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

#Initiate with the first input file
if [ "$ext" == "sqlite" ] ; then
    ogr2ogr -overwrite -f 'SQLite' -nln ${2%.*} -dsco SPATIALITE=YES $outfile $2
    shift; shift
    for i in $@ 
    do
        ogr2ogr -update -append -nln ${i%.*} $outfile $i
    done
else
    ogr2ogr -overwrite -f 'ESRI Shapefile' -nln ${outfile%.*} $outfile $2
    shift; shift
    for i in $@ 
    do
        ogr2ogr -update -append $outfile $i -nln ${outfile%.*} 
    done
fi
