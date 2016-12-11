#!/bin/bash
# input argument: aid_list.csv
while IFS='' read -r line;do

/home/haozhen/anaconda2/bin/python /home/haozhen/Haozhen-data/lightchem/example/demo3.py $line

done <"$1"
