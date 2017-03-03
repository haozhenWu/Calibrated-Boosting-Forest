#!/bin/bash

while IFS='' read -r line;do

echo Running $line
/home/haozhen/anaconda2/bin/python 'make_json.py' ./config.json $line
/home/haozhen/anaconda2/bin/python ../../../wrapper/VS_wrapper.py ./config.json

done <"$1"   
