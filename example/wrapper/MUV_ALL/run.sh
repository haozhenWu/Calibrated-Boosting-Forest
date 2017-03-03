#!/bin/bash

while IFS='' read -r line;do

echo Running $line
/home/haozhen/anaconda2/bin/python 'make_json.py' $line $1
/home/haozhen/anaconda2/bin/python ../../../wrapper/VS_wrapper.py ./config.json
