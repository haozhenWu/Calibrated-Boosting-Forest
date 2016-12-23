#!/bin/bash

while IFS='' read -r line;do

echo Running $line
python $1/'muv_xgboost_models.py' $line $2

done <"$3"

python $1/'gather_muv_result.py' $2
