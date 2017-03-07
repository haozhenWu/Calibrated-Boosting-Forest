# This script makes required json file for VS_wrapper.py

import sys
import pandas as pd

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        info = f.read()
    json = pd.read_json(info)
    name = sys.argv[2]
    json.loc['target_name'] = name
    name_list = []
    name_list.append(name + "_binary")
    name_list.append(name + "_logAC50")
    json.loc['label_name_list'] = name_list

    json.to_json("./config.json")
