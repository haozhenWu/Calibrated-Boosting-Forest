# This script makes required json file for VS_wrapper.py

# templete.json
# label_name_list
# target_name
import sys
sys.path.remove('/usr/lib/python2.7/dist-packages')
import pandas as pd

if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        info = f.read()
    json = pd.read_json(info)
    name = sys.argv[2]
    json.loc['target_name'] = name
    name_list = []
    name_list.append(name)
    # TODO: seems that pd.to_json does not keep []. Find a way to output []
    json.loc['label_name_list'] = name_list

    json.to_json("./config.json")
