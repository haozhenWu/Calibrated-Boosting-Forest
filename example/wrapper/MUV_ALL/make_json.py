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

    json.loc['target_name'] = sys.argv[2]
    json.loc['label_name_list'] = [sys.argv[2]]

    json.to_json("./config.json")
