import sys
import pandas as pd
import re
import os
import glob
import numpy as np

used_model = []
all_files = glob.glob("./run/*_result.csv")
base = pd.read_csv(all_files[0])
used_model.append(base.columns[1])
m = re.search("run/(.+?)_",all_files[0])
if m:
  name = m.group(1)
result = pd.DataFrame({name:base.iloc[:,1]})
result.index = base.iloc[:,0]
result.index.name = 'type'

for i,file in enumerate(all_files[1:len(all_files)]):
    temp = pd.read_csv(file)
    used_model.append(temp.columns[1])
    m = re.search("run/(.+?)_",file)
    if m:
      name = m.group(1)
    temp2 = pd.DataFrame({name:temp.iloc[:,1]})
    temp2.index = temp.iloc[:,0]
    temp2.index.name = 'type'
    result = pd.concat([result,temp2],axis = 1)

# average of results
result.mean(axis=1)
result.to_csv("./run/pcba128_results.csv")
# used model
used_model


############ detailed result
all_files = glob.glob("./run/*_allModels.csv")
base = pd.read_csv(all_files[0])
m = re.search("run/(.+?)_",all_files[0])
if m:
  name = m.group(1)
max = base.iloc[-2:].weighted_score.max()
pos = np.where(base.weighted_score == max)[0]

max_layer1 = base.iloc[0:-2].weighted_score.max()
pos_layer1 = np.where(base.weighted_score == max_layer1)[0]
diff = []
diff.append(np.float64(base.cv_result[pos]) - np.float64(base.cv_result[pos_layer1]))
diff.append(np.float64(base.test_result[pos]) - np.float64(base.test_result[pos_layer1]))
diff.append(np.float64(base.weighted_score[pos]) - np.float64(base.weighted_score[pos_layer1]))
diff_df = pd.DataFrame({name:diff})

for i,file in enumerate(all_files[1:len(all_files)]):
    temp = pd.read_csv(file)
    m = re.search("run/(.+?)_",file)
    if m:
      name = m.group(1)
    max = temp.iloc[-2:].weighted_score.max()
    pos = np.where(temp.weighted_score == max)[0]
    max_layer1 = temp.iloc[0:-2].weighted_score.max()
    pos_layer1 = np.where(temp.weighted_score == max_layer1)[0]
    diff = []
    diff.append(np.float64(temp.cv_result[pos]) - np.float64(temp.cv_result[pos_layer1]))
    diff.append(np.float64(temp.test_result[pos]) - np.float64(temp.test_result[pos_layer1]))
    diff.append(np.float64(temp.weighted_score[pos]) - np.float64(temp.weighted_score[pos_layer1]))
    diff = pd.DataFrame({name:diff})
    diff_df = pd.concat([diff_df,diff],axis=1)

diff_df.index = ["cv_diff","test_diff","weighted_diff"]
diff_df.mean(axis=1)
diff_df.to_csv("./run/ensemble_analysis.csv")


# find out which target is not run
aid_list = pd.read_csv("./pcba128_TargetName.csv",header=None)
all_files = glob.glob("./run/*_result.csv")
not_exist = []
for aid in aid_list.iloc[:,0]:
    exist = False
    for f in all_files:
        if aid in f:
            exist = True
    if not exist:
        not_exist.append(aid)
