import sys
import pandas as pd
import glob
import re

dir_to_store_result = sys.argv[1]
all_files = glob.glob( dir_to_store_result + 'each_target_cv_result/' + "aid*_cv_result.csv")
base = pd.read_csv(all_files[0])
cv_result = pd.DataFrame(base.iloc[:,1])
cv_result.index = base.iloc[:,0]

for i,file in enumerate(all_files[1:len(all_files)]):
    temp = pd.read_csv(file)
    temp2 = pd.DataFrame(temp.iloc[:,1])
    temp2.index = temp.iloc[:,0]
    cv_result = pd.concat([cv_result,temp2],axis = 1)

cv_result.to_csv(dir_to_store_result + "pcba128_cv_result.csv")

# test result
all_files = glob.glob( dir_to_store_result + 'each_target_test_result/' + "aid*_test_result.csv")
base = pd.read_csv(all_files[0])
aid = re.findall('aid\d{1,10}',all_files[0])[0]
# reconstruct dataframe for easy to view
test_result = pd.DataFrame({aid : list(base.iloc[0:10,2])+list(base.iloc[10:20,1])})
test_result.index = base.iloc[:,0]
test_result = test_result.iloc[[0,5,1,2,3,4,10,15,11,12,13,14]]

for i,file in enumerate(all_files[1:len(all_files)]):
    temp = pd.read_csv(file)
    aid = re.findall('aid\d{1,10}',file)[0]
    temp2 = pd.DataFrame({aid : list(temp.iloc[0:10,2])+list(temp.iloc[10:20,1])})
    temp2.index = temp.iloc[:,0]
    temp2 = temp2.iloc[[0,5,1,2,3,4,10,15,11,12,13,14]]
    test_result = pd.concat([test_result,temp2],axis = 1)

test_result.to_csv(dir_to_store_result + "pcba128_test_result.csv")
