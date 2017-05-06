# Generate fold index based on each file.
import sys
#sys.path.remove('/usr/lib/python2.7/dist-packages')
sys.path.append("../virtual-screening/virtual_screening")
from function import *
import pandas as pd
import numpy as np
import os

def reverse_generate_fold_index(whole_df, file_path, fold_num, join_on):
    """
    Use to regenerate fold index from created individual fold.
    Enable lightchem to use user defined fold index.
    """
    fold_index = whole_df
    for i,path in enumerate(file_path):
        temp_fold = pd.read_csv(file_path[i])
        temp_name = 'fold'+ str(fold_num[i])
        temp_fold.loc[:,temp_name] = 1
        temp_fold = temp_fold.loc[:,['Molecule',temp_name]]
        fold_index = pd.merge(fold_index, temp_fold,on='Molecule',how='left')
        fold_index.loc[:,temp_name] = fold_index.loc[:,temp_name].fillna(0)
    fold_names = [item for item in fold_index.columns if 'fold' in item ]
    fold_index = fold_index.loc[:,fold_names]
    return fold_index

# test
    fold_num = 5
    k = fold_num
    #k = 5
    directory = './dataset/fixed_dataset/fold_{}/'.format(k)
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))

    for j in range(k):
        index = list(set(range(k)) - set([j]))
        train_file = list(np.array(output_file_list)[index])
        test_file = [output_file_list[j]]
        print train_file
        train_pd = read_merged_data(train_file)
        print test_file
        test_pd = read_merged_data(test_file)



a = reverse_generate_fold_index(train_pd, train_file, index, 'Molecule')
