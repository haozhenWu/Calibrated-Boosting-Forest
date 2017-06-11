# Generate features using DeepChem.

# Under Tony_lab/working_session/
import sys
#sys.path.remove('/usr/lib/python2.7/dist-packages')
sys.path.append("../../deepchem")
sys.path.append("../../lightchem/example/SmallMolecule/keck/")
import keck_dataset
import pandas as pd
import os
import numpy as np
import time

def array_to_fpString(fp_array, sep = ""):
    '''
    Convert array back to original fingerprint string format
    Reverse function of fpString_to_array.
    '''
    fpString_list = []
    k = 0.0
    for array_1d in fp_array:
        fp = ""
        for index,i in enumerate(list(array_1d)):
            if index == len(list(array_1d))-1:
                fp = fp + str(i)
            else:
                fp = fp + str(i) + sep
        fpString_list.append(fp)
        k += 1
        print k/len(fp_array) # TODO: remove when finish debugging
    return fpString_list

def extract_atom_features(GraphConv_array):
    '''
    Extract atom_features from DeepChem's GraphConv object.
    Each ConvMol is n x 75 dim array, with n being number of atoms(varied),
    75 is the feature generated for each atom.
    Average n array into 1 x 75 array, plus sum, std array
    '''
    feature_list = list()
    for conv in GraphConv_array:
        array_2d = conv.atom_features
        feature = list()
        array_1d = pd.DataFrame(array_2d).mean()
        feature.extend(list(array_1d))
        array_1d = pd.DataFrame(array_2d).sum()
        feature.extend(list(array_1d))
        array_1d = pd.DataFrame(array_2d).std()
        feature.extend(list(array_1d))
        feature_list.append(feature)
    feature_array = np.array(feature_list)
    return feature_array

def main(dataset_name, split, featurizer, root_dir, raw_pcba, reload=True):
    """
    Method to generate splited folds and join with continuous label.

    dataset_name str, name of dataset such as pcba128
    split str, DeepChem's splitting methods, choose from index, random, scaffold
    featurizer str, DeepChem's featurizing methods, choose from ECFP, ConvGraph.
    root_dir str, root directory to store the result.
    raw_pcba dataframe, original datafram that will left join with each fold.
    reload logic, default True.
    """
    fold_name = ['train', 'valid', 'test']
    for fold in fold_name:
        store_dir = os.path.join(root_dir, dataset_name, split, featurizer, fold)
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
    tasks, all_dataset, transformers = keck_dataset.load_keck(
              featurizer=featurizer, split=split, reload=reload)
    train_dataset, valid_dataset, test_dataset = all_dataset
    for i, datasets in enumerate(all_dataset):
        mol_id = datasets.ids
        feature_array = datasets.X
        if featurizer == "GraphConv":
            features = extract_atom_features(feature_array)
            features = array_to_fpString(features, sep="|")
        else:
            features = array_to_fpString(feature_array, sep="|")
        merge = pd.DataFrame({"mol_id":mol_id,featurizer:features})
        store_dir = os.path.join(root_dir, dataset_name, split, featurizer,
                                  fold_name[i], fold_name[i] + ".csv")
        merge.to_csv(store_dir, index = None)

# Start
split_methods = ['index']#'index', 'scaffold', 'random'
featurizer_methods = ['GraphConv']
raw_pcba = None
dataset_name = 'keck_DCfeature'
root_dir = "/home/haozhen/Haozhen-data/Tonys_lab/working_session/dataset"
for split in split_methods:
    for featurizer in featurizer_methods:
        start = time.time()
        main(dataset_name, split, featurizer, root_dir, raw_pcba)
        end = time.time()
        print 'Finish_' + split + "_" + featurizer
        print 'Time took: ' + str(end - start) + "s"
