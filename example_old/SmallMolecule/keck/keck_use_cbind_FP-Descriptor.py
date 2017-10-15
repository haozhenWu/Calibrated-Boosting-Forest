# NOTE: This script is depreciated.


import sys
#sys.path.remove('/usr/lib/python2.7/dist-packages')
sys.path.append("../virtual-screening/virtual_screening")
from lightchem.ensemble.virtualScreening_models import *
from lightchem.featurize import fingerprint
from lightchem.eval import defined_eval
from lightchem.utility.util import reverse_generate_fold_index
from function import *
#from data_preparation import *
#from evaluation import *
from openpyxl import Workbook
import pandas as pd
import numpy as np
import operator
from rdkit import Chem
from rdkit.Chem import AllChem
import time
import os

# Need to make sure relative directory has required datasets.
# Need to download prive datasets from Tony's lab.

#docking = pd.read_csv("../vs_data/raw_dockdata/lc123-pria-dockdata.csv", index_col=0)
# Select only docking score
#docking = docking.iloc[:,3:11]
#docking.columns = ['Feature_'+item for item in list(docking.columns)]
#docking = docking.fillna(-99999)

# chemical descriptor
descriptor = pd.read_csv("../vs_data/ChemicalDescriptors_LC1-4_VSKeckXing/lifechem123_cleaned_2017_03_10_desc.csv")
descriptor.index = descriptor.MOLID
# select only descriptor
descriptor = descriptor.iloc[:,2:198]
descriptor.columns = ['Feature_'+item for item in list(descriptor.columns)]
descriptor = descriptor.fillna(-99999)

extra_data = descriptor
extra_data_name = list(descriptor.columns)

for fold_num in [3,4,5]:#4
    start = time.time()
    complete_df = pd.read_csv('./dataset/keck_complete.csv')
    k = fold_num
    #k = 5
    directory = './dataset/fixed_dataset/fold_{}/'.format(k)
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))
    #greedy_multi_splitting(complete_df, k, directory=directory, file_list=file_list)

    dtype_list = {'Molecule': np.str,
                'SMILES':np.str,
                'Fingerprints': np.str,
                'Keck_Pria_AS_Retest': np.int64,
                'Keck_Pria_FP_data': np.int64,
                'Keck_Pria_Continuous': np.float64,
                'Keck_RMI_cdd': np.float64}
    output_file_list = [directory + f_ for f_ in file_list]
    train_auc = []
    test_auc = []
    train_precision = []
    test_precision = []
    test_ef01 = []
    test_ef02 = []
    test_ef0015 = []
    test_ef001 = []

    for j in range(k):
        index = list(set(range(k)) - set([j]))
        train_file = list(np.array(output_file_list)[index])
        test_file = [output_file_list[j]]
        print train_file
        train_pd = read_merged_data(train_file)
        # inner join train_pd with docking based on molecule id.
        train_pd.index = train_pd.Molecule
        train_pd = pd.merge(extra_data,train_pd,how='right',left_index=True,right_index=True)

        print test_file
        test_pd = read_merged_data(test_file)
        test_pd.index = test_pd.Molecule
        test_pd = pd.merge(extra_data,test_pd,how='right',left_index=True,right_index=True)

        my_fold_index = reverse_generate_fold_index(train_pd, train_file,
                                                     index, 'Molecule')

        # Using lightchem

        target_name = 'KECK_Pria'
        smile_colname = 'SMILES'
        label_name_list = ['Keck_Pria_AS_Retest','Keck_Pria_Continuous']
        eval_name = 'ROCAUC'
        dir_to_store = './'

        preDefined_eval = defined_eval.definedEvaluation()
        preDefined_eval.validate_eval_name(eval_name)
        df = train_pd
        #---------- Build Model
        print 'Preparing training data descriptor score'
        comb1 = (df,label_name_list)

        print 'Preparing training data fingerprints'
        # morgan(ecfp) fp + descriptors
        fp = fingerprint.smile_to_fps(df,smile_colname)
        morgan_fp = fp.Morgan()
        morgan_fp_array = util.fpString_to_array(morgan_fp.fingerprint)
        morgan_fp_df = pd.DataFrame(morgan_fp_array)
        morgan_fp_df.columns = ['Feature_'+str(item) for item in morgan_fp_df.columns]
        morgan_fp_df.index = df.index # need to join label column back.
        morgan_fp_df = pd.merge(morgan_fp_df,df.loc[:,label_name_list + extra_data_name],
        how='left',left_index=True,right_index=True)

        comb2 = (morgan_fp_df,label_name_list)

        # MACCSkeys fp + descriptors
        fp = fingerprint.smile_to_fps(df,smile_colname)
        maccs_fp = fp.MACCSkeys()
        maccs_fp_array = util.fpString_to_array(maccs_fp.fingerprint)
        maccs_fp_df = pd.DataFrame(maccs_fp_array)
        maccs_fp_df.columns = ['Feature_'+str(item) for item in maccs_fp_df.columns]
        maccs_fp_df.index = df.index # need to join label column back.
        maccs_fp_df = pd.merge(maccs_fp_df,df.loc[:,label_name_list + extra_data_name],
        how='left',left_index=True,right_index=True)

        comb3 = (maccs_fp_df,label_name_list)

        #training_info = [comb1, comb2, comb3]
        training_info = [comb2, comb3]

        print 'Preparing testing data docking score'
        df_test = test_pd
        comb1_test = (df_test,None)# test data does not need label name
        print 'Preparing testing data fingerprints'
        # morgan(ecfp) fp + descriptors
        fp = fingerprint.smile_to_fps(df_test,smile_colname)
        morgan_fp = fp.Morgan()
        morgan_fp_array = util.fpString_to_array(morgan_fp.fingerprint)
        morgan_fp_df = pd.DataFrame(morgan_fp_array)
        morgan_fp_df.columns = ['Feature_'+str(item) for item in morgan_fp_df.columns]
        morgan_fp_df.index = df_test.index # need to join label column back.
        morgan_fp_df = pd.merge(morgan_fp_df,df_test.loc[:,extra_data_name],
        how='left',left_index=True,right_index=True)

        comb2_test = (morgan_fp_df,None)

        # MACCSkeys fp + descriptors
        fp = fingerprint.smile_to_fps(df_test,smile_colname)
        maccs_fp = fp.MACCSkeys()
        maccs_fp_array = util.fpString_to_array(maccs_fp.fingerprint)
        maccs_fp_df = pd.DataFrame(maccs_fp_array)
        maccs_fp_df.columns = ['Feature_'+str(item) for item in maccs_fp_df.columns]
        maccs_fp_df.index = df_test.index # need to join label column back.
        maccs_fp_df = pd.merge(maccs_fp_df,df_test.loc[:,extra_data_name],
        how='left',left_index=True,right_index=True)

        comb3_test = (maccs_fp_df,None)

        #testing_info = [comb1_test, comb2_test, comb3_test]
        testing_info = [comb2_test, comb3_test]

        print 'Building and selecting best model'
        # Current VsEnsembleModel create test data by default
        model = VsEnsembleModel_keck(training_info,
                                     eval_name,
                                     fold_info=my_fold_index)
        model.train()
        cv_result = model.training_result()
        all_results = model.detail_result()

        #cv_result.to_csv(dir_to_store + target_name + "_result.csv")
        #all_results.to_csv(dir_to_store + target_name + "_result_allModels.csv")

        #----------- Predict testset
        print 'Predict test data'
        y_pred_on_test = model.predict(testing_info)
        y_pred_on_train = model.predict(training_info)
        y_test = np.array(df_test['Keck_Pria_AS_Retest'])
        y_train = np.array(comb1[0]['Keck_Pria_AS_Retest'])
        #---------- Use same evaluation functions
        f = open('./out.txt', 'a')
        print >> f, "########################################"
        print >> f, "Number of Fold: ", k
        print >> f, "Test file: ", j
        print >> f, "Stopping metric: ", eval_name
        print >> f, all_results
        print >> f, cv_result
        print >> f, " "
        print >> f,('train precision: {}'.format(average_precision_score(y_train, y_pred_on_train)))
        print >> f,('train auc: {}'.format(roc_auc_score(y_train, y_pred_on_train)))
        print >> f,('test precision: {}'.format(average_precision_score(y_test, y_pred_on_test)))
        print >> f,('test auc: {}'.format(roc_auc_score(y_test, y_pred_on_test)))

        EF_ratio_list = [0.02, 0.01, 0.0015, 0.001]
        for EF_ratio in EF_ratio_list:
            n_actives, ef, ef_max = enrichment_factor_single(y_test, y_pred_on_test, EF_ratio)
            print >> f,('ratio: {}, EF: {}, EF_max: {}\tactive: {}'.format(EF_ratio, ef, ef_max, n_actives))

        end = time.time()
        print >> f, 'time used: ', end - start
        f.close()

        train_auc.append(roc_auc_score(y_train, y_pred_on_train))
        test_auc.append(roc_auc_score(y_test, y_pred_on_test))
        train_precision.append(average_precision_score(y_train, y_pred_on_train))
        test_precision.append(average_precision_score(y_test, y_pred_on_test))
        n_actives, ef, ef_max = enrichment_factor_single(y_test, y_pred_on_test, 0.01)
        test_ef01.append(ef)
        n_actives, ef, ef_max = enrichment_factor_single(y_test, y_pred_on_test, 0.02)
        test_ef02.append(ef)
        n_actives, ef, ef_max = enrichment_factor_single(y_test, y_pred_on_test, 0.0015)
        test_ef0015.append(ef)
        n_actives, ef, ef_max = enrichment_factor_single(y_test, y_pred_on_test, 0.001)
        test_ef001.append(ef)

    f = open('./summary.txt', 'a')
    print >> f, "########################################"
    print >> f, "Number of Fold: ", k
    print >> f, 'Train ROC AUC mean: ', np.mean(train_auc)
    print >> f, 'Train ROC AUC std', np.std(train_auc)
    print >> f, 'Test ROC AUC mean: ', np.mean(test_auc)
    print >> f, 'Test ROC AUC std', np.std(test_auc)
    print >> f, 'Train Precision mean: ', np.mean(train_precision)
    print >> f, 'Train Precision std', np.std(train_precision)
    print >> f, 'Test Precision mean: ', np.mean(test_precision)
    print >> f, 'Test Precision std', np.std(test_precision)
    print >> f, 'Test ef@0.01 mean: ', np.mean(test_ef01)
    print >> f, 'Test ef@0.01 std', np.std(test_ef01)
    print >> f, 'Test ef@0.02 mean: ', np.mean(test_ef02)
    print >> f, 'Test ef@0.02 std', np.std(test_ef02)
    print >> f, 'Test ef@0.0015 mean: ', np.mean(test_ef0015)
    print >> f, 'Test ef@0.0015 std', np.std(test_ef0015)
    print >> f, 'Test ef@0.001 mean: ', np.mean(test_ef001)
    print >> f, 'Test ef@0.001 std', np.std(test_ef001)
    f.close()
