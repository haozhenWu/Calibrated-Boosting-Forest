import sys
#sys.path.remove('/usr/lib/python2.7/dist-packages')
sys.path.append("../virtual-screening/virtual_screening")
from lightchem.ensemble import virtualScreening_models
from lightchem.featurize import fingerprint
from lightchem.eval import defined_eval
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

#if __name__ == "__main__"
for fold_num in [3,4,5]:
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
        print test_file
        test_pd = read_merged_data(test_file)

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
        print 'Preparing training data fingerprints'
        # morgan(ecfp) fp
        fp = fingerprint.smile_to_fps(df,smile_colname)
        morgan_fp = fp.Morgan()
        # MACCSkeys fp
        fp = fingerprint.smile_to_fps(df,smile_colname)
        maccs_fp = fp.MACCSkeys()
        comb1 = (morgan_fp,label_name_list)
        comb2 = (maccs_fp,label_name_list)
        training_info = [comb1,comb2]

        print 'Preparing testing data fingerprints'
        df_test = test_pd
        # morgan(ecfp) fp
        fp = fingerprint.smile_to_fps(df_test,smile_colname)
        morgan_fp = fp.Morgan()
        # MACCSkeys fp
        fp = fingerprint.smile_to_fps(df_test,smile_colname)
        maccs_fp = fp.MACCSkeys()
        comb1_test = (morgan_fp,None)# test data does not need label name
        comb2_test = (maccs_fp,None)
        testing_info = [comb1_test,comb2_test]

        print 'Building and selecting best model'
        # Current VsEnsembleModel create test data by default
        model = VsEnsembleModel_keck(training_info,
                                     eval_name,
                                     num_of_fold=4)
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
