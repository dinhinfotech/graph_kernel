import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
greatparentdir = os.path.dirname(parentdir)
greatgreatparentdir = os.path.dirname(greatparentdir)
sys.path.insert(0,parentdir)
sys.path.insert(0,greatparentdir)
sys.path.insert(0,greatgreatparentdir)
import sys
#from skgraph.datasets import load_graph_datasets
import numpy as np
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
import random
from sklearn import svm
import util
seed = 7
np.random.seed(seed)

#label_path = "/media/dinh/DATA/Test/graph_labels/CPDB"
label_path = "/home/dinh/fast-disk/Weisfeiler_GK/scikit-learn-graph/scripts/graph_labels/CPDB"
#kernel_folder_path = "/media/dinh/DATA/Test/kernels/cpdb_reduce/"
kernel_folder_path = "/home/dinh/fast-disk/Weisfeiler_GK/scikit-learn-graph/scripts/kernels/cpdb/"
#save_file = "/media/dinh/DATA/Test/results/cpdb_nested_3"
save_file = "/home/dinh/fast-disk/Weisfeiler_GK/scikit-learn-graph/scripts/results_cv/cpdb"

pre_labels = [int(label) for label in util.load_list_from_file(label_path)]
"""Loading kernels"""
pre_list_kernels = []
kernel_names = util.list_files_in_folder(kernel_folder_path)
for kernel_name in kernel_names:
    pre_list_kernels.append(np.fromfile(kernel_folder_path + kernel_name).reshape(len(pre_labels), len(pre_labels)))
svm_paras = [1e-0, 1e+1, 1e+2, 1e+3, 1e+4]

n_folds = 10

"""Model selection"""
def model_selection(list_kernels=None, svm_paras=None, list_labels=None, n_folds=None):

    kf = cross_validation.KFold(len(list_labels), n_folds = n_folds)

    list_train_fold_index = []
    list_test_fold_index = []

    for train_index, test_index in kf:
        list_train_fold_index.append(train_index)
        list_test_fold_index.append(test_index)

    dict_paras = {}
    for kernel_idx in range(len(list_kernels)):
        for svm_idx in range(len(svm_paras)):
            dict_paras[(kernel_idx,svm_idx)] = 0

    for fold_idx in range(n_folds):
        train_list_index = list_train_fold_index[fold_idx]
        test_list_index = list_test_fold_index[fold_idx]

        labels_train =  [labels[idx] for idx in train_list_index]
        labels_test = [labels[idx] for idx in test_list_index]

        for kernel_idx in range(len(list_kernels)):
            M_tr = util.extract_submatrix(train_list_index, train_list_index, list_kernels[kernel_idx])
            M_te = util.extract_submatrix(test_list_index, train_list_index, list_kernels[kernel_idx])

            for svm_idx, svm_para in enumerate(svm_paras):
                clf = svm.SVC(C = svm_para, kernel='precomputed')
                clf.fit(M_tr, labels_train)

                y_predict = clf.predict(M_te)
                acc = accuracy_score(labels_test, y_predict)
                dict_paras[(kernel_idx, svm_idx)]+= acc

    return max(dict_paras, key=dict_paras.get)

"""Cross validation"""

all_avg_accs = []
f = open(save_file, 'w')
for ran_idx in range(10):
    print "Random ", ran_idx
    
    shuffle_indices = range(len(pre_labels))
    random.shuffle(shuffle_indices)
    
    labels = [pre_labels[idx] for idx in shuffle_indices]
    list_kernels = []
    for kernel in  pre_list_kernels:
        list_kernels.append(util.extract_submatrix(shuffle_indices, shuffle_indices, kernel))
        
    kf = cross_validation.KFold(len(labels), n_folds = n_folds)
    
    list_train_fold_index = []
    list_test_fold_index = []
    
    for train_index, test_index in kf:
        list_train_fold_index.append(train_index)
        list_test_fold_index.append(test_index)
    list_accs = []
    
    for fold_idx in range(n_folds):
        train_list_index = list_train_fold_index[fold_idx]
        test_list_index = list_test_fold_index[fold_idx]
    
        labels_train =  [labels[idx] for idx in train_list_index]
        labels_test = [labels[idx] for idx in test_list_index]
    
        list_kernels_validation = []
        for kernel in list_kernels:
            list_kernels_validation.append(util.extract_submatrix(train_list_index, train_list_index, kernel))
    
        kernel_idx, svm_idx = model_selection(list_kernels=list_kernels_validation, svm_paras=svm_paras, list_labels=labels_train, n_folds=n_folds)
    
        print "\t", kernel_idx
        print "\t", svm_idx
        print "\t", "============"
    
        M_tr = util.extract_submatrix(train_list_index, train_list_index, list_kernels[kernel_idx])
        M_te = util.extract_submatrix(test_list_index, train_list_index, list_kernels[kernel_idx])
    
        clf = svm.SVC(C = svm_paras[svm_idx], kernel='precomputed')
        clf.fit(M_tr, labels_train)
    
        y_predict = clf.predict(M_te)
        acc = accuracy_score(labels_test, y_predict)
    
        list_accs.append(acc)
    all_avg_accs.append(np.average(list_accs))
    f.write("Random: " + str(ran_idx) + "\n")
    f.writelines([str(val) + "\n" for val in list_accs])
    f.write(str(np.average(list_accs)) + "\t" + str(np.std(list_accs)) + "\n")
    f.write("=================\n")
    
f.writelines([str(val) + "\n" for val in all_avg_accs])
f.write(str(np.average(all_avg_accs)) + "\t" + str(np.std(all_avg_accs)) + "\n")
f.close()

print "Done"