#!/usr/bin/env python

import logging
import glob
import numpy as np
import os

from experimentor import Experimentor
from collections import defaultdict
from cloud.serialization import cloudpickle
from ml.ssl.hpfssl import HPFSSLClassifier
from ml.ssl.laprls import LapRLSClassifier
from ml.ssl.svm import LSVMClassifier

LABELED_DATASET_SUFFIX = "l"
UNLABELED_DATASET_SUFFIX = "u"
TEST_DATASET_SUFFIX = "t"

SAMPLED_DATASETS_NUMBER = 40

class SSLRateDatesetEvaluator(Experimentor):
    """
    Evaluator for SSLRateDataset.
    """

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("SSLRateDatesetEvaluator")

    def __init__(self, base_paths, output_path, classifiers_info):
        """
        Arguments:
        - `base_dataset_path`: path ends with _${label_rate}_${validation_rate}_${}unlabel_rate}_${test_rate}.
        - `classifiers_info`: info for every classifiers
        """
        super(SSLRateDatesetEvaluator, self).__init__()

        # base paths
        self.base_paths = base_paths
        self.base_paths.sort()
        
        # output path
        self.output_path = output_path

        # classifiers info
        self.classifiers_info = classifiers_info

        # results
        self.results = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(
                        )))))

    def run(self, ):
        """
        Run experiments. output structure/format is as follows.
        <dataset, <classifier, <rate_pair, <index, <{labels, preds}, {[], []}>>>>>

        to be saved as pickle.
        """

        for base_path in self.base_paths:
            dataset_paths = glob.glob("%s/*" % base_path)
            dataset_paths.sort()
            for dataset_path in dataset_paths:  # for each dataset
                self._run_with(dataset_path)
                pass

        self._save_results(self.output_path)
            
    def _run_with(self, dataset_path):
        """
        Run experiment with dataset_path
        Arguments:

        -'dataset_path': path for a dataset directory
        """

        # check for the num. of sampled datasets
        sampled_datasets_number = len(glob.glob("%s/*" % dataset_path))
        if sampled_datasets_number != SAMPLED_DATASETS_NUMBER:
            self.logger.info("%s is skipped due to shortage")
            return

        self.logger.info("Run with %s" % (dataset_path))

        indices = self._retrieve_indices(dataset_path)
        for i in indices:  # for each dataset sampled from the same dataset
            self._run_internally_with(dataset_path, i)
            pass
                
    def _run_internally_with(self, dataset_path, index):
        """
        Run experiment with a dataset file index by the argument index.
        Arguments:

        -'dataset_path': path for a dataset
        -'index' dataset index
        """

        # labeled dataset
        lpath = "%s/%d_%s.csv" % (
            dataset_path,
            index,
            LABELED_DATASET_SUFFIX)
        ldata = np.loadtxt(lpath, delimiter=" ")
        y_l = ldata[:, 0]
        X_l = ldata[:, 1:]
        l = X_l.shape[0]
        X_l = np.hstack((X_l, np.reshape(np.ones(l), (l, 1))))

        # unlabeled dataset
        upath = "%s/%d_%s.csv" % (
            dataset_path,
            index,
            UNLABELED_DATASET_SUFFIX)
        udata = np.loadtxt(upath, delimiter=" ")
        X_u = udata[:, 1:]
        u = X_u.shape[0]
        X_u = np.hstack((X_u, np.reshape(np.ones(u), (u, 1))))
        
        # validation dataset
        vpath = "%s/%d_%s.csv" % (
            dataset_path,
            index,
            UNLABELED_DATASET_SUFFIX)
        vdata = np.loadtxt(vpath, delimiter=" ")
        X_v = vdata[:, 1:]
        y_v = vdata[:, 0]
        v = X_v.shape[0]
        X_v = np.hstack((X_v, np.reshape(np.ones(v), (v, 1))))

        # test dataset
        tpath = "%s/%d_%s.csv" % (
            dataset_path,
            index,
            TEST_DATASET_SUFFIX)
        tdata = np.loadtxt(tpath, delimiter=" ")
        y_t = tdata[:, 0]
        X_t = tdata[:, 1:]
        t = X_t.shape[0]
        X_t = np.hstack((X_t, np.reshape(np.ones(t), (t, 1))))

        # keys of results map
        dataset_name = dataset_path.split("/")[-1]
        rates = dataset_path.split("/")[-2].split("_")
        trate = rates[-1]
        vrate = rates[-2]
        urate = rates[-3]
        lrate = rates[-4]
        rate_pair = "%s_%s_%s_%s" % (lrate, urate, vrate, trate)

        for classifier_name in self.classifiers_info:
            param_grid = self.classifiers_info[classifier_name]["param_grid"]
            classifier_ = self.classifiers_info[classifier_name]["classifier"]
            try:
                classifier = classifier_.validate_in_ssl(X_l, y_l, X_u, X_v, y_v,
                                                         param_grid=param_grid)

                preds = classifier.predict_classes(X_t)

                classifier_name = classifier.__class__.__name__
                
                self.results[dataset_name][classifier_name][rate_pair][index]["labels"] = y_t
                self.results[dataset_name][classifier_name][rate_pair][index]["preds"] = preds
                
            except Exception:
                self.logger.error(
                    "classifier: %s failed with %s at %d" % (
                        classifier_name, dataset_name, index
                    ))
                
    def _retrieve_indices(self, dataset_path):
        """
        
        Arguments:
        - `dataset_path`:
        """
        indices = set()
        for path in glob.glob("%s/*" % dataset_path):
            i = path.split("/")[-1].split("_")[0]
            indices.add(int(i))

        indices = list(indices)
        indices.sort()
        return indices

    def _save_results(self, output_path):
        """
        
        Arguments:
        - `output_path`:
        """

        if os.path.exists(output_path):
            os.remove(output_path)
        
        with open(output_path, "w") as fpout:
            cloudpickle.dump(self.results, fpout)
            pass
        pass


# Test
def main():

    # arguemnts for ssl
    base_paths = ["/home/kzk/datasets/uci_csv_ssl_1_50_1_48_subset"]
    output_path = "/home/kzk/experiment/final_paper_2015/test/test001.pkl"
    classifiers_info = {
        "hpfssl": {
            "classifier": HPFSSLClassifier(),
            "param_grid": [{"max_itr": 50, "threshold": 1e-4, "learn_type": "batch"}],
        },
        
        "svm": {
            "classifier": LSVMClassifier(),
            "param_grid": [
                {"C": 1e-3}, {"C": 1e-2}, {"C": 1e-1}, {"C": 1},
                {"C": 1e1}, {"C": 1e2}, {"C": 1e3}],
        },

        "laprls": {
            "classifier": LapRLSClassifier(),
            "param_grid": [
                {"lam": 1e-3, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf"},
                {"lam": 1e-3, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf"},
                {"lam": 1e-3, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf"},
                {"lam": 1e-3, "gamma_s": 1e0, "normalized": True, "kernel": "rbf"},
                {"lam": 1e-3, "gamma_s": 1e1, "normalized": True, "kernel": "rbf"},
                {"lam": 1e-3, "gamma_s": 1e2, "normalized": True, "kernel": "rbf"},
                {"lam": 1e-3, "gamma_s": 1e3, "normalized": True, "kernel": "rbf"},

                {"lam": 1e-2, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf"},
                {"lam": 1e-2, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf"},
                {"lam": 1e-2, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf"},
                {"lam": 1e-2, "gamma_s": 1e0, "normalized": True, "kernel": "rbf"},
                {"lam": 1e-2, "gamma_s": 1e1, "normalized": True, "kernel": "rbf"},
                {"lam": 1e-2, "gamma_s": 1e2, "normalized": True, "kernel": "rbf"},
                {"lam": 1e-2, "gamma_s": 1e3, "normalized": True, "kernel": "rbf"},
 
                {"lam": 1e-1, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf"},
                {"lam": 1e-1, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf"},
                {"lam": 1e-1, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf"},
                {"lam": 1e-1, "gamma_s": 1e0, "normalized": True, "kernel": "rbf"},
                {"lam": 1e-1, "gamma_s": 1e1, "normalized": True, "kernel": "rbf"},
                {"lam": 1e-1, "gamma_s": 1e2, "normalized": True, "kernel": "rbf"},
                {"lam": 1e-1, "gamma_s": 1e3, "normalized": True, "kernel": "rbf"},
 
                {"lam": 1e0, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf"},
                {"lam": 1e0, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf"},
                {"lam": 1e0, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf"},
                {"lam": 1e0, "gamma_s": 1e0, "normalized": True, "kernel": "rbf"},
                {"lam": 1e0, "gamma_s": 1e1, "normalized": True, "kernel": "rbf"},
                {"lam": 1e0, "gamma_s": 1e2, "normalized": True, "kernel": "rbf"},
                {"lam": 1e0, "gamma_s": 1e3, "normalized": True, "kernel": "rbf"},
 
                {"lam": 1e1, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf"},
                {"lam": 1e1, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf"},
                {"lam": 1e1, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf"},
                {"lam": 1e1, "gamma_s": 1e0, "normalized": True, "kernel": "rbf"},
                {"lam": 1e1, "gamma_s": 1e1, "normalized": True, "kernel": "rbf"},
                {"lam": 1e1, "gamma_s": 1e2, "normalized": True, "kernel": "rbf"},
                {"lam": 1e1, "gamma_s": 1e3, "normalized": True, "kernel": "rbf"},
 
                {"lam": 1e2, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf"},
                {"lam": 1e2, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf"},
                {"lam": 1e2, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf"},
                {"lam": 1e2, "gamma_s": 1e0, "normalized": True, "kernel": "rbf"},
                {"lam": 1e2, "gamma_s": 1e1, "normalized": True, "kernel": "rbf"},
                {"lam": 1e2, "gamma_s": 1e2, "normalized": True, "kernel": "rbf"},
                {"lam": 1e2, "gamma_s": 1e3, "normalized": True, "kernel": "rbf"},
 
                {"lam": 1e3, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf"},
                {"lam": 1e3, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf"},
                {"lam": 1e3, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf"},
                {"lam": 1e3, "gamma_s": 1e0, "normalized": True, "kernel": "rbf"},
                {"lam": 1e3, "gamma_s": 1e1, "normalized": True, "kernel": "rbf"},
                {"lam": 1e3, "gamma_s": 1e2, "normalized": True, "kernel": "rbf"},
                {"lam": 1e3, "gamma_s": 1e3, "normalized": True, "kernel": "rbf"},
                
            ],
        },
        
    }

    experimentor = SSLRateDatesetEvaluator(
        base_paths,
        output_path,
        classifiers_info)

    experimentor.run()

if __name__ == '__main__':
    main()
    
