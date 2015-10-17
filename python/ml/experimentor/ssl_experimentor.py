#!/usr/bin/env python

import logging
import glob
import numpy as np
import os
import gc
import time

from ml.util import Utility
from experimentor import Experimentor
from collections import defaultdict
from cloud.serialization import cloudpickle
from ml.ssl.hpfssl import HPFSSLClassifier
from ml.ssl.laprls import LapRLSClassifier
from ml.ssl.svm import LSVMClassifier
from ml.service.worker import WorkerPool
from ml.service.worker import Task
from multiprocessing import Queue

LABELED_DATASET_SUFFIX = "l"
UNLABELED_DATASET_SUFFIX = "u"
TEST_DATASET_SUFFIX = "t"

SAMPLED_DATASETS_NUMBER = 40

class SSLRateDatesetEvaluator(Experimentor):
    """
    Evaluator for SSLRateDataset.
    """

    FORMAT = '%(asctime)s::%(levelname)s::%(name)s::%(funcName)s::%(message)s'
    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG)
    logger = logging.getLogger("SSLRateDatesetEvaluator")

    def __init__(self, base_paths, output_path, classifiers_info, blocked_datasets):
        """
        Arguments:
        - `base_dataset_path`: path ends with _${label_rate}_${validation_rate}_${}unlabel_rate}_${test_rate}.
        - `classifiers_info`: info for every classifiers
        - `blocked_datasets`: datasets list to be filtered out
        """
        super(SSLRateDatesetEvaluator, self).__init__()

        # base paths
        self.base_paths = base_paths
        self.base_paths.sort()
        
        # output path
        self.output_path = output_path

        # classifiers info
        self.classifiers_info = classifiers_info

        # blocked_datasets
        self.blocked_datasets = blocked_datasets

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
            for dataset_path in dataset_paths:   # for each dataset
                dataset_name = dataset_path.split("/")[-1]
                if dataset_name in self.blocked_datasets:   # filtered out
                    self.logger.info("%s is filtered out" % dataset_name)
                    continue
                    pass
                self._run_with(dataset_path)
                gc.collect()
            pass
            gc.collect()

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

        # create workerpool
        indices = self._retrieve_indices(dataset_path)
        worker_pool = WorkerPool()

        # put tasks
        classifiers_info = self.classifiers_info
        for i in indices:
            task = ExperimentWithOneDatasetTask(classifiers_info,
                                                dataset_path, i, task_name="task %d" % (i))
            worker_pool.put(task)
            self.logger.info("put task %d" % i)
            pass

        # get tasks
        for i in indices:
            task_result = worker_pool.get()
            self.logger.info("get task %d" % i)
            self._set_task_result(task_result)

        # terminate worker
        worker_pool.terminate()

    def _set_task_result(self, task_result):
        """
        """

        if task_result is None:
            return
        
        for k, v in task_result.items():
            dataset_name = k.dataset_name
            classifier_name = k.classifier_name
            rate_pair = k.rate_pair
            index = k.index
            self.results[dataset_name][classifier_name][rate_pair][index]["labels"] = v["labels"]
            self.results[dataset_name][classifier_name][rate_pair][index]["preds"] = v["preds"]
            pass
        
        pass
        
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

    def _run_internally_with(self, dataset_path, index):
        """
        Run experiment with a dataset file index by the argument index.
        Arguments:

        -'dataset_path': path for a dataset
        -'index' dataset index
        """
        self.logger.info("processing %s at %d" % (dataset_path, index))

        try:
            
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

        except Exception:
            self.logger.info("dataset_path, %s at %d was not loaded."
                             % (dataset_path, index))
            return
        
        for classifier_name in self.classifiers_info:
            self.logger.info("processing %s at %d with %s" % (
                dataset_path,
                index,
                classifier_name
            ))
            param_grid = self.classifiers_info[classifier_name]["param_grid"]
            classifier_klass = self.classifiers_info[classifier_name]["classifier"]
            classifier_ = Utility.get_class(classifier_klass)
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

# Task
class ExperimentWithOneDatasetTask(Task):
    """
    Task for expeirment with one dataset.
    """
    FORMAT = '%(asctime)s::%(levelname)s::%(name)s::%(funcName)s::%(message)s'
    logging.basicConfig(
        format=FORMAT,
        level=logging.DEBUG)
    logger = logging.getLogger("SSLRateDatesetEvaluator")

    
    def __init__(self, classifiers_info, dataset_path, index, task_name=""):
        """
        Arguments:
        - `task_name`:
        - `classifiers_info`:
        - `dataset_path`: dataset path specifying to datast directory
        - `index`: data index with which experiment will be run.
        """
        super(ExperimentWithOneDatasetTask, self).__init__(task_name)

        self.classifiers_info = classifiers_info
        self.dataset_path = dataset_path
        self.index = index
        self.task_name = task_name

    def run(self, ):
        """
        """

        classifiers_info = self.classifiers_info
        dataset_path = self.dataset_path
        index = self.index
        
        self.logger.info("processing %s at %d" % (dataset_path, index))
        
        try:
            
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

        except Exception:
            self.logger.info("dataset_path, %s at %d was not loaded."
                             % (dataset_path, index))
            return None

        task_result = {}
        for classifier_name in classifiers_info:
            #self.logger.info("processing %s at %d with %s" % (
            #    dataset_path,
            #    index,
            #    classifier_name
            #))
            param_grid = classifiers_info[classifier_name]["param_grid"]
            classifier_class = classifiers_info[classifier_name]["classifier"]
            classifier_instance = Utility.get_instance(classifier_class)()

            try:
                classifier = classifier_instance.validate_in_ssl(X_l, y_l, X_u, X_v, y_v,
                                                                 param_grid=param_grid)
                preds = classifier.predict_classes(X_t)
                classifier_name = classifier.__class__.__name__
                
                key = ExperimentWithOneDatasetTaskResultKey(dataset_name,
                                                            classifier_name, rate_pair, index)
                task_result[key] = {}
                task_result[key]["labels"] = y_t
                task_result[key]["preds"] = preds

            except Exception as e:
                self.logger.error(e)
                self.logger.error(
                    "classifier: %s failed with %s at %d" % (
                        classifier_name, dataset_name, index
                    ))
                pass

        return task_result

# Task Result
class ExperimentWithOneDatasetTaskResultKey(object):
    """
    """
    
    def __init__(self, dataset_name, classifier_name, rate_pair, index):
        """
        """
        self.dataset_name = dataset_name
        self.classifier_name = classifier_name
        self.rate_pair = rate_pair
        self.index = index

        pass

# Test
def main():

    # arguemnts for ssl
    base_paths = ["/home/kzk/datasets/uci_csv_ssl_1_50_1_48_subset_02"]
    output_path = "/home/kzk/experiment/final_paper_2015/results_test/test001.pkl"
    classifiers_info = {
        "hpfssl": {
            "classifier": "ml.ssl.hpfssl.HPFSSLClassifier",
            "param_grid": [{"max_itr": 50, "threshold": 1e-4, "learn_type": "batch"}],
        },
        
        "svm": {
            "classifier": "ml.ssl.svm.LSVMClassifier",
            "param_grid": [
                {"C": 1e-3}, {"C": 1e-2}, {"C": 1e-1}, {"C": 1},
                {"C": 1e1}, {"C": 1e2}, {"C": 1e3}],
        },

        #"laprls": {
        #    "classifier": "ml.ssl.laprls.LapRLSClassifier",
        #    "param_grid": [  # too slow
        #        {"lam": 1e-0, "gamma_s": 1e-0, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e-3, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e-3, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e-3, "gamma_s": 1e0, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e-3, "gamma_s": 1e1, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e-3, "gamma_s": 1e2, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e-3, "gamma_s": 1e3, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #         
        #        {"lam": 1e-2, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e-2, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e-2, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e-2, "gamma_s": 1e0, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e-2, "gamma_s": 1e1, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e-2, "gamma_s": 1e2, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e-2, "gamma_s": 1e3, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #         
        #        {"lam": 1e-1, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e-1, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e-1, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e-1, "gamma_s": 1e0, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e-1, "gamma_s": 1e1, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e-1, "gamma_s": 1e2, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e-1, "gamma_s": 1e3, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #         
        #        {"lam": 1e0, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e0, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e0, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e0, "gamma_s": 1e0, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e0, "gamma_s": 1e1, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e0, "gamma_s": 1e2, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e0, "gamma_s": 1e3, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #         
        #        {"lam": 1e1, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e1, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e1, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e1, "gamma_s": 1e0, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e1, "gamma_s": 1e1, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e1, "gamma_s": 1e2, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e1, "gamma_s": 1e3, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #         
        #        {"lam": 1e2, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e2, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e2, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e2, "gamma_s": 1e0, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e2, "gamma_s": 1e1, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e2, "gamma_s": 1e2, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e2, "gamma_s": 1e3, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #         
        #        {"lam": 1e3, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e3, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e3, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e3, "gamma_s": 1e0, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e3, "gamma_s": 1e1, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e3, "gamma_s": 1e2, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        {"lam": 1e3, "gamma_s": 1e3, "normalized": True, "kernel": "rbf", "multi_class": "ovr"},
        #        
        #    ],
        #},
        
    }

    
    blocked_datasets = []

    experimentor = SSLRateDatesetEvaluator(
        base_paths,
        output_path,
        classifiers_info,
        blocked_datasets
    )

    experimentor.run()


if __name__ == '__main__':
    main()
    
