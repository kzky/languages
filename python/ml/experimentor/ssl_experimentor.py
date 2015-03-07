#!/usr/bin/env python

import logging
import glob
import numpy as np

from experimentor import Experimentor
from collections import defaultdict
from cloud.serialization import cloudpickle

LABELED_DATASET_SUFFIX = "l"
UNLABELED_DATASET_SUFFIX = "u"
TEST_DATASET_SUFFIX = "t"

class SSLRateDatesetEvaluator(Experimentor):
    """
    Evaluator for SSLRateDataset.
    """

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("SSLRateDatesetEvaluator")

    def __init__(self, base_dataset_path, output_path, classifiers=[]):
        """
        Arguments:
        - `base_dataset_path`: path ends with _${label_rate}_${validation_rate}_${}unlabel_rate}_${test_rate}.
        - `classifiers`: classifiers to be compared for experiments.
        """
        super(SSLRateDatesetEvaluator, self).__init__()

        # dataset info
        self.base_dataset_path = base_dataset_path
        components = base_dataset_path.split("/")[-1].split("_")
        self.trate = components[-1]
        self.urate = components[-2]
        self.vrate = components[-3]
        self.lrate = components[-4]
        self.rate_pair = "%d_%d_%d" % (self.lrate, self.vrate, self.urate, self.trate)

        # output path
        self.output_path = output_path

        # classifiers
        self.classifiers = classifiers

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
        <dataset, <rate_pair, <classifier, <index, <{labels, preds}, {[], []}>>>>>

        to be saved with json.
        """

        dataset_paths = glob.glob("%s/*" % self.base_dataset_path)

        for dataset_path in dataset_paths:  # for each dataset
            self._run_dataset(dataset_path)
            pass

        self._save_results(self.output_path)
            
    def _run_with(self, dataset_path):
        """
        Arguments:

        -'dataset_path': path for a dataset
        """

        indices = self._retrieve_indices(dataset_path)

        for i in indices:  # for each sampled dataset from the same dataset
            self._run_internally_with(dataset_path, i)
            pass
                
    def _run_internally_with(self, dataset_path, index):
        """
        Arguments:

        -'dataset_path': path for a dataset
        -'index' dataset index
        """

        # dataset name
        dataset_name = dataset_path.split("/")[-1]

        # labeled dataset
        lpath = "%s/%d_%s_%s.csv" % (
            dataset_path, index, dataset_name, LABELED_DATASET_SUFFIX)
        ldata = np.loadtxt(lpath, delimiter=",")
        y_l = ldata[:, 0]
        X_l = ldata[:, 1:]

        # unlabeled dataset
        upath = "%s/%d_%s_%s.csv" % (
            dataset_path, index, dataset_name, UNLABELED_DATASET_SUFFIX)
        udata = np.loadtxt(upath, delimiter=",")
        X_u = udata[:, 1:]

        # test dataset
        tpath = "%s/%d_%s_%s.csv" % (
            dataset_path, index, dataset_name, TEST_DATASET_SUFFIX)
        tdata = np.loadtxt(tpath, delimiter=",")
        y_t = tdata[:, 0]
        X_t = tdata[:, 1:]

        # TODO
        # should one classifier be used becuase Lap-RLS has a lot of parameters
        # change classsifiers_info: dict
        for classifier in self.classifiers:

            try:
                classifier.learn(X_l, y_l, X_u)
                preds = []
                for x in X_t:
                    preds.add(classifier.predict(x))

                classifier_name = classifier.__class__.__name__
                self.results[dataset_name][self.rate_pair][classifier_name][index]["labels"] = y_t
                self.results[dataset_name][self.rate_pair][classifier_name][index]["preds"] = preds
                
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
        for path in glob.glob(dataset_path):
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

        with open(output_path, "w") as fpout:
            cloudpickle.dump(self.results, fpout)
            pass
        pass
