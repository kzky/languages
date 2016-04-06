#!/usr/bin/env python

import pickle
import logging

class ResultsCoordinator(object):
    """
    """
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("ResultsCoordinator")

    def __init__(self, input_filepaths=[], output_dirpath=""):
        """
        Here data pickled is of the form,
        <datasetname, <classifier_name, <rate_l_u_v_t, <data_index, <{preds/labels}, []>>>>>
        Arguments:
        - `input_filepaths`: base input filepaths from which results are loaded.
        - `output_filepaths`: dir
        """

        # input file paths
        self.input_filepaths = input_filepaths
        self.input_filepaths.sort()

        # output dirpath
        self.output_dirpath = output_dirpath

        # DI
        self._compute_ave_std_acc = self.__compute_ave_std_acc
        self._save_as_figure = self.__save_as_figure
        self._save_as_table = self.__save_as_table
                
        pass
        
    def coordinate(self, ):
        """
        Coordinate results and return dictionary of the form,
        <datasetname, <classifier_name, <rate_l_u_v_t, <{ave_acc, ave_sd}, val>>>>
        """

        # load results
        results = self._load_results()
        
        # merge resutls
        result = self._merge_results(results)

        # computes average of acc and std
        result_acc = self._compute_ave_std_acc(result)

        # save as figure
        self._save_as_figure(result_acc)

        # save as table
        self._save_as_table(result_acc)

    def _load_results(self, ):
        """
        
        Arguments:
        - `datapath`:
        """
        results = []
        self.logger.info(self.input_filepaths)
        for datapath in self.input_filepaths:
            with open(datapath) as fpin:
                results.append(pickle.load(fpin))
                self.logger.info("%s is loaded" % (datapath))
                pass
        return results
        
    def _merge_results(self, results):
        """
        
        Arguments:
        - `results`:
        """
        result = results[0]
        if len(results) == 1:
            return result

        for result_ in results[1:]:
            for dataname in result_:
                for classifier_name in result_[dataname]:
                    # latter wins
                    result[dataname][classifier_name] = result_[dataname][classifier_name]
                    pass
                pass
            pass
        return result

    def __compute_ave_std_acc(self, result):
        """
        
        Arguments:
        - `result`:
        """

        self.logger.debug("__compute_ave_std_acc")
        pass

    def __save_as_figure(self, result):
        """
        
        Arguments:
        - `result`:
        """
        self.logger.debug("__save_as_figure")
        pass


    def __save_as_table(self, result):
        """
        
        Arguments:
        - `result`:
        """
        self.logger.debug("__save_as_table")

        pass
        
