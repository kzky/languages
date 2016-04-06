#!/usr/bin/env python

from matplotlib.ticker import MaxNLocator
from results_coordinator import ResultsCoordinator
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import logging
import csv

class SSLResultsCoordinator(ResultsCoordinator):
    """
    """
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("SSLResultsCoordinator")

    header = ["dataset", "classifier", "rate_l_u_v_t", "ave_acc", "std_acc"]
    
    def __init__(self, input_filepaths=[], output_dirpath="", index=0):
        """
        """

        super(SSLResultsCoordinator, self).__init__(
            input_filepaths, output_dirpath)

        # DI
        self._compute_ave_std_acc = self.__compute_ave_std_acc
        self._save_as_figure = self.__save_as_figure_fixed_urate
        self._save_as_table = self.__save_as_table
        self._index = index
        
        pass
        
    def __compute_ave_std_acc(self, result):
        """
        
        Arguments:
        - `result`:
        """
        self.logger.debug("__compute_ave_std_acc")

        # acc
        for dataname in result:
            for classifier_name in result[dataname]:
                for l_u_v_t in result[dataname][classifier_name]:
                    for i in result[dataname][classifier_name][l_u_v_t]:
    
                        preds = result[dataname][classifier_name][l_u_v_t][i]['preds']
                        labels = result[dataname][classifier_name][l_u_v_t][i]['labels']
                        preds = np.asarray(preds)
                        labels = np.asarray(labels)
                        n = len(labels)
                        hits = len(np.where(preds == labels)[0])
                        result[dataname][classifier_name][l_u_v_t][i]['acc'] = 100.0 * hits / n
                    
                    pass
                pass
            pass

        # ave (acc) and std(acc)
        for dataname in result:
            for classifier_name in result[dataname]:
                for l_u_v_t in result[dataname][classifier_name]:
                    acc = []
                    for i in result[dataname][classifier_name][l_u_v_t]:
                        acc_ = result[dataname][classifier_name][l_u_v_t][i]['acc']
                        acc.append(acc_)
                    pass
                    ave_acc = np.average(acc)
                    std_acc = np.std(acc)
                    
                    result[dataname][classifier_name][l_u_v_t]["ave_acc"] = ave_acc
                    result[dataname][classifier_name][l_u_v_t]["std_acc"] = std_acc
                pass
            pass
        
        return result

    def __save_as_figure_fixed_urate(self, result):
        """
        save results as figure for each dataset.
        x-axis is labeled rate.
        y-axis is average of accuracy +- standard deviation
        
        Arguments:
        - `result`: dictionary of the form, <datasetname, <classifier_name, <rate_l_u_v_t, <{ave_acc, std_acc}, val>>>>

        """
        # output dirpath
        output_dirpath = self.output_dirpath
        
        datanames = result.keys()
        datanames.sort()
        for dataname in datanames:
            classifier_names = result[dataname].keys()
            classifier_names.sort()

            # figure
            fig, ax = plt.subplots(1, 1)
            ax = fig.add_subplot(111)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            for classifier_name in classifier_names:
                l_u_v_ts = result[dataname][classifier_name].keys()
                l_luvt_map = self._create_l_luvt_map(l_u_v_ts)
                ls = l_luvt_map.keys()
                ls.sort()
                
                result_ = result[dataname][classifier_name]
                ave_accs = []
                std_accs = []
                x_lim = []
                
                for l in ls:
                    luvt = l_luvt_map[l]
                    ave_acc = result_[luvt]["ave_acc"]
                    std_acc = result_[luvt]["std_acc"]
                    ave_accs.append(ave_acc)
                    std_accs.append(std_acc)
                    x_lim.append(l)
                    pass

                # plot
                eb = ax.errorbar(x_lim, ave_accs, std_accs,
                                 label=classifier_name, fmt="-")
                eb.lines[-1][0].set_linestyle('--')
                pass

            # save figure
            path = "%s/%s.png" % (output_dirpath, dataname)
            ax.legend(loc=4)
            xlim_min = ls[0] - 0.5
            xlim_max = ls[-1] + 0.5

            ax.set_xlim(xlim_min, xlim_max)
            fig.savefig(path, format="png", dpi=300)
            plt.close(fig)
            self.logger.info("%s is saved" % path)
        pass


    def __save_as_table(self, result):
        """
        save result as table with the following format,
        ----------
        header
        dataset_name, classifier_name, rate_l_u_v_t, ave_acc, std_acc
        dataset_name, classifier_name, rate_l_u_v_t, ave_acc, std_acc
        ...
        ----------
        
        Arguments:
        - `result`: dictionary of the form, <datasetname, <classifier_name, <rate_l_u_v_t, <{ave_acc, std_acc}, val>>>>

        """
        # output dirpath
        output_dirpath = self.output_dirpath
        output_path = "%s/table.csv" % (output_dirpath)

        with open(output_path, "w") as fpout:
            writer = csv.writer(fpout)
            writer.writerow(self.header)

            datanames = result.keys()
            datanames.sort()
            for dataname in datanames:
                classifier_names = result[dataname].keys()
                classifier_names.sort()
                
                for classifier_name in classifier_names:
                    l_u_v_ts = result[dataname][classifier_name].keys()
                    l_u_v_ts.sort()
                    
                    result_ = result[dataname][classifier_name]
                    
                    for l_u_v_t in l_u_v_ts:
                        ave_acc = result_[l_u_v_t]["ave_acc"]
                        std_acc = result_[l_u_v_t]["std_acc"]

                        row = [
                            dataname,
                            classifier_name,
                            l_u_v_t,
                            ave_acc,
                            std_acc
                        ]
                        writer.writerow(row)
                    pass
                pass
            pass

    def _create_l_luvt_map(self, l_u_v_ts):
        """
        retrieve ls from l_u_v_ts
        Arguments:
        - `l_u_v_ts`: list of lrate_urate_vrate_trate, e.g.,
        ["1_2_3_4", "5_6_7_8"],
        ["ssl_biased_1_4", "ssl_unbiased_7_8"],

        """

        l_luvt_map = defaultdict()
        for l_u_v_t in l_u_v_ts:
            l = int(l_u_v_t.split("_")[self._index])
            l_luvt_map[l] = l_u_v_t
            pass

        return l_luvt_map
        
def main():
    base_intput_dirpath = "/home/kzk/experiment/final_paper_2015/uci_dataset/pkl"
    input_filepaths = [
        "%s/%s" % (base_intput_dirpath, "lrate_fixed_with_1_reghpfssl_batch_with_beta_upper_bound.pkl")
    ]

    output_dirpath = "/home/kzk/experiment/final_paper_2015/uci_dataset/results/lrate_fixed_with_1_reghpfssl_batch_with_beta_upper_bound"
    
    index = 1
    results_coordinator = SSLResultsCoordinator(
        input_filepaths=input_filepaths,
        output_dirpath=output_dirpath,
        index=index
    )

    results_coordinator.coordinate()
    pass

if __name__ == '__main__':
    main()
