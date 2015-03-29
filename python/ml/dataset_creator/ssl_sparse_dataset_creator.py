import numpy as np
import os
import logging

from ssl_dataset_creator import SSLRateDatasetCreator
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file

class SSLRateSparseDatasetCreator(SSLRateDatasetCreator):
    """
    SSLRateDatasetCreator creates dataset for SSL.

    """

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("SSLRateSparseDatasetCreator")

    TIRAL_THRESHOLD = 50

    # all folds are not overlaped
    DATA_TYPE_NO_OVERLAP = "no_overlap"

    # labeled and validation fold are not overlaped
    # but unlabeled and test may be overlaped with each other
    DATA_TYPE_OVERLAP = "overlap"
    
    def __init__(self,
                 input_paths,
                 output_dirpath,
                 n=10,
                 l_rate=0.1,
                 u_rate=0.5,
                 v_rate=0.1,
                 delimiter=" ",
                 data_type=DATA_TYPE_NO_OVERLAP):

        """
        """
        
        super(SSLRateSparseDatasetCreator, self).__init__(
            input_paths=input_paths,
            output_dirpath=output_dirpath,
            n=n,
            l_rate=l_rate,
            u_rate=u_rate,
            v_rate=v_rate,
            delimiter=delimiter,
            data_type=data_type,
        )

        if data_type == self.DATA_TYPE_OVERLAP:
            self._create_ssl_dataset = self._create_ssl_dataset_for_overlap
        elif data_type == self.DATA_TYPE_NO_OVERLAP:
            self._create_ssl_dataset = self._create_ssl_dataset_for_no_overlap
        else:
            raise Exception("data_type = %s does not exist." % (self.data_type))
            
        pass



    def _create_ssl_dataset_for_overlap(self,
                                        base_output_dirpath, dataset_path, prefix):
        """
        Create dataset.
        
        Samples in each dataset type; unlabeld and test, are overlaped partialy
        so that there exists the exactly same samples among these datasets
        but not for labeled and validaion dataset.
        
        Arguments:
        - `base_output_dirpath`:
        - `dataset_path`:
        - `prefix`:
        """
        if not os.path.exists(dataset_path):
            self.logger.info("%s does not exists" % dataset_path)
            return

        (X, y) = load_svmlight_file(dataset_path)
        n_samples = len(y)
        all_indices = np.random.permutation(n_samples)
        
        # labeled and validation samples
        (l_indices, v_indices) = self._create_labeled_and_validation_indices(y)
            
        l_X = X[l_indices, :]
        l_y = y[l_indices]
        v_X = X[v_indices, :]
        v_y = y[v_indices]
        
        # unlabeled samples
        r_indices = list(set(all_indices) - set(l_indices) - set(v_indices))
        np.random.shuffle(r_indices)
        u_indices = r_indices[0:int(1.0 * n_samples * self.u_rate)]
        u_X = X[u_indices, :]
        u_y = y[u_indices]

        # test samples
        t_indices = r_indices
        t_X = X[t_indices, :]
        t_y = y[t_indices]

        # dump data
        dataset_name = dataset_path.split("/")[-1].split(".")[0]
        dirpath = "%s/%s" % (
            base_output_dirpath,
            dataset_name,
        )

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
            
        f_ldata_path = "%s/%d_l.csv" % (dirpath, prefix)
        dump_svmlight_file(l_X, l_y, f_ldata_path)
        self.logger.debug("%s created" % f_ldata_path)

        f_vdata_path = "%s/%d_v.csv" % (dirpath, prefix)
        dump_svmlight_file(v_X, v_y, f_vdata_path)
        self.logger.debug("%s created" % f_vdata_path)

        f_udata_path = "%s/%d_u.csv" % (dirpath, prefix)
        dump_svmlight_file(u_X, u_y, f_udata_path)
        self.logger.debug("%s created" % f_udata_path)
        
        f_tdata_path = "%s/%d_t.csv" % (dirpath, prefix)
        dump_svmlight_file(t_X, t_y, f_tdata_path)
        self.logger.debug("%s created" % f_tdata_path)
    
        pass
            
    def _create_ssl_dataset_for_no_overlap(self,
                                           base_output_dirpath, dataset_path, prefix):
        """
        Create dataset.
        
        Samples in each dataset type; labeled, unlabeld, validation, and test,
        are not overlaped so that there is no the exactly same samples
        among datasets.
        
        Arguments:
        - `base_output_dirpath`:
        - `dataset_path`:
        - `prefix`:
        """

        if not os.path.exists(dataset_path):
            self.logger.info("%s does not exists" % dataset_path)
            return

        (X, y) = load_svmlight_file(dataset_path)
        n_samples = len(y)
        all_indices = np.random.permutation(n_samples)

        # labeled and validation samples
        (l_indices, v_indices) = self._create_labeled_and_validation_indices(y)
            
        l_X = X[l_indices, :]
        l_y = y[l_indices]
        v_X = X[v_indices, :]
        v_y = y[v_indices]
                    
        # unlabeled samples
        r_indices = list(set(all_indices) - set(l_indices) - set(v_indices))
        np.random.shuffle(r_indices)
        u_indices = r_indices[0:int(1.0 * n_samples * self.u_rate)]
        u_X = X[u_indices, :]
        u_y = y[u_indices]

        # test samples
        t_indices = list(set(all_indices) - set(l_indices) - set(u_indices) - set(v_indices))
        t_X = X[t_indices, :]
        t_y = y[t_indices]

        # dump data
        dataset_name = dataset_path.split("/")[-1].split(".")[0]
        dirpath = "%s/%s" % (
            base_output_dirpath,
            dataset_name,
        )

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
            
        f_ldata_path = "%s/%d_l.csv" % (dirpath, prefix)
        dump_svmlight_file(l_X, l_y, f_ldata_path)
        self.logger.debug("%s created" % f_ldata_path)

        f_vdata_path = "%s/%d_v.csv" % (dirpath, prefix)
        dump_svmlight_file(v_X, v_y, f_vdata_path)
        self.logger.debug("%s created" % f_vdata_path)

        f_udata_path = "%s/%d_u.csv" % (dirpath, prefix)
        dump_svmlight_file(u_X, u_y, f_udata_path)
        self.logger.debug("%s created" % f_udata_path)
        
        f_tdata_path = "%s/%d_t.csv" % (dirpath, prefix)
        dump_svmlight_file(t_X, t_y, f_tdata_path)
        self.logger.debug("%s created" % f_tdata_path)
    
        pass


