import numpy as np
import os
import logging
import shutil

from collections import defaultdict
from dataset_creator import DatasetCreator

class SSLRateDatasetCreator(DatasetCreator):
    """
    SSLDatasetCreator creates dataset for SSL.

    Creation process for test data in the ambient space is the following way.
    1. loop for each dataset
    2. load dataset
    3. loop until a specified number
    3.1. select labeled samples uniform-randomly with a specified rate,
            where each class sample is included at least one.
    3.2. take all samples from dataset except for the selected ones in 3.1 with a rate.
    3.3. take all samples from dataset except for the selected ones in 3.1. and 3.2.
    4. dump a created dataset into a file.

    Thus, dataset devided by 3-fold is created.

    Dataest format have to be as follows,
    y_1,x_11,x_12, ...,x_1d
    y_2,x_11,x_22, ...,x_2d
    ...
    y_n,x_n1,x_n2, ...,x_nd
    .
    
    This is a comma-seprated format, but (tab, space)-separated is also permitted.
    
    """

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("SSLRateDatasetCreator")

    TIRAL_THRESHOLD = 50

    DATA_TYPE_NO_OVERLAP = "no_overlap"
    DATA_TYPE_OVERLAP = "overlap"
    
    def __init__(self,
                 input_paths,
                 output_prefix_dirpath,
                 n=10,
                 l_rate=0.1,
                 u_rate=0.5,
                 v_rate=0.1,
                 delimiter=" ",
                 data_type=DATA_TYPE_NO_OVERLAP):

        """
        
        Arguments:
        - `input_paths`: list of dataset path, a files contain a dataset, filename have to include ".", e.g., dataset.csv.
        - `output_prefix_dirpath`: used as prefix for output dataset paths and files
        - `n`: the number of datasets to be created for one dataset.
        - `l_rate`: rate at which labeled samples are generated and added as suffix.
        - `u_rate`: rate at which labeled samples are generated and added as suffix.
        - `delimiter`: delimiter.
        """
        super(SSLRateDatasetCreator, self).__init__()
        
        self.input_paths = input_paths
        self.output_prefix_dirpath = output_prefix_dirpath
        self.n = n
        self.l_rate = l_rate
        self.u_rate = u_rate
        self.v_rate = v_rate
        self.delimiter = delimiter
        self.data_type = data_type

        if data_type == self.DATA_TYPE_OVERLAP:
            self._create_ssl_dataset = self._create_ssl_dataset_for_overlap
        elif data_type == self.DATA_TYPE_NO_OVERLAP:
            self._create_ssl_dataset = self._create_ssl_dataset_for_no_overlap
        else:
            raise Exception("data_type = %s does not exist." % (self.data_type))
            
        pass

    def create_ssl_datasets(self, ):
        """
        """
        base_output_dirpath = "%s_%d_%d_%d_%d" % (
            self.output_prefix_dirpath,
            int(self.l_rate * 100),
            int(self.u_rate * 100),
            int(self.v_rate * 100),
            int(100 - int(self.l_rate * 100) - int(self.u_rate * 100) - int(self.l_rate * 100))
        )

        if not os.path.exists(base_output_dirpath):
            os.makedirs(base_output_dirpath)
        else:
            shutil.rmtree(base_output_dirpath)

        for dataset_path in self.input_paths:  # each dataset file
            self.logger.info("processing %s ..." % dataset_path)
            for i in xrange(self.n):
                self.logger.info("creating %d-th" % i)
                self._create_ssl_dataset(base_output_dirpath, dataset_path, i)
                pass
            pass

    def _create_ssl_dataset_for_overlap(self,
                                        base_output_dirpath, dataset_path, prefix):
        """
        Create dataset.
        
        Samples in each dataset type; unlabeld and test, are overlaped partialy
        so that there is the exactly same samples among these datasets
        but not in labeled and validaion dataset.
        
        Arguments:
        - `base_output_dirpath`:
        - `dataset_path`:
        - `prefix`:
        """
        if not os.path.exists(dataset_path):
            self.logger.info("%s does not exists" % dataset_path)
            return

        data = np.loadtxt(dataset_path, delimiter=self.delimiter)
        n_samples = data.shape[0]
        all_indices = np.random.permutation(n_samples)
        
        # labeled and validation samples
        (l_indices, v_indices) = self._create_labeled_and_validation_indices(data)
            
        l_data = data[l_indices, :]
        v_data = data[v_indices, :]

        # unlabeled samples
        r_indices = list(set(all_indices) - set(l_indices) - set(v_indices))
        np.random.shuffle(r_indices)
        u_indices = r_indices[0:int(1.0 * n_samples * self.u_rate)]
        u_data = data[u_indices, :]

        # test samples
        t_indices = r_indices
        t_data = data[t_indices, :]

        # dump data
        dataset_name = dataset_path.split("/")[-1].split(".")[0]
        dirpath = "%s/%s" % (
            base_output_dirpath,
            dataset_name,
        )

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
            
        f_ldata_path = "%s/%d_l.csv" % (dirpath, prefix)
        np.savetxt(f_ldata_path, l_data, fmt="%s")
        self.logger.debug("%s created" % f_ldata_path)

        f_vdata_path = "%s/%d_v.csv" % (dirpath, prefix)
        np.savetxt(f_vdata_path, v_data, fmt="%s")
        self.logger.debug("%s created" % f_vdata_path)

        f_udata_path = "%s/%d_u.csv" % (dirpath, prefix)
        np.savetxt(f_udata_path, u_data, fmt="%s")
        self.logger.debug("%s created" % f_udata_path)
        
        f_tdata_path = "%s/%d_t.csv" % (dirpath, prefix)
        np.savetxt(f_tdata_path, t_data, fmt="%s")
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

        data = np.loadtxt(dataset_path, delimiter=self.delimiter)
        n_samples = data.shape[0]
        all_indices = np.random.permutation(n_samples)

        # labeled and validation samples
        (l_indices, v_indices) = self._create_labeled_and_validation_indices(data)
            
        l_data = data[l_indices, :]
        v_data = data[v_indices, :]
            
        # unlabeled samples
        r_indices = list(set(all_indices) - set(l_indices) - set(v_indices))
        np.random.shuffle(r_indices)
        u_indices = r_indices[0:int(1.0 * n_samples * self.u_rate)]
        u_data = data[u_indices, :]

        # test samples
        t_indices = list(set(all_indices) - set(l_indices) - set(u_indices) - set(v_indices))
        t_data = data[t_indices, :]

        # dump data
        dataset_name = dataset_path.split("/")[-1].split(".")[0]
        dirpath = "%s/%s" % (
            base_output_dirpath,
            dataset_name,
        )

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
            
        f_ldata_path = "%s/%d_l.csv" % (dirpath, prefix)
        np.savetxt(f_ldata_path, l_data, fmt="%s")
        self.logger.debug("%s created" % f_ldata_path)

        f_vdata_path = "%s/%d_v.csv" % (dirpath, prefix)
        np.savetxt(f_vdata_path, v_data, fmt="%s")
        self.logger.debug("%s created" % f_vdata_path)

        f_udata_path = "%s/%d_u.csv" % (dirpath, prefix)
        np.savetxt(f_udata_path, u_data, fmt="%s")
        self.logger.debug("%s created" % f_udata_path)
        
        f_tdata_path = "%s/%d_t.csv" % (dirpath, prefix)
        np.savetxt(f_tdata_path, t_data, fmt="%s")
        self.logger.debug("%s created" % f_tdata_path)
    
        pass

    def _create_labeled_and_validation_indices(self, data):
        """
        """

        # labeled samples and samples for validation
        l_indices = None
        v_indices = None
        y = data[:, 0]
        classes = set(y)
        n_samples = len(y)
        
        cnt = 0
        while True:
            self.logger.info("%d-th trial" % cnt)
            cnt += 1

            if cnt == self.TIRAL_THRESHOLD:
                return self._add_labeled_and_validation_indices(y, l_indices, v_indices)
                pass

            # labeled samples
            indices = np.random.permutation(n_samples)
            l_indices = indices[0:int(n_samples * self.l_rate)]
            y_l_indices_set = set(y[l_indices])

            # validation samples
            r_indices = list(set(indices) - set(l_indices))
            np.random.shuffle(r_indices)
            v_indices = r_indices[0:int(n_samples * self.v_rate)]
            y_v_indices_set = set(y[v_indices])
            
            if y_l_indices_set == classes and y_v_indices_set == classes:
                break
        
        return (l_indices, v_indices)

    def _add_labeled_and_validation_indices(self,
                                            y, l_indices, v_indices):
        
        """
        
        called when method, _create_labeled_and_validation_indices, can not
        create enough samples.
        """
        # classes
        classes = set(y)

        # class distribution
        class_indices_map = defaultdict(list)

        for i, y_ in enumerate(y):
            class_indices_map[y_].append(i)
            pass

        # add for y_l_indices
        l_indices = list(l_indices)
        short_classes = classes - set(l_indices)
        for c in short_classes:
            r = int(np.random.uniform(0, len(class_indices_map[c]), 1)[0])
            idx = class_indices_map[c][r]
            l_indices.append(idx)
            pass

        # add for y_v_indices
        v_indices = list(v_indices)
        short_classes = classes - set(v_indices)
        for c in short_classes:
            ridx = int(np.random.uniform(0, len(class_indices_map[c]), 1)[0])
            idx = class_indices_map[c][ridx]
            v_indices.append(idx)
            pass

        return (l_indices, v_indices)
