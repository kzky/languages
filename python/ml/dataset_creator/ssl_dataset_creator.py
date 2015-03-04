import numpy as np
import os
import logging
import shutil
from dataset_creator import DatasetCreator

# TODO Ambient version
class SSLRateDatasetCreator(DatasetCreator):
    """
    SSLDatasetCreator creates dataset for SSL.

    Creation process is the following way.
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
    
    def __init__(self, input_paths, output_dir_path, n=10, lrate=0.1, urate=0.5, delimiter=" "):
        """
        
        Arguments:
        - `input_paths`: list of dataset path, a files contain a dataset, filename have to include ".", e.g., dataset.csv.
        - `output_dir_path`: used as prefix for output dataset paths and files
        - `n`: the number of datasets to be created for one dataset.
        - `lrate`: rate at which labeled samples are generated and added as suffix.
        - `urate`: rate at which labeled samples are generated and added as suffix.
        - `delimiter`: delimiter.
        """
        super(SSLRateDatasetCreator, self).__init__()
        
        self.input_paths = input_paths
        self.output_dir_path = output_dir_path
        self.n = n
        self.lrate = lrate
        self.urate = urate
        self.delimiter = delimiter

        pass

    def create_ssl_datasets(self, ):
        """
        """
        base_dir_path = "%s_%d_%d_%d" % (
            self.output_dir_path,
            int(self.lrate * 100),
            int(self.urate * 100),
            int(100 - int(self.lrate * 100) - int(self.urate * 100))
        )

        if not os.path.exists(base_dir_path):
            os.makedirs(base_dir_path)
        else:
            shutil.rmtree(base_dir_path)

        for dataset_path in self.input_paths:  # each dataset file
            self.logger.info("processing %s ..." % dataset_path)
            for i in xrange(self.n):
                self.logger.info("creating %d-th" % i)
                self._create_ssl_dataset(base_dir_path, dataset_path, i)
                pass
            pass

    def _create_ssl_dataset(self, base_dir_path, dataset_path, prefix):
        """
        
        Arguments:
        - `dataset_path`:
        """

        if not os.path.exists(dataset_path):
            self.logger.info("%s does not exists" % dataset_path)
            return
        
        data = np.loadtxt(dataset_path, delimiter=self.delimiter)
        y = data[:, 0]
        X = data[:, 1]

        classes = set(y)
        n_samples = X.shape[0]
        all_indices = np.random.permutation(n_samples)

        # labeled samples
        l_data = None
        l_indices = None

        cnt = 0
        while True:
            self.logger.info("%d-th trial" % cnt)
            cnt += 1

            if cnt == self.TIRAL_THRESHOLD:
                return
                pass

            indices = np.random.permutation(n_samples)
            l_indices = indices[0:int(n_samples * self.lrate)]
            y_l_set = set(y[l_indices])
            
            if y_l_set == classes:
                l_data = data[l_indices, :]
                break
                
        # unlabeled samples
        r_indices = list(set(all_indices) - set(l_indices))
        u_indices = r_indices[0:int(1.0 * n_samples * self.urate)]
        u_data = data[u_indices, :]

        # test samples
        t_indices = list(set(all_indices) - set(l_indices) - set(u_indices))
        t_data = data[t_indices, :]

        # dump data
        dataset_name = dataset_path.split("/")[-1].split(".")[0]
        dir_path = "%s/%s" % (
            base_dir_path,
            dataset_name,
        )

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        f_ldata_path = "%s/%d_%s_l.csv" % (dir_path, prefix, dataset_name)
        np.savetxt(f_ldata_path, l_data, fmt="%s")
        self.logger.debug("%s created" % f_ldata_path)

        f_udata_path = "%s/%d_%s_u.csv" % (dir_path, prefix, dataset_name)
        np.savetxt(f_udata_path, u_data, fmt="%s")
        self.logger.debug("%s created" % f_udata_path)
        
        f_tdata_path = "%s/%d_%s_t.csv" % (dir_path, prefix, dataset_name)
        np.savetxt(f_tdata_path, t_data, fmt="%s")
        self.logger.debug("%s created" % f_tdata_path)
        
    
        pass

# TODO Create Transductive/Ambient version
