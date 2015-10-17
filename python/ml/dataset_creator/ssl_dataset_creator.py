import numpy as np
import os
import logging
import sys

from collections import defaultdict
from dataset_creator import DatasetCreator

class SSLRateDatasetCreator(DatasetCreator):
    """
    SSLRateDatasetCreator creates dataset for SSL.

    """

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("SSLRateDatasetCreator")

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
        
        Arguments:
        - `input_paths`: list of dataset path, a files contain a dataset, filename have to include ".", e.g., dataset.csv.
        - `output_dirpath`: used as prefix for output dataset paths and files
        - `n`: the number of datasets to be created for one dataset.
        - `l_rate`: rate at which labeled samples are generated and added as suffix
        - `u_rate`: rate at which unlabeled samples are generated and added as suffix
        - `v_rate`: rate at which validation samples are generated and added as suffix
        - `delimiter`: delimiter.
        - `data_type`: data type
        """
        super(SSLRateDatasetCreator, self).__init__()
        
        self.input_paths = input_paths
        self.output_dirpath = output_dirpath
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

        data = np.loadtxt(dataset_path, delimiter=self.delimiter)
        n_samples = data.shape[0]
        all_indices = np.random.permutation(n_samples)
        
        # labeled and validation samples
        y = data[:, 0]
        (l_indices, v_indices) = self._create_labeled_and_validation_indices(y)
            
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
        y = data[:, 0]
        (l_indices, v_indices) = self._create_labeled_and_validation_indices(y)
            
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

    def _create_labeled_and_validation_indices(self, y):
        """
        """

        # labeled samples and samples for validation
        l_indices = None
        v_indices = None
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
        Called when the method, _create_labeled_and_validation_indices, can not
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
        short_classes = classes - set(y[l_indices])
        for c in short_classes:
            ridx = int(np.random.uniform(0, len(class_indices_map[c]), 1)[0])
            idx = class_indices_map[c][ridx]
            l_indices.append(idx)
            pass

        # add for y_v_indices
        v_indices = list(v_indices)
        short_classes = classes - set(y[v_indices])
        for c in short_classes:
            ridx = int(np.random.uniform(0, len(class_indices_map[c]), 1)[0])
            idx = class_indices_map[c][ridx]
            v_indices.append(idx)
            pass

        return (l_indices, v_indices)


    def _create_base_output_dirpath(self, ):
        """
        """

        if self.data_type == self.DATA_TYPE_NO_OVERLAP:
            base_output_dirpath = "%s_%d_%d_%d_%d" % (
                self.output_dirpath,
                int(self.l_rate * 100),
                int(self.u_rate * 100),
                int(self.v_rate * 100),
                100 - int(self.l_rate * 100) - int(self.u_rate * 100) - int(self.v_rate * 100)
            )
            return base_output_dirpath

        elif self.data_type == self.DATA_TYPE_OVERLAP:
            base_output_dirpath = "%s_%d_%d_%d_%d" % (
                self.output_dirpath,
                int(self.l_rate * 100),
                int(self.u_rate * 100),
                int(self.v_rate * 100),
                100 - int(self.l_rate * 100) - int(self.v_rate * 100)
            )
            return base_output_dirpath

        return None


class SSLDatasetCreator(DatasetCreator):
    
    """
    SSLDatasetCreator creates dataset for SSL.

    Number of labeled and vaildation samples is fixed.
    Number of unlabeld samples is ${urate}-times number of samples.
    Number of test samples is number of ({samples} - {labeled and validation samples})
    """

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("SSLDatasetCreator")

    # class prior is known
    DATA_TYPE_BIASED = "biased"

    # class prior is unknown, i.e., assuming uniform distribution
    DATA_TYPE_UNBIASED = "unbiased"
    
    def __init__(self,
                 input_paths,
                 prefix_output_dirpath,
                 n=10,
                 u_rate=0.5,
                 base_class_n_samples=1,
                 per_class_n_samples=1,
                 delimiter=" ",
                 data_type=DATA_TYPE_BIASED):

        """
        Arguments:
        - `input_paths`: list of dataset path, a files contain a dataset, filename have to include ".", e.g., dataset.csv.
        - `prefix_output_dirpath`: used as prefix for output dataset paths and files
        - `n`: the number of datasets to be created for one dataset.
        - `u_rate`: rate at which unlabeled samples are generated and added as suffix
        -`base_class_n_samples`: used when data type is biased
        -`per_class_n_samples`: used when data type is unbiased
        - `delimiter`: delimiter.
        -`data_type`:  data type is either biased or unbiased
        """
        super(SSLDatasetCreator, self).__init__()
        
        self.input_paths = input_paths
        self.prefix_output_dirpath = prefix_output_dirpath
        self.n = n
        self.u_rate = u_rate
        self.base_class_n_samples = base_class_n_samples
        self.per_class_n_samples = per_class_n_samples
        self.delimiter = delimiter
        self.data_type = data_type

        # DI in method-level
        if data_type == self.DATA_TYPE_BIASED:
            self._create_labeled_and_validation_indices = self._create_labeled_and_validation_indices_for_biased
        elif data_type == self.DATA_TYPE_UNBIASED:
            self._create_labeled_and_validation_indices = self._create_labeled_and_validation_indices_for_unbiased
        else:
            raise Exception("data_type = %s does not exist." % (self.data_type))
            
        pass

    def _create_base_output_dirpath(self, ):
        """
        """

        if self.data_type == self.DATA_TYPE_BIASED:
            base_prefix_output_dirpath = "%s_%s_%d_%d" % (
                self.prefix_output_dirpath,
                self.data_type,
                int(self.base_class_n_samples),
                int(self.u_rate * 100),
            )
            return base_prefix_output_dirpath

        elif self.data_type == self.DATA_TYPE_UNBIASED:
            base_prefix_output_dirpath = "%s_%s_%d_%d" % (
                self.prefix_output_dirpath,
                self.data_type,
                int(self.per_class_n_samples),
                int(self.u_rate * 100),
            )
            return base_prefix_output_dirpath

        return None

    def _create_ssl_dataset(self,
                            base_prefix_output_dirpath, dataset_path, prefix):
        """
        """
        if not os.path.exists(dataset_path):
            self.logger.info("%s does not exists" % dataset_path)
            return

        # load data
        data = np.loadtxt(dataset_path, delimiter=self.delimiter)
        n_samples = data.shape[0]
        all_indices = np.random.permutation(n_samples)

        # labeled and validation samples
        y = data[:, 0]
        (l_indices, v_indices) = self._create_labeled_and_validation_indices(y)
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
            base_prefix_output_dirpath,
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

    def _create_labeled_and_validation_indices_for_biased(self, y):
        """
        create indices of labeld samples and validation samples.
        """

        class_indices_map = defaultdict(list)
        class_prior_base = defaultdict(int)

        # create class indices map
        for i, cls in enumerate(y):
            class_indices_map[cls].append(i)
            class_prior_base[cls] += 1
            pass

        # create class distribution, value = freq / base_freq * base_class_samples
        class_prior = {}
        class_prior_ = sorted(class_prior_base.items(), key=lambda x: x[1])
        base_freq = class_prior_[0][1]

        base_class_n_samples = self.base_class_n_samples
        for cls, freq in class_prior_:
            class_prior[cls] = int(1.0 * freq / base_freq) * base_class_n_samples
            pass

        # sampling for indices of labled samples
        l_indices = []
        for cls, n_samples in class_prior.items():
            indices = class_indices_map[cls]
            np.random.shuffle(indices)
            indices_sampled = indices[0:n_samples]
            l_indices += indices_sampled
            pass

        # sampling for indices of validation samples
        v_indices = []
        l_indices_set = set(l_indices)
        for cls, n_samples in class_prior.items():
            indices = list(set(class_indices_map[cls]) - l_indices_set)
            np.random.shuffle(indices)
            indices_sampled = indices[0:n_samples]
            v_indices += indices_sampled
            pass
            
        return (l_indices, v_indices)

    def _create_labeled_and_validation_indices_for_unbiased(self, y):
        """
        create indices of labeld samples and validation samples.
        """

        class_indices_map = defaultdict(list)
        class_prior = defaultdict(int)

        for i, cls in enumerate(y):
            class_indices_map[cls].append(i)
            class_prior[cls] += 1
            pass
            
        # sampling for indices of labled samples
        l_indices = []
        for cls in class_prior.keys():
            indices = class_indices_map[cls]
            np.random.shuffle(indices)
            indices_sampled = indices[0:self.per_class_n_samples]
            l_indices += indices_sampled
            pass

        # sampling for indices of validation samples
        v_indices = []
        l_indices_set = set(l_indices)
        for cls in class_prior.keys():
            indices = list(set(class_indices_map[cls]) - l_indices_set)
            np.random.shuffle(indices)
            indices_sampled = indices[0:self.per_class_n_samples]
            v_indices += indices_sampled
            pass
            
        return (l_indices, v_indices)
            

# How to use
def main():
    input_paths = [
        "/home/kzk/datasets/uci_csv/activity.csv",
        "/home/kzk/datasets/uci_csv/wpbcRN.csv",
        "/home/kzk/datasets/uci_csv/yeast.csv",
    ]
     
    # "/home/kzk/datasets/uci_csv_ssl_lrate_fixed"
    # "/home/kzk/datasets/uci_csv_ssl_urate_fixed"
     
    prefix_output_dirpath = str(sys.argv[1])
    n = int(sys.argv[2])
    l_rate = float(sys.argv[3])
    u_rate = float(sys.argv[4])
    v_rate = float(sys.argv[5])
    data_type = str(sys.argv[6])
     
    creator = SSLRateDatasetCreator(
        input_paths,
        prefix_output_dirpath,
        n=n,
        l_rate=l_rate,
        u_rate=u_rate,
        v_rate=v_rate,
        data_type=data_type,
        delimiter=" ",
    )
     
    creator.create_ssl_datasets()

    pass

        
    

if __name__ == '__main__':
    main()
