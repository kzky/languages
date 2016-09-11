import numpy as np
import os

class DataReader(object):

    def __init__(self,
        l_train_path=\
            "/home/kzk/.chainer/dataset/pfnet/chainer/mnist/l_train.npz",
        u_train_path=\
            "/home/kzk/.chainer/dataset/pfnet/chainer/mnist/u_train.npz", 
        test_path="/home/kzk/.chainer/dataset/pfnet/chainer/mnist/test.npz",
        batch_size=32):
        """
        Args:
          train_path: Dict, NpzFile, or some like that, one key-value holds whole data.
          test_path: Dict, NpzFile, or some like that, one key-value holds whole data.
        """
            
        self.l_train_data = np.load(l_train_path)
        self.u_train_data = np.load(u_train_path)
        self.test_data = np.load(test_path)

        self._batch_size = batch_size
        self._next_position_l_train = 0
        self._next_position_u_train = 0
        self._next_position_test = 0

        self._n_l_train_data = len(self.l_train_data["x"])
        self._n_u_train_data = len(self.u_train_data["x"])
        self._n_test_data = len(self.test_data["x"])

    def get_l_train_batch(self,):
        """Return next batch data.

        Returns:
           tuple of 2: First is for sample and the second is for label.
                               First data is binarized if a value is greater than 0, then 1;
                               otherwise 0.
        """
        # Read data
        beg = self._next_position_l_train
        end = self._next_position_l_train+self._batch_size
        batch_data_x_ = self.l_train_data["x"][beg:end, :]
        batch_data_y_ = self.l_train_data["y"][beg:end]

        # Change to one-hot representaion
        batch_data_y = np.zeros((len(batch_data_y_), self._n_cls))
        batch_data_y[np.arange(len(batch_data_y_)), batch_data_y_] = 1

        # Reset pointer
        self._next_position_l_train += self._batch_size
        if self._next_position_l_train >= self._n_l_train_data:
            self._next_position_l_train = 0

            # shuffle
            idx = np.arange(self._n_l_train_data)
            np.random.shuffle(idx)
            self.l_train_data["x"] = self.l_train_data["x"][idx]
            self.l_train_data["y"] = self.l_train_data["y"][idx]
        
        return batch_data_x, batch_data_y

    def get_u_train_batch(self,):
        """Return next batch data.

        Returns:
           tuple of 2: First is for sample and the second is for label.
                               First data is binarized if a value is greater than 0, then 1;
                               otherwise 0.
        """
        # Read data
        beg = self._next_position_u_train
        end = self._next_position_u_train+self._batch_size
        batch_data_x_ = self.u_train_data["x"][beg:end, :]
        batch_data_y_ = self.u_train_data["y"][beg:end]

        # Change to one-hot representaion
        batch_data_y = np.zeros((len(batch_data_y_), self._n_cls))
        batch_data_y[np.arange(len(batch_data_y_)), batch_data_y_] = 1

        # Reset pointer
        self._next_position_u_train += self._batch_size
        if self._next_position_u_train >= self._n_u_train_data:
            self._next_position_u_train = 0

            # shuffle
            idx = np.arange(self._n_u_train_data)
            np.random.shuffle(idx)
            self.u_train_data["x"] = self.u_train_data["x"][idx]
            self.u_train_data["y"] = self.u_train_data["y"][idx]
        
        return batch_data_x, batch_data_y

    def get_test_batch(self,):
        """Return next batch data.

        Returns:
           tuple of 2: First is for sample and the second is for label.
                               First data is binarized if a value is greater than 0, then 1;
                               otherwise 0.
        """

        # Read data
        beg = self._next_position_test
        end = self._next_position_test+self._batch_size

        batch_data_x_ = self.test_data["x"][beg:end, :]
        batch_data_x = np.reshape(batch_data_x_, (len(batch_data_x_), 28, 28, 1))
        
        # Change to one-hot representaion
        batch_data_y_ = self.test_data["y"][beg:end]
        batch_data_y = np.zeros((len(batch_data_y_), self._n_cls))
        batch_data_y[np.arange(len(batch_data_y_)), batch_data_y_] = 1

        # Reset pointer
        self._next_position_test += self._batch_size
        if self._next_position_test >= self._n_test_data:
            self._next_position_test = 0

        return batch_data_x, batch_data_y

class Separator(object):
    """Seprate the original samples to labeled samples and unlabeled samples.

    Seprate the original samples to labeled samples and unlabeled samples in such
    way; the number of labeled samples are selected randomly, it is equal to `l`, 
    and the others are unlabeled samples.
    """

    def __init__(self, u=100):
        self.u = u

    def separate_then_save(self,
                fpath="/home/kzk/.chainer/dataset/pfnet/chainer/mnist/train.npz"):
        ldata, udata = self._seprate(fpath)
        self._save_ssl_data(fpath, ldata, udata)
        
    def _separate(self,
                 fpath="/home/kzk/.chainer/dataset/pfnet/chainer/mnist/train.npz"):
        
        data = np.load(fpath)
        n = len(data)
        idxs = np.arange(n)
        idxs_l = np.random.choice(idxs, size=self.u, replace=False)
        idxs_u = np.asarray(list(set(idxs) - set(idx_l)))
        
        return data[idxs_l], data[idxs_u]
        
    def _save_ssl_data(self, fpath, ldata, udata):
        dpath = os.path.dirname(fpath)
        fname = os.path.basename(fpath)

        l_fname = "l_{}".format(fname)
        u_fname = "u_{}".format(fname)
        
        ldata_fpath = os.path.join(dpath, l_frame)
        udata_fpath = os.path.join(dpath, u_frame)

        np.savez(ldata_fpath)
        np.savez(udata_fpath)
