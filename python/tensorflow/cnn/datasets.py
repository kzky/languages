import numpy as np

class DataReader(object):

    def __init__(self,
            train_path="/home/kzk/.chainer/dataset/pfnet/chainer/mnist/train.npz", 
            test_path="/home/kzk/.chainer/dataset/pfnet/chainer/mnist/test.npz",
            batch_size=32,
            n_cls=10):
        """
        Args:
          train_path: Dict, NpzFile, or some like that, one key-value holds whole data.
          test_path: Dict, NpzFile, or some like that, one key-value holds whole data.
        """
            
        self.train_data = dict(np.load(train_path))
        self.test_data = dict(np.load(test_path))

        self._batch_size = batch_size
        self._n_cls = n_cls
        self._next_position_train = 0
        self._next_position_test = 0

        self._n_train_data = len(self.train_data["x"])
        self._n_test_data = len(self.test_data["x"])

    def get_train_batch(self,):
        """Return next batch data.

        Returns:
           tuple of 2: First is for sample and the second is for label.
                               First data is binarized if a value is greater than 0, then 1;
                               otherwise 0.
        """
        # Read data
        beg = self._next_position_train
        end = self._next_position_train+self._batch_size

        batch_data_x_ = self.train_data["x"][beg:end, :]
        batch_data_x = np.reshape(batch_data_x_, (len(batch_data_x_), 28, 28, 1))

        # Change to one-hot representaion
        batch_data_y_ = self.train_data["y"][beg:end]
        batch_data_y = np.zeros((len(batch_data_y_), self._n_cls))
        batch_data_y[np.arange(len(batch_data_y_)), batch_data_y_] = 1

        # Reset pointer
        self._next_position_train += self._batch_size
        if self._next_position_train >= self._n_train_data:
            self._next_position_train = 0

            # shuffle
            idx = np.arange(self._n_train_data)
            np.random.shuffle(idx)
            self.train_data["x"] = self.train_data["x"][idx]
            self.train_data["y"] = self.train_data["y"][idx]
        
        return batch_data_x / 256. , batch_data_y
            
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

        return batch_data_x / 256. , batch_data_y

