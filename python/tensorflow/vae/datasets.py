import numpy as np

class DataReader(object):

    def __init__(self,
            train_path="/home/kzk/.chainer/dataset/pfnet/chainer/mnist/train.npz", 
            test_path="/home/kzk/.chainer/dataset/pfnet/chainer/mnist/test.npz",
            batch_size=32):
            
        self.train_data = np.load(train_path)
        self.test_data = np.load(test_path)

        self._batch_size = 32
        self._next_position_train = 0
        self._next_position_test = 0

        self._n_train_data = len(self.train_data["x"])
        self._n_test_data = len(self.test_data["x"])

    def get_train_batch(self,):
        # Read data
        batch_data_x = \
          self.train_data["x"][self._next_position_train:self._next_position_train+self._batch_size]
        batch_data_y = \
          self.train_data["y"][self._next_position_train:self._next_position_train+self._batch_size]

        # Reset pointer
        self._next_position_train += self._batch_size
        if self._next_position_train >= self._n_train_data:
            self._next_position_train = 0
        
        return batch_data_x, batch_data_y
            
    def get_test_batch(self,):
        # Read data
        batch_data_x = \
          self.test_data["x"][self._next_position_test:self._next_position_test+self._batch_size]
        batch_data_y = \
          self.test_data["y"][self._next_position_test:self._next_position_test+self._batch_size]

        # Reset pointer
        self._next_position_test += self._batch_size
        if self._next_position_test >= self._n_test_data:
            self._next_position_test = 0

        return batch_data_x, batch_data_y

    
