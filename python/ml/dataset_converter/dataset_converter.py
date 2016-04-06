#!/usr/bin/env python

import logging
import csv
import os
import numpy as np

class DatasetConverter(object):
    """
    """

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("DatasetConverter")

    def __init__(self, input_filepath="", output_filepath=""):
        """
        - `input_filepath`: input file path
        - `output_filepath`: outpt file path
        
        """
        
        pass
        
    def convert(self, ):
        """
        """
        
class SparseDatasetConverter(DatasetConverter):
    """
    Convert sparse dataset of the form,
    -----
    y0<space>x0:v0<space>x1:v1<space>...
    y0<space>x0:v0<space>x1:v1<space>...
    y0<space>x0:v0<space>x1:v1<space>...
    -----,
    where y is label, x is index, v is value.

    to a dataset of 0 padding dataet, i.e., not sparse format.
    """

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("SparseDatasetConverter")
    
    def __init__(self, input_filepath, output_filepath,
                 in_sample_delimiter=" ", feature_delimiter=":",
                 out_sample_delimiter=" "):
        """
        """

        # args
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.in_sample_delimiter = in_sample_delimiter
        self.feature_delimiter = feature_delimiter
        self.out_sample_delimiter = out_sample_delimiter
        
        # data info
        self.n_samples = 0
        self.n_dims = 0
        self.feature_indices = set()
        
        pass


    def _loadfile(self, ):
        """
        """

        data = {}
        feature_indices = set()
        with open(self.input_filepath) as fpin:
            reader = csv.reader(fpin, delimiter=self.in_sample_delimiter)
            for i, row in enumerate(reader):
                sample = row[1:]
                x = {}
                for kv in sample:
                    kv_ = kv.split(self.feature_delimiter)
                    idx = int(kv_[0])
                    val = kv_[1]
                    feature_indices.add(idx)
                    x[idx] = val
                    pass
                pass
                y = row[0]
                data[i] = (y, x)
            pass

        # set data info
        self.feature_indices = feature_indices
        self.n_dims = max(feature_indices) + 1
        self.n_samples = len(data)

        return data
        
    def convert(self, ):
        """
        """

        # load file
        data = self._loadfile()
        if os.path.exists(self.output_filepath):
            os.remove(self.output_filepath)
            pass

        self.logger.info("convert start")
        with open(self.output_filepath, "w") as fpout:
            writer = csv.writer(fpout, delimiter=self.out_sample_delimiter)

            for lsample in data.values():  # foreach sample
                y = lsample[0]
                x = lsample[1]
                sample = np.zeros(self.n_dims)
                sample[x.keys()] = x.values()
                l = [y] + sample.tolist()
                writer.writerow(l)
            pass
        self.logger.info("convert finish")
        pass


def main():
    input_filepath = "/home/kzk/datasets/news20/news20.dat"
    output_filepath = "/home/kzk/datasets/news20/news20.csv"
    converter = SparseDatasetConverter(input_filepath, output_filepath)
    converter.convert()

if __name__ == '__main__':
    main()
