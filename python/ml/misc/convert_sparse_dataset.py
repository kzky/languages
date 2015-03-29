#!/usr/bin/env python

from ml.dataset_converter.dataset_converter import SparseDatasetConverter

input_filepath = "/home/kzk/datasets/news20/news20.dat"
output_filepath = "/home/kzk/datasets/news20/news20.csv"
converter = SparseDatasetConverter(input_filepath, output_filepath)
converter.convert()
