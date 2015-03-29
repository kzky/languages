#!/usr/bin/env python

from ml.dataset_creator.ssl_sparse_dataset_creator import SSLRateSparseDatasetCreator
import sys


input_paths = [
    "/home/kzk/datasets/news20/news20.dat",
]

# "/home/kzk/datasets/uci_csv_ssl_lrate_fixed"
# "/home/kzk/datasets/uci_csv_ssl_urate_fixed"

prefix_output_dirpath = str(sys.argv[1])
n = int(sys.argv[2])
l_rate = float(sys.argv[3])
u_rate = float(sys.argv[4])
v_rate = float(sys.argv[5])
data_type = str(sys.argv[6])

creator = SSLRateSparseDatasetCreator(
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

