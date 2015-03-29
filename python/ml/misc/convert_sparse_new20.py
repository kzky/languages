#!/usr/bin/env python

import os
import glob
import shutil
import gc

from ml.dataset_converter.dataset_converter import SparseDatasetConverter


input_dirpaths = [
    "/home/kzk/datasets/sparse_news20_ssl_lrate_fixed_1_30_1_98/news20",
    "/home/kzk/datasets/sparse_news20_ssl_lrate_fixed_1_40_1_98/news20",
    "/home/kzk/datasets/sparse_news20_ssl_lrate_fixed_1_50_1_98/news20",
    "/home/kzk/datasets/sparse_news20_ssl_lrate_fixed_1_60_1_98/news20",
    "/home/kzk/datasets/sparse_news20_ssl_lrate_fixed_1_70_1_98/news20",
    "/home/kzk/datasets/sparse_news20_ssl_lrate_fixed_1_80_1_98/news20",
]

for input_dirpath in input_dirpaths:
    input_filepaths = glob.glob("%s/*" % input_dirpath)
    input_filepaths.sort()

    for input_filepath in input_filepaths:

        print "processing %s" % (input_filepath)
        
        output_dirpath_ = os.path.dirname(input_filepath)
        output_dirpath = output_dirpath_.replace("sparse_", "")
        
        if not os.path.exists(output_dirpath):
            os.makedirs(output_dirpath)

        filename = os.path.basename(input_filepath)
        output_filepath = "%s/%s" % (output_dirpath, filename)
        converter = SparseDatasetConverter(input_filepath, output_filepath)

        converter.convert()

        gc.collect()
        pass
    pass


