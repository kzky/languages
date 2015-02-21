#!/usr/bin/env python

import glob
import os
import numpy as np
import csv
from collections import defaultdict

# vars
in_dir_path = "/home/kzk/datasets/uci_csv"
out_dir_basepath = "/home/kzk/datasets"

# pre-process
if not os.path.exists("%s/uci_csv_train" % out_dir_basepath):
    os.mkdir("%s/uci_csv_train" % out_dir_basepath)
    pass
if not os.path.exists("%s/uci_csv_test" % out_dir_basepath):
    os.mkdir("%s/uci_csv_test" % out_dir_basepath)
    pass

# process
for data_path in glob.glob("%s/*.csv" % in_dir_path):
    dataname = data_path.split("/")[-1]

    print "processing %s" % dataname
    
    # read data
    idx_samples = defaultdict()
    with open(data_path) as fpin:
        data = csv.reader(fpin, delimiter=" ")
        for idx, row in enumerate(data):
            idx_samples[idx] = row
            pass
        pass

    indices = np.asarray(idx_samples.keys())
    np.random.shuffle(indices)
    num_samples = len(indices)
    
    # write train data
    with open("%s/uci_csv_train/%s" % (out_dir_basepath, dataname), "w") as fpout:
        writer = csv.writer(fpout, delimiter=" ")
        for idx in indices[0:(num_samples / 2)]:
            writer.writerow(idx_samples[idx])
            pass
        pass

    # write test data
    with open("%s/uci_csv_test/%s" % (out_dir_basepath, dataname), "w") as fpout:
        writer = csv.writer(fpout, delimiter=" ")
        for idx in indices[(num_samples / 2):num_samples]:
            writer.writerow(idx_samples[idx])
            pass
        pass
    pass
