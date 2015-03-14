#!/usr/bin/env python

from ml.dataset_creator.ssl_dataset_creator import SSLRateDatasetCreator
import sys


input_paths = [
    "/home/kzk/datasets/uci_csv/activity.csv",
    "/home/kzk/datasets/uci_csv/ad.csv",
    "/home/kzk/datasets/uci_csv/adult.csv",
    "/home/kzk/datasets/uci_csv/breast_cancer.csv",
    "/home/kzk/datasets/uci_csv/car.csv",
    "/home/kzk/datasets/uci_csv/credit.csv",
    "/home/kzk/datasets/uci_csv/gisette.csv",
    "/home/kzk/datasets/uci_csv/glass.csv",
    "/home/kzk/datasets/uci_csv/haberman.csv",
    "/home/kzk/datasets/uci_csv/ionosphere.csv",
    "/home/kzk/datasets/uci_csv/iris.csv",
    "/home/kzk/datasets/uci_csv/isolet.csv",
    "/home/kzk/datasets/uci_csv/liver.csv",
    "/home/kzk/datasets/uci_csv/magicGamaTelescope.csv",
    "/home/kzk/datasets/uci_csv/mammographic.csv",
    "/home/kzk/datasets/uci_csv/parkinsons.csv",
    "/home/kzk/datasets/uci_csv/pima.csv",
    # "/home/kzk/datasets/uci_csv/sonar.csv",  # data is too small
    "/home/kzk/datasets/uci_csv/spam.csv",
    "/home/kzk/datasets/uci_csv/spect.csv",
    "/home/kzk/datasets/uci_csv/spectf.csv",
    "/home/kzk/datasets/uci_csv/transfusion.csv",
    "/home/kzk/datasets/uci_csv/usps.csv",
    "/home/kzk/datasets/uci_csv/wdbcMB.csv",
    "/home/kzk/datasets/uci_csv/wine.csv",
    "/home/kzk/datasets/uci_csv/wineq_red.csv",
    "/home/kzk/datasets/uci_csv/wineq_white.csv",
    "/home/kzk/datasets/uci_csv/wpbcRN.csv",
    "/home/kzk/datasets/uci_csv/yeast.csv",
]

output_prefix_dirpath = "/home/kzk/datasets/uci_csv_ssl"

n = int(sys.argv[1])
l_rate = float(sys.argv[2])
u_rate = float(sys.argv[3])
v_rate = float(sys.argv[4])
data_type = str(sys.argv[5])

creator = SSLRateDatasetCreator(
    input_paths,
    output_prefix_dirpath,
    n=n,
    l_rate=l_rate,
    u_rate=u_rate,
    v_rate=v_rate,
    delimiter=" ",
    data_type=data_type
)

creator.create_ssl_datasets()

