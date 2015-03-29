#!/usr/bin/env python

from ml.experimentor.ssl_experimentor import SSLRateDatesetEvaluator

from ml.ssl.hpfssl import HPFSSLClassifier
from ml.ssl.laprls import LapRLSClassifier
from ml.ssl.svm import LSVMClassifier
from ml.ssl.rvm import RVMClassifier

# arguemnts for ssl
base_paths = [
    "/home/kzk/datasets/uci_csv_ssl_urate_fixed_1_50_1_48",
]
output_path = "/home/kzk/experiment/final_paper_2015/results_test/.pkl"
classifiers_info = {
    "hpfssl": {
        "classifier": HPFSSLClassifier(),
        "param_grid": [{"max_itr": 50, "threshold": 1e-4, "learn_type": "batch"}],
    },

    "rvm": {
        "classifier": RVMClassifier(),
        "param_grid": [{"max_itr": 50, "threshold": 1e-4,
                        "learn_type": "batch", "alpha_threshold": 1e-24}],
    },
    
    "svm": {
        "classifier": LSVMClassifier(),
        "param_grid": [
            {"C": 1e-3}, {"C": 1e-2}, {"C": 1e-1}, {"C": 1},
            {"C": 1e1}, {"C": 1e2}, {"C": 1e3}],
    },

    "laprls": {
        "classifier": LapRLSClassifier(),
        "param_grid": [  # too slowa
            {"lam": 1e-0, "gamma_s": 1e-0, "normalized": True, "kernel": "rbf"},
            {"lam": 1e-3, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf"},
            {"lam": 1e-3, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf"},
            {"lam": 1e-3, "gamma_s": 1e0, "normalized": True, "kernel": "rbf"},
            {"lam": 1e-3, "gamma_s": 1e1, "normalized": True, "kernel": "rbf"},
            {"lam": 1e-3, "gamma_s": 1e2, "normalized": True, "kernel": "rbf"},
            {"lam": 1e-3, "gamma_s": 1e3, "normalized": True, "kernel": "rbf"},
             
            {"lam": 1e-2, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf"},
            {"lam": 1e-2, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf"},
            {"lam": 1e-2, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf"},
            {"lam": 1e-2, "gamma_s": 1e0, "normalized": True, "kernel": "rbf"},
            {"lam": 1e-2, "gamma_s": 1e1, "normalized": True, "kernel": "rbf"},
            {"lam": 1e-2, "gamma_s": 1e2, "normalized": True, "kernel": "rbf"},
            {"lam": 1e-2, "gamma_s": 1e3, "normalized": True, "kernel": "rbf"},
             
            {"lam": 1e-1, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf"},
            {"lam": 1e-1, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf"},
            {"lam": 1e-1, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf"},
            {"lam": 1e-1, "gamma_s": 1e0, "normalized": True, "kernel": "rbf"},
            {"lam": 1e-1, "gamma_s": 1e1, "normalized": True, "kernel": "rbf"},
            {"lam": 1e-1, "gamma_s": 1e2, "normalized": True, "kernel": "rbf"},
            {"lam": 1e-1, "gamma_s": 1e3, "normalized": True, "kernel": "rbf"},
             
            {"lam": 1e0, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf"},
            {"lam": 1e0, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf"},
            {"lam": 1e0, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf"},
            {"lam": 1e0, "gamma_s": 1e0, "normalized": True, "kernel": "rbf"},
            {"lam": 1e0, "gamma_s": 1e1, "normalized": True, "kernel": "rbf"},
            {"lam": 1e0, "gamma_s": 1e2, "normalized": True, "kernel": "rbf"},
            {"lam": 1e0, "gamma_s": 1e3, "normalized": True, "kernel": "rbf"},
             
            {"lam": 1e1, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf"},
            {"lam": 1e1, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf"},
            {"lam": 1e1, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf"},
            {"lam": 1e1, "gamma_s": 1e0, "normalized": True, "kernel": "rbf"},
            {"lam": 1e1, "gamma_s": 1e1, "normalized": True, "kernel": "rbf"},
            {"lam": 1e1, "gamma_s": 1e2, "normalized": True, "kernel": "rbf"},
            {"lam": 1e1, "gamma_s": 1e3, "normalized": True, "kernel": "rbf"},
             
            {"lam": 1e2, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf"},
            {"lam": 1e2, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf"},
            {"lam": 1e2, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf"},
            {"lam": 1e2, "gamma_s": 1e0, "normalized": True, "kernel": "rbf"},
            {"lam": 1e2, "gamma_s": 1e1, "normalized": True, "kernel": "rbf"},
            {"lam": 1e2, "gamma_s": 1e2, "normalized": True, "kernel": "rbf"},
            {"lam": 1e2, "gamma_s": 1e3, "normalized": True, "kernel": "rbf"},
             
            {"lam": 1e3, "gamma_s": 1e-3, "normalized": True, "kernel": "rbf"},
            {"lam": 1e3, "gamma_s": 1e-2, "normalized": True, "kernel": "rbf"},
            {"lam": 1e3, "gamma_s": 1e-1, "normalized": True, "kernel": "rbf"},
            {"lam": 1e3, "gamma_s": 1e0, "normalized": True, "kernel": "rbf"},
            {"lam": 1e3, "gamma_s": 1e1, "normalized": True, "kernel": "rbf"},
            {"lam": 1e3, "gamma_s": 1e2, "normalized": True, "kernel": "rbf"},
            {"lam": 1e3, "gamma_s": 1e3, "normalized": True, "kernel": "rbf"},
            
        ],
    },
}
blocked_datasets = [
#    "adult",
#    "magicGamaTelescope",
]


experimentor = SSLRateDatesetEvaluator(
    base_paths,
    output_path,
    classifiers_info,
    blocked_datasets
)

experimentor.run()

