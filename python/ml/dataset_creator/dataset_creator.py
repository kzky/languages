#!/usr/bin/env python

import os
import shutil


class DatasetCreator(object):
    """
    Create dataest from existing dataset.
    
    """
    
    def __init__(self, ):
        """
        """
        pass

        self.input_paths = None
        

    def create_ssl_datasets(self, ):
        """
        """

        base_output_dirpath = self._create_base_output_dirpath()

        if not os.path.exists(base_output_dirpath):
            os.makedirs(base_output_dirpath)
        else:
            shutil.rmtree(base_output_dirpath)

        for dataset_path in self.input_paths:  # each dataset file
            self.logger.info("processing %s ..." % dataset_path)
            for i in xrange(self.n):
                self.logger.info("creating %d-th" % i)
                self._create_ssl_dataset(base_output_dirpath, dataset_path, i)
                pass
            pass

    def _create_base_output_dirpath(self, ):
        """
        """
        
        pass
            
    def _create_ssl_dataset(self, base_output_dirpath, dataset_path, i):
        """
        
        Arguments:
        - `base_output_dirpath`:
        - `dataset_path`:
        - `i`:
        """
        
        pass
