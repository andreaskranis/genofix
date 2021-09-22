'''
Created on Sep 21, 2021

@author: mhindle
'''
import os
import gzip

class Stats(object):

    def __init__(self, outputdir=None):
        self.outputdir = outputdir if outputdir is not None else os.getcwd()
        os.makedirs(self.outputdir, exist_ok=True)
        self.pair_stats = gzip.open("%s/pair_stats.txt.gz" % self.outputdir, "wt")
        self.single_stats = gzip.open("%s/single_stats.txt.gz" % self.outputdir, "wt")
    
    def __enter__(self):
        return self

    # ...
    def __exit__(self, exc_type, exc_value, traceback):
        self.pair_stats.close()
        self.single_stats.close()