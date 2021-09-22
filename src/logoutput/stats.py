'''
Created on Sep 21, 2021

@author: mhindle
'''
import os

class Stats(object):

    def __init__(self, outputdir=None):
        self.outputdir = outputdir if outputdir is not None else os.getcwd()
        os.makedirs(self.outputdir, exist_ok=True)
        self.pair_stats = open("%s/pair_stats.txt" % self.outputdir, "wt")
        self.single_stats = open("%s/single_stats.txt" % self.outputdir, "wt")
    
    def __enter__(self):
        return self

    # ...
    def __exit__(self, exc_type, exc_value, traceback):
        self.pair_stats.close()
        self.single_stats.close()