'''
Created on Oct 14, 2021

@author: mhindle
'''
import os
import cloudpickle
import mgzip
import psutil

def dumpToPickle(filename, objin, replace=True, threads=psutil.cpu_count(logical = False), blocksize=20*10**8):
    '''
    ## Default compression block size is set to 2000MB/2GB
    '''
    if replace or not (os.path.isfile(filename) and os.path.exists(filename)) :
        with mgzip.open(filename, "wb", thread=threads, blocksize=blocksize) as fout:
            cloudpickle.dump(objin, fout)

