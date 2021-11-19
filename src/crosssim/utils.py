'''
Created on Apr 23, 2021

@author: mhindle
'''
import os, traceback
import argparse
import gzip
import numpy as np
from numpy.typing import ArrayLike
from typing import List, Tuple, Dict, Union
from sortedcontainers import SortedDict, SortedList
import psutil
import mgzip
import cloudpickle
from collections import Counter

def scale_number(unscaled, to_min, to_max, from_min, from_max):
    return (to_max-to_min)*(unscaled-from_min)/(from_max-from_min)+to_min     

def scale_list(l, to_min, to_max, from_min, from_max):
    return [scale_number(i, to_min, to_max, from_min, from_max) for i in l]

def openfile(filename:str, mode:str='r') :
    '''
    tests for gzip ending and uses appropriate
    '''
    if filename.endswith('.gz'):
        return gzip.open(filename, mode) 
    else:
        return open(filename, mode)

def find_runs(x:ArrayLike, ignorevalue=9) -> Tuple:
    """
    Find runs of consecutive items in an array.
    code from https://gist.github.com/alimanfoo
    x: a numpy array to find repeat runs
    """
    x = np.asanyarray(x) # ensure array
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]
    if n == 0: # handle empty array
        return np.array([]), np.array([]), np.array([])
    else:
        loc_run_start = np.empty(n, dtype=bool) # find run starts
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        #print(loc_run_start)
        #if ignorevalue is not None:
            #ofvalue = x == ignorevalue
            #loc_run_start[np.logical_or(ofvalue,np.insert(np.delete(ofvalue, -1), 0, False))] = False
        #print(loc_run_start)
        run_starts = np.nonzero(loc_run_start)[0]
        run_values = x[loc_run_start] # find run values
        run_lengths = np.diff(np.append(run_starts, n)) # find run lengths
        
        def eliminateMissing(run_values, run_starts, run_lengths):
            for i, value in enumerate(run_values):
                if value == 9:
                    if i == 0:
                        #a start
                        run_starts[i+1] = run_starts[i]
                        run_lengths[i+1] = run_lengths[i]+run_lengths[i+1]
                        run_lengths = np.delete(run_lengths, i)
                        run_values = np.delete(run_values, i)
                        run_starts = np.delete(run_starts, i)
                    elif i == len(run_values)-1:
                        #end
                        run_lengths[i-1] = run_lengths[i]+run_lengths[i-1]
                        run_lengths = np.delete(run_lengths, i)
                        run_values = np.delete(run_values, i)
                        run_starts = np.delete(run_starts, i)
                    elif run_values[i-1] == run_values[i+1]:
                        #merge
                        run_lengths[i-1] = run_lengths[i-1]+run_lengths[i]+run_lengths[i+1]
                        run_lengths = np.delete(run_lengths, [i,i+1])
                        run_values = np.delete(run_values, [i,i+1])
                        run_starts = np.delete(run_starts, [i,i+1])
                    elif run_values[i-1] != run_values[i+1]:
                        # overlap
                        tmplenmissing=run_lengths[i]
                        run_lengths[i-1] = tmplenmissing+run_lengths[i-1]
                        run_lengths[i+1] = tmplenmissing+run_lengths[i+1]
                        run_starts[i+1] = run_starts[i+1]-tmplenmissing
                        run_lengths = np.delete(run_lengths, i)
                        run_values = np.delete(run_values, i)
                        run_starts = np.delete(run_starts, i)
                    return(run_values, run_starts, run_lengths, True)
            return(run_values, run_starts, run_lengths, False)
        
        while True:
            run_values, run_starts, run_lengths, changed = eliminateMissing(run_values, run_starts, run_lengths)
            if not changed:
                break
        return run_values, run_starts, run_lengths

#print(find_runs([9,9,9,0,0,0,0,1,1,1,1,1], ignorevalue=9))
#print(find_runs([0,0,0,0,9,9,9,9,1,1,1,1,1], ignorevalue=9))
#print(find_runs([0,0,0,0,9,9,9,9,0,1,1,1,1,1], ignorevalue=9))
#print(find_runs([0,0,0,0,9,9,9,9,0,1,1,1,1,1,9,9,9], ignorevalue=9))


def predictcrosspoints(p:ArrayLike, pp:ArrayLike, pm:ArrayLike, ignorevalue=9, paternal_strand=0,maternal_strand=1) -> ArrayLike:
    '''
    p: parental haplotype of indiv
    pm: parental maternal hap
    pp: parental paternal hap
    '''
    # ensure array
    p = np.asanyarray(p) 
    pp = np.asanyarray(pp) 
    pm = np.asanyarray(pm) 
    
    hapsource = np.full(len(p), -9)
    hapsource[np.where(np.equal(pm,p) & np.not_equal(pp,p))] = maternal_strand
    hapsource[np.where(np.equal(pp,p) & np.not_equal(pm,p))] = paternal_strand
    runvalue, runstart, runlength = find_runs(hapsource,ignorevalue=ignorevalue)
    for i, value in enumerate(runvalue) :
        if value == -9 and len(runvalue) > 1: # i.e. ren(runvalue) == 1 if their are no hetroz stretches in the chromosome
            if i > 0 and i < (len(runvalue)-1):
                if runvalue[i-1] == runvalue[i+1]:
                    hapsource[runstart[i]:runstart[i]+runlength[i]] = runvalue[i-1]
                else :
                    hapsource[runstart[i]:runstart[i]+runlength[i]] = 3
            elif i == 0 :
                hapsource[runstart[i]:runstart[i]+runlength[i]] = runvalue[i+1]
            elif i == (len(runvalue)-1) :
                hapsource[runstart[i]:runstart[i]+runlength[i]] = runvalue[i-1]
    return(hapsource)





print(predictcrosspoints([0,1,0,1,0,0,0,1,0,1,0,1,1,1,1,1,0,1,0,1], 
                   [0,1,0,1,0,0,0,1,0,1,0,1,1,1,0,0,0,1,1,1], 
                   [0,1,1,0,0,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1], 
                   ignorevalue=9, paternal_strand=0,maternal_strand=1))

def indexSnps(snpdetails: Dict[str, Dict[str, str]], snpIds: List[str]) :# -> Tuple[Dict[int,SortedList[Tuple[str,int]]], Dict[int, str], Dict[int, Dict[str, Union[int, float]]]] :
    '''
    index snpdetails
    This is a tad complex return! Should really return an object
    ''' 
    chr2snp2pos: Dict[int,SortedList[Tuple[str,int]]] = SortedDict()
    snp2chromosome: Dict[str,int] = dict(zip(snpIds, [int(snpdetails[x]["chr"]) for x in snpIds]))
    gm: Dict[int, Dict[str, Union[int, float]]] = {}
    for snpdetail in [snpdetails[x] for x in snpIds] :
        #"chr", "id", "cm","bp"
        morgans = float(snpdetail["cm"])/100
        chrom = int(snpdetail["chr"])
        pos = int(snpdetail["bp"])
        if chrom not in gm:
            gm[chrom] = {'pos':0,'nvars':0, 'morgans':0}
        elif morgans > gm[chrom]['morgans']:
            gm[chrom]['morgans'] = float(morgans)
            gm[chrom]['pos'] = int(pos)
        
        if chrom not in chr2snp2pos:
            chr2snp2pos[chrom] = SortedList(key=lambda x: x[1]) # sorts on position
        chr2snp2pos[chrom].add((snpdetail["id"],pos))
        gm[chrom]['nvars'] += 1
    return(chr2snp2pos, snp2chromosome, gm)

def addPaddedArrays(A, B):
    ma,na = A.shape
    mb,nb = B.shape
    m,n = max(ma,mb) , max(na,nb)
    
    if ma != m or na != n:
        newA = np.zeros((m,n),A.dtype)
        newA[:ma,:na]=A
    else:
        newA = A
    
    if mb != m or nb != n:
        newB = np.zeros((m,n),B.dtype)
        newB[:mb,:nb]=B
    else:
        newB = B
    
    return(np.add(newA,newB))

def dumpToPickle(filename, objin, replace=True, threads=psutil.cpu_count(logical = False), blocksize=20*10**8):
    '''
    ## Default compression block size is set to 2000MB/2GB
    '''
    if replace or not (os.path.isfile(filename) and os.path.exists(filename)) :
        with mgzip.open(filename, "wb", thread=threads, blocksize=blocksize) as fout:
            cloudpickle.dump(objin, fout)

def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Statistics(object):                    

    def __init__(self):
        self.length_overlap_actuals: Dict[int,int] = Counter()
        self.n_actual: List[int] = list()
        self.n_detected: List[int]  = list()
        self.n_actual2predicted: List[int] = list()
        self.n_predicte2actual: List[int]  = list()
        self.missedactuals: List[Tuple[int]] = list()
    
    def addCrossoverStatistics(self, crossactual: List[List[int]], crossregion):
        #print("actuals %s  %s" % (type(crossactual), crossactual))
        #print("predicted %s  %s" % (type(crossregion), list(zip(crossregion))))
        for actuals,(start_r, len_r) in [(a,b) for a,b in zip(crossactual, crossregion) if a is not None and b is not None]:  
            ends_r = start_r+len_r
            self.n_actual.append(len(actuals))
            self.n_detected.append(len(start_r))
            withinpredictedregion = [np.logical_and(a >= start_r,a <= ends_r) for a in actuals]
            self.missedactuals.extend(actuals[[sum(x) == 0 for x in withinpredictedregion]])
            self.n_actual2predicted.extend([sum(x) for x in withinpredictedregion])
            regionhasactualxover = [[s <= a and e >= a for a in actuals] for s,e in zip(start_r, ends_r)]
            self.n_predicte2actual.extend([sum(x) for x in regionhasactualxover])
            for x in withinpredictedregion :
                self.length_overlap_actuals.update(list(len_r[x]))
        #print("missed %s " % self.missedactuals)
        
    def mergeStatistics(self, stats):
        self.length_overlap_actuals.update(stats.length_overlap_actuals)
        self.n_actual.extend(stats.n_actual)
        self.n_detected.extend(stats.n_detected)
        self.n_actual2predicted.extend(stats.n_actual2predicted)
        self.n_predicte2actual.extend(stats.n_predicte2actual)
        
# import sys
# import pickle, gzip
# sys.path.append('/home/mhindle/eclipse-workspace/qsim_mmh/crosssim')
# actual = pickle.load(gzip.open('/mnt/md0/mhindle/gensys/chr3sim9/sim_crossovers/pickle/chr2/sim_actual_4004156851058.pickle.gz'))
# regions = pickle.load(gzip.open('/mnt/md0/mhindle/gensys/chr3sim9/sim_crossovers/pickle/chr2/sim_ambigregions_4004156851058.pickle.gz'))
# crossactual_pat = [sim[0] for sim in actual]
# crossregion_pat = [sim[0] for sim in regions]
#
# stats = Statistics()
# stats.addCrossoverStatistics(crossactual_pat, crossregion_pat)
#
# dumpToPickle("/mnt/md0/mhindle/gensys/chr3sim9b/sim_crossovers/population_sim/chr2/perfomstats_file.pickle.gz", stats)


