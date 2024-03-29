'''
Created on Sep 6, 2021

@author: mhindle
'''
import numpy as np
import numbers
from typing import Tuple, List, Dict, Union, Set
import itertools
from collections import defaultdict
import pandas as pd
import concurrent.futures
from tqdm import tqdm
import multiprocessing
from copy import deepcopy
import zipfile
import sys

class JointAllellicDistribution(object):

    def __init__(self, chromosome, snp_ordered, chromosome2snp=None, pseudocount = 0.0001, surround_size=1, conditions_index=[0,1,2]):
        self.chromosome2snp = chromosome2snp
        self.pseudocount = pseudocount
        self.surround_size = surround_size
        self.chromosome = chromosome
        self.window_size = (surround_size*2)+1
        self.snp_ordered = [sys.intern(x) for x in snp_ordered if chromosome2snp is None or x in chromosome2snp[chromosome]]
        self.windows = {snp:self.__getWindow(snp) for snp in self.snp_ordered if chromosome2snp is None or snp in chromosome2snp[chromosome]}
        self.state_values = [values for values in list(itertools.product(conditions_index, repeat=self.window_size))]
        #print("init %s keys from %s snps" % (len(self.windows)*len(self.state_values), len(snp_ordered)))
        #state_keys = {[tuple([(k,s) for k,s in zip(window, values)]) for values in self.state_values for snp,window in self.windows.items()]}
        #print("init frequency")
        
        self.snp2frequency: Dict[str,Dict[tuple,int]] = {} #dict.fromkeys(state_keys,pseudocount) for snp in snp_ordered
        for snp in snp_ordered:
            state_keys = [tuple([(k,s) for k,s in zip(self.windows[snp], values)]) for values in self.state_values]
            self.snp2frequency[snp] = dict.fromkeys(state_keys,pseudocount)
        print("init DONE for %s snps!" % len(snp_ordered))
        #self.n_observations: Dict[str,int] = defaultdict(int)
    
    '''
    
    def toDisk(self, file_name_out):
        with zipfile.ZipFile(file_name_out, 'w') as zipped_f:
            zipped_f.writestr("pseudocount", str(self.pseudocount))
            zipped_f.writestr("surround_size", str(self.surround_size))
            zipped_f.writestr("window_size", str(self.window_size))
            zipped_f.writestr("snp_ordered", "\n".join(map(str,self.snp_ordered)))
            zipped_f.writestr("chromosome", str(self.chromosome))
            if self.chromosome2snp is not None and len(self.chromosome2snp) > 0:
                zipped_f.writestr("chromosome2snp", "\n".join([str(k)+"\t"+"\t".join(v) for k,v in self.chromosome2snp.items()]))                  
            zipped_f.writestr("windows", "\n".join([str(k)+"\t"+"\t".join(v) for k,v in self.windows.items()]))
            zipped_f.writestr("state_values", "\n".join(["\t".join(map(str,k)) for k in self.state_values]))
            zipped_f.writestr("frequency", "\n".join(["\t".join([str(s)+":"+str(g) for s,g in k])+"\t"+"{:.9f}".format(v) for k,v in self.frequency.items()]))
    '''
    def getWindow(self, targetSnp):
        '''
        targetSnp is the snp around which to extract the symetric window of +- window_size
        '''
        return(self.windows.get(targetSnp))
    
    def __getWindow(self, targetSnp):
        '''
        private method for determining from scratch see getWindow for preindexed values
        targetSnp is the snp around which to extract the symetric window of +- window_size
        '''
        targetpos = self.snp_ordered.index(targetSnp)
        startpos_snp = targetpos-self.surround_size
        if startpos_snp < 0:
            startpos_snp = 0
        endpos_snp = targetpos+self.surround_size+1
        if endpos_snp >= len(self.snp_ordered):
            endpos_snp = len(self.snp_ordered)-1
        snpWindow = self.snp_ordered[startpos_snp:endpos_snp]
        return(snpWindow)
    
    def getObservations(self, targetSnp):
        window = self.getWindow(targetSnp)
        return(self.getCountTable({x:9 for x in window}, targetSnp))
    
    '''
    turn all unknown 9 values into new observations of 0,1,2
    '''
    def permutTheUnknown(self, all_obs, _found=list()):
        isIncomplete = False
        for i, (snpid, value) in enumerate(all_obs):
            if value == 9:
                isIncomplete = True
                for x in self.conditions_index:
                    clone = deepcopy(all_obs)
                    clone[i] = (snpid,x)
                    if clone not in _found:
                        self.permutTheUnknown(clone, _found=_found)
        if not isIncomplete and all_obs not in _found:
            _found.append(all_obs)
        return(_found)

    '''
    
    '''
    def getCountTable(self, observedstates: dict, targetSnp):
        all_obs = [(sys.intern(snpid),observedstates.get(snpid,9)) for snpid in self.windows[targetSnp]] #if they don't provide a observed state we assume 9
        
        if 9 in [x[1] for x in all_obs]: # this is a bit inefficiant
            queries = self.permutTheUnknown(all_obs) # replace 9s with new observations containing 0,1 or 2
        else:
            queries = [all_obs]
        
        def copypastefunc(x, query):
            return(tuple([(snpid,state) if snpid != targetSnp else (sys.intern(targetSnp),x) for snpid,state in query]))
        
#        for state in [0,1,2] :
#            for query in queries :
#                if copypastefunc(state, query) not in self.frequency.keys():
#                    raise Exception("%s \n %s" % ("_".join(self.windows[targetSnp]),"--".join(query)))
        
        return [np.sum([self.snp2frequency[targetSnp][copypastefunc(state, query)] for query in queries]) for state in [0,1,2]]
    
    @staticmethod
    def countJointFrq(table, mask, column_names: List[str], s_values, pseudocount, sum_inner_count=True):
        '''
        sum_inner_count add the inner windows ie. for snp flanking window size 4 also add observations for 1,2,3
        '''
        results = dict.fromkeys({tuple([(k,s) for k,s in zip(column_names, values)]) for values in s_values})
        column_names = np.array(column_names)
        subset = table[np.all(mask,axis=1),:]
        for values in s_values:
            conditions = list(zip(column_names, values))
            rows_that_meet = np.logical_and.reduce([np.equal(subset[:,column_names == snp],value) for snp,value in conditions])
            state_key = tuple([(k,s) for k,s in zip(column_names, values)])
            obs = np.count_nonzero(rows_that_meet)
            results[state_key] = obs+pseudocount
            if sum_inner_count: # add the inner windows ie. for snp flanking window size 4 also add observations for 1,2,3
                for i in range(int((len(conditions)-2)/2)):
                    rows_that_meet = np.logical_and.reduce([np.equal(subset[:,column_names == snp],value) for snp,value in conditions[i:-i]])
                    obs = np.count_nonzero(rows_that_meet)
                    obs = obs - results[state_key] # remove outer overlap
                    results[state_key] = (obs/(i+1))+results[state_key]
            #if 9 not in values: # only count complete real value arrays
            #    self.n_observations[snp_key] += (obs+self.pseudocount) # this we keep track of how many observations there have been for these three snps
        return(results)
    
    def countJointFrqAll(self, table:pd.DataFrame, mask=None, threads=multiprocessing.cpu_count()):
        '''
        table expect pandas Dataframe with columns as snpids and rows being observations
        mask expect numpy bool matrix but will deal with pandas bool Dataframe
        '''
        if mask is None:
            mask = np.ones(table.shape,dtype=bool)
        elif mask is pd.DataFrame:
            mask = mask.to_numpy(dtype=bool)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            futures = {}
            for snp,window in self.windows.items() :
                futures[executor.submit(self.countJointFrq,
                                table.loc[:,window].to_numpy(dtype=np.uint8, copy=True),
                                mask[:,[x in window for x in table.columns]],
                                window,
                                self.state_values,
                                self.pseudocount)] = snp
            print("waiting on %s queued jobs with %s threads" % (len(futures), threads))
            with tqdm(total=len(futures)) as pbar:
                for future in concurrent.futures.as_completed(futures) :
                    pbar.update(1)
                    e = future.exception()
                    if e is not None:
                        print(repr(e))
                        raise(e)
                    results = future.result()
                    snp = futures[future]
                    self.snp2frequency[snp].update(results)
                    del future
            print("empirical count done")
            
#===============================================================================
# import pickle
# import gzip
# myobj = pickle.load(gzip.open("C:\\Users\\mhindle\\Downloads\\empiricalIndex.idx.gz"))
# 
# print(myobj.pseudocount)
# print(myobj.surround_size)
# print(myobj.window_size)
# print(len(myobj.snp_ordered))
# myobj.toDisk("C:\\Users\\mhindle\\Downloads\\custom_empiricalIndex.zip")
# 
# print("done")
#===============================================================================

