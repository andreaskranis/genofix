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

class JointAllellicDistribution(object):

    def __init__(self, snp_ordered, chromosome2snp=None, pseudocount = 0.0001, surround_size=1, conditions_index=[0,1,2]):
        self.pseudocount = pseudocount
        self.surround_size = surround_size
        self.window_size = (surround_size*2)+1
        self.snp_ordered = snp_ordered
        self.chromosome2snp = chromosome2snp
        self.windows = {snp:self.getWindow(snp) for snp in self.snp_ordered}
        self.state_values = [values for values in list(itertools.product(conditions_index, repeat=self.window_size))]
        print("init %s keys from %s snps" % (len(self.windows)*len(self.state_values), len(snp_ordered)))
        print()
        state_keys = set([tuple([(k,s) for k,s in zip(window, values)]) for values in self.state_values for window in self.windows.values()])
        print("init frequency")
        self.frequency: Dict[str,int] = dict.fromkeys(state_keys)
        print("init DONE!")
        #self.n_observations: Dict[str,int] = defaultdict(int)
    
    def getWindow(self, targetSnp):
        '''
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
        if self.chromosome2snp is not None:
            targetchr = self.chromosome2snp[targetSnp]
            return([snpId for snpId in snpWindow if self.chromosome2snp[snpId] == targetchr])
        return(snpWindow)
    
    def getObservations(self, targetSnp):
        window = self.getWindow(targetSnp)
        return(self.getCountTable({x:9 for x in window}, targetSnp))
    
    def getCountTable(self, observedstates: dict, targetSnp):
        all_obs = [(snpid,observedstates[snpid]) for snpid in self.windows[targetSnp]]
        
        # we essentially create all possible queries for all 9 states and then restrict list to known states (recursion is probubly faster than this)
        if 9 in [x[1] for x in all_obs]:
            queries = [tuple([(k,s) for k,s in zip(observedstates.keys(), values)]) for values in self.state_values]
            for i, (snp, state) in enumerate(all_obs):
                if state != 9:
                    queries = [query for query in queries if query[i] == (snp,state)]
        else:
            queries = [all_obs]

        def copypastefunc(x, query):
            return(tuple([(snpid,state) if snpid != targetSnp else (targetSnp,x) for snpid,state in query]))
        
        return [np.sum([self.frequency[copypastefunc(state, query)] for query in queries]) for state in [0,1,2]]
    
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
                    self.frequency.update(results)
            print("empirical count done")
