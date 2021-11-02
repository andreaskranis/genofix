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

class JointAllellicDistribution(object):

    def __init__(self, snp_ordered, chromosome2snp=None, pseudocount = 1, surround_size=1):
        self.pseudocount = pseudocount
        self.frequency: Dict[str,int] = {}
        #self.n_observations: Dict[str,int] = defaultdict(int)
        self.surround_size = surround_size
        self.window_size = (surround_size*2)+1
        self.snp_ordered = snp_ordered
        self.chromosome2snp = chromosome2snp
    
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
        return(self.n_observations[":".join(self.getWindow(targetSnp))])
    
    def getCountTable(self, observedstates: dict, targetSnp):
        all_obs = [(snpid,observedstates[snpid]) for snpid in self.getWindow(targetSnp)]
        
        def copypastefunc(x):
            return([str(snpid)+"_"+str(state) if snpid != targetSnp else str(targetSnp)+"_"+str(x) for snpid,state in all_obs])
        
        for query in list(map(copypastefunc, [0,1,2])):
            yield self.frequency[":".join(query)]
    
    def countJointFrq(self, table, mask, column_names: List[str], conditions_index=[0,1,2,9]):
        column_names = np.array(column_names)
        subset = table[np.all(mask,axis=1),:]
        for values in list(itertools.product(conditions_index, repeat=self.window_size)):
            conditions = list(zip(column_names, values))
            nine_truth = np.ones((subset.shape[0],1), dtype=bool)
            rows_that_meet = np.logical_and.reduce([nine_truth if value == 9 else np.equal(subset[:,column_names == snp],value) for snp,value in conditions])
            state_key = ":".join([str(k)+"_"+str(s) for k,s in zip(column_names, values)])
            #snp_key = ":".join(column_names)
            obs = np.count_nonzero(rows_that_meet)
            self.frequency[state_key] = obs+self.pseudocount
            #if 9 not in values: # only count complete real value arrays
            #    self.n_observations[snp_key] += (obs+self.pseudocount) # this we keep track of how many observations there have been for these three snps
    
    def countJointFrqAll(self, table:pd.DataFrame, mask=None):
        '''
        table expect pandas Dataframe with columns as snpids and rows being observations
        mask expect numpy bool matrix but will deal with pandas bool Dataframe
        '''
        if mask is None:
            mask = np.ones(table.shape,dtype=bool)
        elif mask is pd.DataFrame:
            mask = mask.to_numpy(dtype=bool)
        for targetSnp in self.snp_ordered:
            snp_window = self.getWindow(targetSnp)
            indexofsnps = [x in snp_window for x in table.columns]
            self.countJointFrq(table.loc[:,snp_window].to_numpy(dtype=int), mask[:,indexofsnps], snp_window)
