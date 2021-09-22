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
        self.frequency: Dict[Tuple[str,int],Dict[Tuple[str,int],Dict[Tuple[str,int],int]]] = dict()
        self.n_observations: Dict[Tuple[str,str,str]] = defaultdict(int)
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
    
    def getCountTable(self, observedstates, targetSnp):
        all_obs = [(snpid,observedstates[snpid]) for snpid in self.getWindow(targetSnp)]
        
        def copypastefunc(x):
            return([(snpid,state) if snpid != targetSnp else (targetSnp, x) for snpid,state in all_obs])
        
        for state, query in enumerate(list(map(copypastefunc, [0,1,2]))):
            #print("%s == %s" % (state, query))
            workinghash = self.frequency
            for item in query:
                workinghash = workinghash[item]
            if "obs" in workinghash:
                yield workinghash["obs"] #it should be the result
            else:
                print("query %s" % query)
                print("first %s" % self.frequency[query[0]])
                print("workinghash %s" % workinghash)
                print("item %s" % "_".join(map(str,item)))
                raise Exception("incomplete traversal of nested hash: final %s state %s" % (workinghash, state))

    def countJointFrq(self, table, mask, column_names: List[str]):
        column_names = np.array(column_names)
        subset = table[np.all(mask,axis=1),:]
        for values in list(itertools.product([0,1,2], repeat=self.window_size)):
            conditions = list(zip(column_names, values))
            rows_that_meet = np.logical_and.reduce([np.equal(subset[:,column_names == snp],value) for snp,value in conditions])
            keys = list(zip(column_names, values))
            obs = np.count_nonzero(rows_that_meet)
            self.recurse_set_dict(self.frequency, keys, obs)
            self.n_observations[tuple(column_names)] += (obs+self.pseudocount) # this we keep track of how many observations there have been for these three snps
    
    def recurse_set_dict(self, d, queue, value):
        f = queue.pop(0)
        if len(queue) > 0:
            if f not in d:
                d[f] = dict()
            self.recurse_set_dict(d[f], queue, value)
        else:
            if f not in d:
                d[f] = dict()
            if "obs" not in d[f]:
                d[f]["obs"] = value+self.pseudocount # we record the observations for this state combo
            elif d[f]["obs"] != value+self.pseudocount:
                raise Exception("overwriting value %s with %s " % (d[f]["obs"], value))
        
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
