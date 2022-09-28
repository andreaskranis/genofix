#import sys,os
#sys.path.append(os.getcwd())

'''
Created on Jul 26, 2021

@author: mhindle
'''
from collections import OrderedDict, defaultdict
from typing import Tuple, List, Dict, Union, Set

from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.models import BayesianModel
from pgmpy.readwrite import BIFReader, BIFWriter
from pgmpy.sampling import BayesianModelSampling
from pgmpy.utils import get_example_model 

import numpy as np
from pedigree.pedigree_dag import PedigreeDAG


    #probability distribution for 0,1,2
_default_alleleprobs: List[List[float]] = [[0.25], [0.5], [0.25]]

###################################### sire 0  0   0   1   1   1  2  2  2
####################################### dam 0  1   2   0   1   2  0  1  2
_default_mendelprobs: List[List[float]] = [[1, .5, 0, .5, .25, 0, 0, 0,  0], #0
                                           [0, .5, 1, .5, .5, .5, 1, .5, 0], #1
                                           [0,  0, 0,  0, .25,.5, 0, .5, 1]] #2

_default_mendelprobs_np = np.array(_default_mendelprobs)

tinyamount = 0.0001
_default_mendelprobs_c = np.add(_default_mendelprobs,tinyamount)/(tinyamount+1) #bound away from zero

#===============================================================================
# def generate_probs_kids(kidstates, oneparentstates):
#     '''
#      sire 0  0   0   1   1   1  2  2  2
#      dam 0  1   2   0   1   2  0  1  2
#     '''
#     structA = [0,0,0,1,1,1,2,2,2]
#     structB = [0,1,2,0,1,2,0,1,2]
#     
#     kidstates = np.asarray(kidstates)
#     oneparentstates = np.asarray(oneparentstates)
#     include = [x in [0,1,2] for x in kidstates]
#     
#     if np.count_nonzero(include) > 0:
#         prob_table = np.array([_default_mendelprobs[x] for x in kidstates[include]])
#         for i,parent in enumerate(oneparentstates[include]):
#             if parent is not None and parent in [0,1,2]:
#                 prob_table[i,np.not_equal(structA,parent)] = 0
#             rowsum = np.sum(prob_table[i,])
#             if rowsum > 0:
#                 prob_table[i,] = np.divide(prob_table[i,],rowsum)
#         prob_result = [np.nansum(prob_table[:, np.equal(structB,x)]) for x in [0,1,2]]
#         if sum(prob_result) > 0:
#             prob_result = prob_result/sum(prob_result)
#         return(prob_result)
#     return[1/3,1/3,1/3]
#===============================================================================

def generate_probs_kids(kidstates, oneparentstates):
    '''
     sire 0  0   0   1   1   1  2  2  2
     dam 0  1   2   0   1   2  0  1  2
    '''
    structA = [0,0,0,1,1,1,2,2,2]
    structB = [0,1,2,0,1,2,0,1,2]
    
    kidstates = np.asarray(kidstates)
    oneparentstates = np.asarray(oneparentstates)
    include = [x in [0,1,2] for x in kidstates]
    
    if np.count_nonzero(include) > 0:
        prob_table = np.array([_default_mendelprobs[x] for x in kidstates[include]])
        for i,parent in enumerate(oneparentstates[include]):
            if parent is not None and parent in [0,1,2]:
                prob_table[i,np.not_equal(structA,parent)] = 0
            rowsum = np.sum(prob_table[i,])
            if rowsum > 0:
                prob_table[i,] = np.divide(prob_table[i,],rowsum)
        prob_result = [np.nanmean(prob_table[:, np.equal(structB,x)]) for x in [0,1,2]]
        if sum(prob_result) > 0:
            prob_result = prob_result/sum(prob_result)
        return(prob_result)
    return[1/3,1/3,1/3]

#print(generate_probs_kids([1,0,1,0,1],[2,2,2,2,2]))

def generate_probs_differences_kids(kidstates, oneparentstates, observedstate):
    '''
     sire 0  0   0   1   1   1  2  2  2
     dam 0  1   2   0   1   2  0  1  2
    '''
    structA = [0,0,0,1,1,1,2,2,2]
    structB = [0,1,2,0,1,2,0,1,2]
    statprobtable = np.zeros((len(kidstates),3), np.float16)
    
    kidstates = np.asarray(kidstates)
    oneparentstates = np.asarray(oneparentstates)
    include = [x in [0,1,2] for x in kidstates]
    
    if np.count_nonzero(include) > 0:
        prob_table = np.array([_default_mendelprobs[x] for x in kidstates[include]])
        for i,parent in enumerate(oneparentstates[include]):
            if parent is not None and parent in [0,1,2]:
                prob_table[i,np.not_equal(structA,parent)] = 0
            rowsum = np.sum(prob_table[i,])
            if rowsum > 0:
                prob_table[i,] = np.divide(prob_table[i,],rowsum)
            statprobtable[i,:] = [np.nansum(prob_table[i, np.equal(structB,x)]) for x in [0,1,2]]
        return([np.nanmax(row)-row[observedstate] for row in statprobtable])
    return([])
        
def generate_probs_differences_parent(parent1, parent2, observedstate):
    '''
     sire 0  0   0   1   1   1  2  2  2
     dam 0  1   2   0   1   2  0  1  2
    '''
    structA = [0,0,0,1,1,1,2,2,2]
    structB = [0,1,2,0,1,2,0,1,2]
    if parent1 in [0,1,2] and parent2 in [0,1,2]:
        column = [x[0] for x in _default_mendelprobs_np[:, np.logical_and(np.equal(structA,parent1), np.equal(structB,parent2))].tolist()]
    elif parent1 in [0,1,2]:
        column = np.nanmean(_default_mendelprobs_np[:, np.equal(structA,parent1)], 1, dtype=float)
    elif parent2 in [0,1,2]:
        column = np.nanmean(_default_mendelprobs_np[:, np.equal(structB,parent2)], 1, dtype=float)
    else:
        return 0
    return(np.nanmax(column)-column[observedstate])

#===============================================================================
# print(generate_probs_differences_parent(2,2,1))
# print(generate_probs_differences_parent(1,1,0))
# print(generate_probs_differences_parent(2,9,0))
# print(generate_probs_differences_parent(9,2,0))
# print(generate_probs_differences_parent(9,9,1))
# 
# print(generate_probs_kids([1,0,1,0,1],[2,2,2,2,2]))
#===============================================================================

class BayesPedigreeNetworkModel(BayesianModel):
    '''
    classdocs
    '''

    def __init__(self, ebunch=None, latents=set()):
        super(BayesianModel, self).__init__(ebunch=ebunch, latents=latents)
        self.cpds = []
        self.cardinalities = defaultdict(int)
        self.blanket = ebunch
    
    def copy(self):
        model_copy = BayesPedigreeNetworkModel()
        model_copy.add_nodes_from(self.nodes())
        model_copy.add_edges_from(self.edges())
        if self.cpds:
            model_copy.add_cpds(*[cpd.copy() for cpd in self.cpds])
        model_copy.latents = self.latents
        model_copy.blanket = self.blanket
        return model_copy
    
    def adjustAlleleprobs(self, newprobs: List[List[float]]):
        for cpd in self.cpds:
            if np.array_equal([[x] for x in cpd.values],_default_alleleprobs):
                cpd.values = np.array([[x] for x in newprobs])
    
    
    @staticmethod
    def generateModel(pedigree: PedigreeDAG,
                      alleleprobs: List[List[float]]=_default_alleleprobs, 
                      mendelprobs: List[List[float]]=_default_mendelprobs) :
        #print("generate for %s sire %s dam %s" % (kid, sire, dam))
        id2CPD: Dict[str,TabularCPD]= OrderedDict()
        graphedges: Set[Tuple[str,str]]=set()
        
        graphedges.update(set([(str(sire),str(kid)) for kid, sire in pedigree.kid2sire.items()]))
        graphedges.update(set([(str(dam),str(kid)) for kid, dam in pedigree.kid2dam.items()]))
        
        malefounders = [x for x in pedigree.males if x not in pedigree.kid2sire and x not in pedigree.kid2dam]
        femalefounders = [x for x in pedigree.females if x not in pedigree.kid2sire and x not in pedigree.kid2dam]
        
        for founder in malefounders+femalefounders:
            id2CPD[str(founder)] = TabularCPD(variable=str(founder), 
                                              variable_card=3, 
                                              values=alleleprobs, 
                                              state_names={str(founder): [0, 1, 2]})
        
        for kid,sire in pedigree.kid2sire.items():
            dam = pedigree.kid2dam[kid]
            id2CPD[str(kid)] = TabularCPD(variable=str(kid), variable_card=3, 
                          values=mendelprobs,
                          evidence=[str(sire), str(dam)],
                          evidence_card=[3, 3], 
                          state_names={str(sire): [0, 1, 2], str(dam): [0, 1, 2], str(kid): [0, 1, 2]})
            if str(dam) not in id2CPD:
                id2CPD[str(dam)] = TabularCPD(variable=str(dam), 
                                              variable_card=3, 
                                              values=alleleprobs, 
                                              state_names={str(dam): [0, 1, 2]})
            if str(dam) not in id2CPD:
                id2CPD[str(sire)] = TabularCPD(variable=str(sire), 
                                              variable_card=3, 
                                              values=alleleprobs, 
                                              state_names={str(sire): [0, 1, 2]})    
        
        
        model_struct = BayesPedigreeNetworkModel(graphedges)
        try :
            model_struct.add_cpds(*[v for k,v in id2CPD.items() if str(k) in model_struct.nodes])
            model_struct.check_model()
        except ValueError as err:
            import re
            
            print(err)
            print(graphedges)
            kiderr = int(re.findall(r'\d+', str(err))[0])
            print(kiderr)
            print(str(kiderr) in id2CPD)
            print(id2CPD[str(kiderr)])
            print(model_struct.get_cpds(str(kiderr)).get_evidence())
            print(model_struct.get_parents(str(kiderr)))
            print(kiderr in malefounders)
            print(kiderr in femalefounders)
            print(id2CPD[str(kiderr)].get_values())
            raise(err)
        return(model_struct)
