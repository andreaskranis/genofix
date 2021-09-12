'''
Created on Aug 18, 2021

@author: mhindle
'''
from typing import Tuple, List, Dict, Union, Any, Set

from bayesnet.model import BayesPedigreeNetworkModel


class ResultPairInfer(object):
    '''
    classdocs
    '''


    def __init__(self, sire, dam, model):
        '''
        Constructor
        '''
        self.sire = sire
        self.dam = dam
        self.bestprobs = {}
        self.beststate = {}
        self.bestprob_sire = {}
        self.bestprob_dam = {}
        self.prob_sire = {}
        self.prob_dam = {}
        self.observedprob_sire = {}
        self.observedprob_dam = {}
        self.model: BayesPedigreeNetworkModel = model

class ResultGroupInfer(object):
    '''
    classdocs
    '''


    def __init__(self, sire, dam, model):
        '''
        Constructor
        '''
        self.results: List[ResultSingleInfer] = list()
        self.model: BayesPedigreeNetworkModel = model

class ResultSingleInfer(object):
    '''
    classdocs
    '''

    def __init__(self, kid, model):
        '''
        Constructor
        '''
        self.kid = kid
        self.bestprobs = {}
        self.beststate = {}
        self.observedprob = {}
        self.prob = {}
        self.model: BayesPedigreeNetworkModel = model
    