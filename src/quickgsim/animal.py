# -*- coding: utf-8 -*-
"""Implementing the Animal class

"""
from dataclasses import dataclass,field
from typing import List, Dict

@dataclass
class Animal:
    """Dataclass implementation to provide alternative access to Genotype() 

    Attributes:
        tag (str): An identifier for the animal. Useful for retrieving in 
                   backend
        sex (int): The sex of the animal [1:male, 2:female]
        genotype(dict): A dictionary compatible to a Genotype (ie a UserDict)
        pcr(list): A 
        mcr(list): 
        
    """
    tag: str
    sex : int
    genotype : Dict
    pcr: List[float] = field(default_factory=list)
    mcr: List[float] = field(default_factory=list)

    def get_haplo_genotype(self,strand):
        return self.genotype.as_haplo(strand)

    def get_array_genotype(self):
        return self.genotype.as_snps()
