### Import all the useful modules, they can be used from the rest of the modules
import os 
import time
import numpy as np
import tqdm

#####################
# Utility functions #
#####################

def fix_seed(seed):
    global RAN_GEN
    RAN_GEN = randomise(seed)

def randomise(seed=None):
    '''
    ensures seed value is between 0 and 2**32 by taking the remainder for values over limit and multiplying floats below 0 by 1E4
    '''
    if not seed:
        seed = time.time_ns()+os.getpid()
    seed = abs(seed)
    if seed >= (2**32)-1:
        seed = seed % (2**32)
    while seed <= 0:
        seed = abs((seed+1)*1E4)
    return np.random.default_rng(int(seed))


## Defualt Random Generator
RAN_GEN = randomise(seed=None)


## Module exports for public API (order important to avoid circular imports)
from .animal import Animal
from .genotype import Genotype
from .genome import Genome
from .importers import read_snps,read_real_haplos
from .sim import create_founders,mate,drop_pedigree


