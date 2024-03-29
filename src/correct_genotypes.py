#import sys,os
#sys.path.append(os.getcwd())

#from collections import defaultdict, OrderedDict
import concurrent.futures
#import itertools
#import multiprocessing
#import sys
from typing import Tuple, List, Dict, Union, Any, Set

#from joblib import Parallel, delayed
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination, BeliefPropagation
from tqdm import tqdm

import bayesnet
import logoutput
from logoutput import debugutil
from bayesnet.model import BayesPedigreeNetworkModel, _default_mendelprobs, _default_alleleprobs
from bayesnet.result import ResultSingleInfer, ResultPairInfer
import numpy as np
import pandas as pd
from pedigree.pedigree_dag import PedigreeDAG
from scipy.stats import rankdata, boxcox
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from empirical.jalleledist3 import JointAllellicDistribution
import logoutput.stats

_DEBUG_NO_CHANGE = False

def initializer(corrected_genotype_c, pedigree_c, cacheIn=None):
    multiprocessing.current_process().genotypes = corrected_genotype_c.copy()
    multiprocessing.current_process().pedigree = PedigreeDAG(pedigree_c)
    if cacheIn is not None:
        multiprocessing.current_process().cacheIn = cacheIn
    else:
        multiprocessing.current_process().cacheIn = {}

def accumulateCombinations(states, length, current=0, thisarray=None):
    if thisarray is None:
        thisarray = np.full(length, 9,dtype=np.int8)
    elif current == length:
        yield thisarray
        return
    for state in states:
        thisarray[current] = state
        yield from accumulateCombinations(states, length, current=current+1, thisarray=thisarray.copy())

class CorrectGenotypes(object):
   
    def __init__(self, surround_size=4, chromosome2snp=None):
        self.chromosome2snp=chromosome2snp
        self.surround_size=surround_size
    
    @staticmethod
    def mendelProbsSingle(kid:int, back:int):
        '''
        yields mendel probability of states 0,1,2 on node and observed-expected probability sum for edges
        '''
        current = multiprocessing.current_process()
        pedigree = current.pedigree
        genotypes = current.genotypes
        
        parents = [x for x in list(pedigree.get_parents_depth([kid], back)) if x is not None]
        model = None
        if len(parents) > 0:
            blanketnodes: Set[int] = set([kid])
            blanketnodes.update([x for x in parents if x is not None])
            blanket = pedigree.get_subset(list(blanketnodes), balance_parents = True)
            model = BayesPedigreeNetworkModel.generateModel(blanket,_default_alleleprobs, _default_mendelprobs)
        
        result = np.empty((len(genotypes.columns),3), np.float16)
        result_errors = np.empty((len(genotypes.columns),1), np.float16)
        children = np.array(list(set(list(pedigree.get_kids(kid)))))
        
        allindiv = set(children)
        if model is not None:
            allindiv.update({int(x) for x in model.nodes})
        
        cache = dict()
        for j, SNP_id in enumerate(genotypes.columns):
            observed = genotypes.loc[kid,SNP_id]
            
            if len(parents) > 0:
                all_values = dict(zip(list(map(str,allindiv)), genotypes.loc[allindiv,SNP_id]))
                evidence = {x:v for x,v in all_values.items() if int(x) != kid and v != 9 and x in model.nodes}
                key = tuple([*sorted(evidence.items()),kid])
                if key not in cache:
                    infer = VariableElimination(model.copy())
                    prediction = infer.query([str(kid)], evidence=evidence, show_progress=False, joint=False)
                    anc_probs = np.array(prediction[str(kid)].values)
                    cache[key] = anc_probs
                else:
                    anc_probs = cache[key]
                if observed in [0,1,2] :
                    anc_error = np.nanmax(anc_probs) - anc_probs[observed]
            else :
                anc_probs = None
                anc_error = None
            #print("sire %s dam %s %s/%s  " % (sire, dam, all_values[str(sire)], all_values[str(dam)]))
            state_kids = np.array([genotypes.loc[child,SNP_id] for child in children])
            include = [x in [0,1,2] for x in state_kids]
            if len(children) > 0 and len(state_kids[include]) > 0 : #all kids not 9
                state_parents = [None]*len(children[include])
                for i, child in enumerate(children[include]):
                    otherparent = [x for x in pedigree.get_parents(child) if x is not None]
                    if len(otherparent) == 0:
                        state_parents[i] = genotypes.loc[otherparent[0],SNP_id]
                probs = bayesnet.model.generate_probs_kids(state_kids[include], state_parents)
                if observed in [0,1,2] :
                    prob_error = bayesnet.model.generate_probs_differences_kids(state_kids[include], state_parents, observed)                                                         
            else :
                probs = None
                prob_error = None
            
            if observed in [0,1,2] :
                if anc_error is not None and prob_error is not None:
                    result_errors[j] = np.sum(prob_error) + (anc_error*2)
                elif anc_error is not None and prob_error is None:
                    result_errors[j] = anc_error*2
                elif anc_error is None and prob_error is not None:
                    result_errors[j] = np.sum(prob_error)
            
            if anc_probs is not None and probs is not None:
                result[j] = np.nanmean([anc_probs,probs],0, dtype=float)
            elif anc_probs is not None and probs is None:
                result[j] = anc_probs
            elif anc_probs is None and probs is not None:
                result[j] = probs
            else :
                raise Exception("%s is an illegal lone node in the pedigree graph" % kid)
        return(result, result_errors, cache)
    
    @staticmethod
    def correctPair(sire:int, dam:int, error_probs, 
                    back=2, 
                    tiethreshold=0.1, 
                    elimination_order="MinNeighbors", 
                    test_p=0.5,
                    suspect_t=0.2):
        current = multiprocessing.current_process()
        pedigree = current.pedigree
        genotypes = current.genotypes
        cacheIn = current.cacheIn
        
        suspect_snps_sire = error_probs.columns[error_probs.loc[sire,:] >= test_p].tolist()
        suspect_snps_dam = error_probs.columns[error_probs.loc[dam,:] >= test_p].tolist()
        
        suspect_snps = set(suspect_snps_sire)
        suspect_snps.update(suspect_snps_dam)
        
        if len(suspect_snps) > 0:
            parents = [x for x in list(pedigree.get_parents_depth([sire,dam], back)) if x is not None]
            
            model = None 
            if len([x for x in pedigree.get_parents(sire) if x is not None]) > 0 and len([x for x in pedigree.get_parents(dam) if x is not None]) > 0 :
                blanketnodes: Set[int] = set([sire,dam])
                blanketnodes.update(set(parents))
                blanket = pedigree.get_subset(list(blanketnodes), balance_parents = True)
                model = BayesPedigreeNetworkModel.generateModel(blanket,_default_alleleprobs, _default_mendelprobs)
            
            result = ResultPairInfer(sire,dam, model)
            
            children_sire = set(list(pedigree.get_kids(sire)))
            children_dam  =  set(list(pedigree.get_kids(dam)))
            
            allindiv = set([dam, sire])
            allindiv.update(children_sire)
            allindiv.update(children_dam)
            if model is not None:
                allindiv.update({int(x) for x in model.nodes})
            
            if cacheIn is None :
                cache = dict()
            else:
                cache = cacheIn
            
            for SNP_id in suspect_snps:
                all_values = dict(zip(list(map(str,allindiv)), genotypes.loc[allindiv,SNP_id].values))
                observed_state_sire = all_values[str(sire)]
                observed_state_dam = all_values[str(dam)]
                
                error_probs_all = {k:error_probs.loc[int(k),SNP_id] for k, state in all_values.items() if state in [0,1,2]}
                error_probs_pair_max = 1
                if observed_state_sire in [0,1,2] and observed_state_dam in [0,1,2] :
                    error_probs_pair_max = max(error_probs.loc[int(sire),SNP_id], error_probs.loc[int(dam),SNP_id]) 
                elif observed_state_sire not in [0,1,2] and observed_state_dam not in [0,1,2] :
                    error_probs_pair_max = 1
                elif observed_state_sire in [0,1,2] :
                    error_probs_pair_max = error_probs.loc[int(sire),SNP_id]
                elif observed_state_dam in [0,1,2] :
                    error_probs_pair_max = error_probs.loc[int(dam),SNP_id]
                
                suspect_nodes = [node for node, errorv in error_probs_all.items() if errorv > (error_probs_pair_max-suspect_t)]
                
                if model is not None:
                    evidence = {x:v for x,v in all_values.items() 
                                if int(x) != sire  
                                and int(x) != dam 
                                and int(v) != 9 
                                and x not in suspect_nodes
                                and x in model.nodes}
                    key_sire = tuple([*sorted(evidence.items()),sire])
                    key_dam = tuple([*sorted(evidence.items()),dam])
                    if key_sire not in cache or key_dam not in cache:
                        infer = VariableElimination(model.copy())
                        prediction = infer.query([str(sire),str(dam)], evidence=evidence, show_progress=False, joint=False, elimination_order=elimination_order)
                        anc_probs_sire = np.array(prediction[str(sire)].values, dtype=float)
                        anc_probs_dam = np.array(prediction[str(dam)].values, dtype=float)
                        cache[key_sire] = anc_probs_sire
                        cache[key_dam] = anc_probs_dam
                    else:
                        anc_probs_sire = cache[key_sire]
                        anc_probs_dam = cache[key_dam]
                else:
                    anc_probs_sire = None
                    anc_probs_dam = None
                #print("sire %s dam %s %s/%s  " % (sire, dam, all_values[str(sire)], all_values[str(dam)]))
                
                if len(children_sire) > 0:
                    state_kids = [genotypes.loc[child,SNP_id] for child in children_sire]
                    state_parents = [None]*len(children_sire)
                    for i, child in enumerate(children_sire):
                        otherparent = [x for x in pedigree.get_parents(child) if x is not None and x != sire]
                        if len(otherparent) == 1:
                            state_parents[i] = genotypes.loc[otherparent[0],SNP_id]
                    probs_sire = bayesnet.model.generate_probs_kids(state_kids, state_parents)
                else :
                    probs_sire = None
                
                if len(children_dam) > 0:
                    state_kids = [genotypes.loc[child,SNP_id] for child in children_dam]
                    state_parents = [None]*len(children_dam)
                    for i, child in enumerate(children_dam):
                        otherparent = [x for x in pedigree.get_parents(child) if x is not None and x != dam]
                        if len(otherparent) == 1:
                            state_parents[i] = genotypes.loc[otherparent[0],SNP_id]
                    probs_dam = bayesnet.model.generate_probs_kids(state_kids, state_parents)
                else :
                    probs_dam = None
                
                if anc_probs_sire is not None and probs_sire is not None: 
                    stateProbs_sire = np.nanmean([anc_probs_sire,probs_sire],0, dtype=float)
                    #stateProbs_sire = np.multiply(anc_probs_sire,probs_sire)
                elif anc_probs_sire is not None and probs_sire is None: 
                    stateProbs_sire = anc_probs_sire
                elif anc_probs_sire is None and probs_sire is not None: 
                    stateProbs_sire = probs_sire
                else:
                    raise Exception("%s is an illegal lone node in the pedigree graph" % sire)
                
                if np.sum(stateProbs_sire) > 0:
                    stateProbs_sire = np.divide(stateProbs_sire, np.sum(stateProbs_sire))
                    
                if anc_probs_dam is not None and probs_dam is not None: 
                    stateProbs_dam = np.nanmean([anc_probs_dam,probs_dam], 0,dtype=float)
                    #stateProbs_dam = np.multiply(anc_probs_dam,probs_dam)
                elif anc_probs_dam is not None and probs_dam is None: 
                    stateProbs_dam = anc_probs_dam
                elif anc_probs_dam is None and probs_dam is not None: 
                    stateProbs_dam = probs_dam
                else:
                    raise Exception("%s is an illegal lone node in the pedigree graph" % dam)
                
                if np.sum(stateProbs_dam) > 0:
                    stateProbs_dam = np.divide(stateProbs_dam, np.sum(stateProbs_dam))
                
                observed_prob_sire = stateProbs_sire[observed_state_sire] if observed_state_sire != 9 else np.nan #nan differentiates unknown state from a state with prob zero
                observed_prob_dam = stateProbs_dam[observed_state_dam] if observed_state_dam != 9 else np.nan
                
                #stateProbs_joint = np.empty((3,3), dtype=float)
                #for sire_state,dam_state in accumulateCombinations([0,1,2],2):
                #    stateProbs_joint[sire_state,dam_state] = stateProbs_sire[sire_state]*stateProbs_dam[dam_state]
                stateProbs_joint = np.outer(stateProbs_sire,stateProbs_dam)
                                        
                
                max_prob = np.nanmax(stateProbs_joint)
                max_states = [list(x) for x in np.asarray(stateProbs_joint >= (max_prob-tiethreshold)).nonzero()]
            
                result.jointprobs[SNP_id] = stateProbs_joint
                result.beststate[SNP_id] = max_states
                result.bestprobs[SNP_id] = max_prob
                result.observedprob_sire[SNP_id] = observed_prob_sire
                result.observedprob_dam[SNP_id] = observed_prob_dam
                result.bestprob_sire[SNP_id] = [stateProbs_sire[x] for x in max_states[0]]
                result.bestprob_dam[SNP_id] = [stateProbs_dam[x] for x in max_states[1]]
                result.prob_sire[SNP_id]  = stateProbs_sire
                result.prob_dam[SNP_id] = stateProbs_dam        
            return(result)
        return(None)
        
    @staticmethod
    def correctSingle(kid:int, error_probs, 
                      back=1, 
                    tiethreshold=0.1, 
                    elimination_order="MinNeighbors", 
                    test_p=0.5,
                    suspect_t=0.2):
        current = multiprocessing.current_process()
        pedigree = current.pedigree
        genotypes = current.genotypes
        cacheIn = current.cacheIn
                
        suspect_snps = error_probs.columns[error_probs.loc[kid,:] >= test_p].tolist()
        
        if len(suspect_snps) > 0:
            parents = [x for x in list(pedigree.get_parents_depth([kid], back)) if x is not None]
            
            model = None 
            if len(parents) > 0:
                blanketnodes: Set[int] = set([kid])
                blanketnodes.update([x for x in parents if x is not None])
                blanket = pedigree.get_subset(list(blanketnodes), balance_parents = True)
                model = BayesPedigreeNetworkModel.generateModel(blanket,_default_alleleprobs, _default_mendelprobs)
                
            result = ResultSingleInfer(kid, model)
            
            children = set(list(pedigree.get_kids(kid)))
            allindiv = set(children)
            if model is not None:
                allindiv.update({int(x) for x in model.nodes})
            
            if cacheIn is None:
                cache = dict()
            else :
                cache = cacheIn
            for SNP_id in suspect_snps:
                all_values = dict(zip(list(map(str,allindiv)), genotypes.loc[allindiv,SNP_id].values))
                observed_state = all_values[str(kid)]
                
                error_probs_all = {k:error_probs.loc[int(k), SNP_id] for k, state in all_values.items() if state in [0,1,2]}
                error_probs_pair_max = 1
                if observed_state in [0,1,2] :
                    error_probs_pair_max = error_probs.loc[int(kid), SNP_id]
                
                suspect_nodes = [node for node, errorv in error_probs_all.items() if errorv > (error_probs_pair_max-suspect_t)]
                
                if len(parents) > 0:
                    evidence = {x:v for x,v in all_values.items() 
                                if int(x) != kid 
                                and v != 9
                                and x not in suspect_nodes
                                and x in model.nodes }
                    key = tuple([*sorted(evidence.items()),kid])
                    if key not in cache:
                        infer = VariableElimination(model.copy())
                        prediction = infer.query([str(kid)], evidence=evidence, show_progress=False, joint=False, elimination_order=elimination_order)
                        anc_probs = np.array(prediction[str(kid)].values, dtype=float)
                        cache[key] = anc_probs
                    else:
                        anc_probs = cache[key]
                else:
                    anc_probs = None
                #print("sire %s dam %s %s/%s  " % (sire, dam, all_values[str(sire)], all_values[str(dam)]))
                
                if len(children) > 0:
                    state_kids = [genotypes.loc[child,SNP_id] for child in children]
                    state_parents = [None]*len(children)
                    for i, child in enumerate(children):
                        otherparent = [x for x in pedigree.get_parents(child) if x is not None and x != kid]
                        if len(otherparent) == 1:
                            state_parents[i] = genotypes.loc[otherparent[0],SNP_id]
                    probsk = bayesnet.model.generate_probs_kids(state_kids, state_parents)
                else :
                    probsk = None
                
                if anc_probs is not None and probsk is not None: 
                    stateProbs = np.nanmean([anc_probs,probsk],0, dtype=float)
                elif anc_probs is not None and probsk is None: 
                    stateProbs = anc_probs
                elif anc_probs is None and probsk is not None: 
                    stateProbs = probsk
                else:
                    raise Exception("%s is an illegal lone node in the pedigree graph" % kid)
                
                if np.sum(stateProbs) > 0:
                    stateProbs_sire = np.divide(stateProbs, np.sum(stateProbs))
                
                observed_prob = stateProbs_sire[observed_state] if observed_state != 9 else np.nan #nan differentiates unknown state from a state with prob zero
                
                max_prob = np.nanmax(stateProbs)
                max_states = [list(x) for x in np.asarray(stateProbs  >= (max_prob-tiethreshold)).nonzero()]
                result.beststate[SNP_id] = max_states
                result.bestprobs[SNP_id] = max_prob
                result.observedprob[SNP_id] = observed_prob
                result.prob[SNP_id]  = stateProbs     
            return(result)
        return(None)
    
    def correctMatrix(self, genotypes: pd.DataFrame, pedigree:PedigreeDAG, 
                      threshold_pair, 
                      threshold_single,
                      lddist='global', # ['none', 'global', 'local']
                      back=2,
                      init_filter_p=0.8,
                      filter_e=0.7,
                      tiethreshold=0.05,
                      threads=1, 
                      err_thresh=0.1,
                      weight_empirical=3,
                      outputerrors=False,
                      DEBUGDIR=None, debugreal=None) -> pd.DataFrame:
            
            print("correcting genotype %s x %s with %s threads" % (len(genotypes.index), len(genotypes.columns), threads))
            
            with logoutput.stats.Stats(DEBUGDIR) as log_stats:
                corrected_genotype = genotypes.copy()
                probs = {}
                probs_errors = pd.DataFrame(np.zeros(genotypes.shape), columns=genotypes.columns, index=genotypes.index)
                cache_store = {}
                #if debugreal is not None:
                #    errors = np.not_equal(genotypes,debugreal)
                
                errors_all = 0
                excludedNotBest_all = 0
                partner_pairs = list(set([(sire,dam) for sire,dam in [pedigree.get_parents(kid) for kid in genotypes.index] if sire != None and dam != None]))
                
                blankets = {}
                
                for kid in pedigree.males.union(pedigree.females):
                    blanket = set() #pedigree.get_kids(kid) #pedigree.get_kids(kid)
                    #blanket.update(pedigree.get_partners(kid))
                    blanket.update([x for x in pedigree.get_parents(kid) if x is not None])
                    blanket.discard(kid)
                    blankets[kid] = blanket
                
                #populate_base_probs
                print("pre-calculate mendel probs on all individuals")
                with concurrent.futures.ProcessPoolExecutor(max_workers=threads, 
                                                            initializer=initializer,
                                                            initargs=(corrected_genotype,pedigree)) as executor:
                    #for kid in tqdm(pedigree.males.union(pedigree.females)):
                    #    x, b = self.mendelProbsSingle(corrected_genotype, pedigree, kid, back)
                    #    print("kid %s done" % kid)
                    futures = {executor.submit(self.mendelProbsSingle, kid, back):kid for kid in pedigree.males.union(pedigree.females)}
                    print("waiting on %s queued jobs with %s threads" % (len(futures), threads))
                    with tqdm(total=len(futures)) as pbar:
                        for future in concurrent.futures.as_completed(futures) :
                            kid = futures[future]
                            pbar.update(1)
                            e = future.exception()
                            if e is not None:
                                print(repr(e))
                                raise(e)
                            probs[kid], probsErrors, cache_store[kid] = future.result()
                            probs_errors.loc[kid,:] = np.squeeze(probsErrors)
                            #blanket of partners partents and kids
                            del futures[future]
                            del future
                    print("mendel precalc done")
                
                #minnonzeroprobs = np.min(maxsumprobs)
                maxsumprobs = np.nanmax(probs_errors)
                print("maximum ranking for error = %s" % maxsumprobs)
                
                #maxsumprobs = np.nanmax(list(np.concatenate(list(probs_errors.values()))))
                #for kid,snprobs in probs_errors.items():
                #    probs_errors[kid] = np.divide(snprobs,maxsumprobs)
                
                probs_errors = np.log(np.log(probs_errors+1)+1)
                maxsumprobs = np.nanmax(probs_errors)
                probs_errors = probs_errors/np.nanmax(maxsumprobs)
                probs_errors[corrected_genotype == 9] = 1
                
                distribution_of_ranks = probs_errors.to_numpy().flatten()
                quant95_t, quant99_t, quantE_t, quantP_t = np.quantile(distribution_of_ranks, [0.95,0.99, filter_e, init_filter_p], interpolation='higher')
                
                ax = sns.distplot(distribution_of_ranks)
                ax.set(xlabel='sum difference in observed vs expected', ylabel='count')
                plt.axvline(quantE_t, 0,1, color="black")
                plt.axvline(quant95_t, 0,1, color="blue")
                plt.axvline(quant99_t, 0,1, color="red")
                plt.axvline(quantP_t, 0,1, color="yellow")
                plt.savefig("%s/distribution_of_sum_error_ranks_histogram_preld.png" % DEBUGDIR, dpi=300)
                plt.clf()
                
                if lddist != 'none':
                    print("calculating LDDist")
                    emp = JointAllellicDistribution(list(genotypes.columns),
                                                    surround_size=self.surround_size,
                                                    chromosome2snp=self.chromosome2snp)
                    print("create mask")
                    mask = np.array(probs_errors.to_numpy() <= filter_e, dtype=bool)
                    print("calc empirical ld on genotype with %s of %s (%6.2f pc) over under cuttoff %6.6f mendel errors after removing > %s quantile of mendel errors" % (np.count_nonzero(mask), mask.size, (np.count_nonzero(mask)/mask.size)*100,quantE_t, filter_e))
                    #print("calc empirical ld on genotype with %s of %s (%6.2f pc) over under cuttoff 0 mendel errors" % (np.count_nonzero(mask), mask.size, (np.count_nonzero(mask)/mask.size)*100,))
                    emp.countJointFrqAll(genotypes, mask)
                    print("done")
                
                if lddist != 'none':
                    empvalues = list()
                    print("adjusting rank by lnprobs")
                    for kid in tqdm(pedigree.males.union(pedigree.females)):
                        if lddist != 'none':
                            for j, SNP_id in enumerate(genotypes.columns):
                                observed_state = genotypes.loc[kid,SNP_id]
                                observedstatesevidence = {snpid:genotypes.loc[kid,snpid] for snpid in emp.getWindow(SNP_id)}
                                if observed_state in [0,1,2]:
                                    empiricalcount = emp.getCountTable(observedstatesevidence,SNP_id)
                                    prob_states_normalised = np.divide(empiricalcount,np.nansum(empiricalcount))
                                    probs[kid][j] = np.nanmean([prob_states_normalised,probs[kid][j]],0,dtype=float)
                                    empdiff = np.nanmax(prob_states_normalised)-prob_states_normalised[observed_state]
                                    empvalues.append(empdiff)
                                    #probs_errors[kid][j] = np.nanmean([empdiff,probs_errors[kid][j]], 0, dtype=float)
                                    #probs_errors[kid][j] = np.multiply(empdiff,probs_errors[kid][j])
                                    probs_errors.loc[kid,SNP_id] = np.nansum([probs_errors.loc[kid,SNP_id],empdiff*weight_empirical],0,dtype=float)
                    
                    maxsumprobs = np.nanmax(probs_errors)
                    probs_errors = np.divide(probs_errors,maxsumprobs)
                    probs_errors[corrected_genotype == 9] = 1
                    ax = sns.distplot(empvalues)
                    ax.set(xlabel='empvalues probability', ylabel='count')
                    plt.savefig("%s/distribution_of_empvalues_prob.png" % DEBUGDIR, dpi=300)
                    plt.clf()
                
                distribution_of_ranks = probs_errors.to_numpy().flatten()
                quant95_t, quant99_t, quantP_t = np.quantile(distribution_of_ranks, [0.95,0.99, init_filter_p])
                ax = sns.distplot(distribution_of_ranks)
                plt.axvline(quant95_t, 0,1, color="blue")
                plt.axvline(quant99_t, 0,1, color="red")
                plt.axvline(quantP_t, 0,1, color="yellow")
                ax.set(xlabel='probability of error', ylabel='count')
                plt.savefig("%s/distribution_of_error_ranks_histogram_postld.png" % DEBUGDIR, dpi=300)
                plt.clf()
                
                print("n over 95pc quantile = %s" % len([x for x in distribution_of_ranks if x > quant95_t]))
                print("n over 99pc quantile error = %s" % len([x for x in distribution_of_ranks if x > quant99_t]))
               
                print("setting initial filter to %s quantile threshold %s with %s observations" % (quantP_t, init_filter_p, len([x for x in distribution_of_ranks if x > quantP_t])))
                
                print("Computing joint probs for pairs \n there are %s generations and %s unique breeding pairs" % (len(pedigree.generationIndex.keys()),len(partner_pairs)))
                for generation, kids in sorted(pedigree.generationIndex.items()):
                    gen_pairs = [(sire,dam) for sire, dam in partner_pairs if sire in kids or dam in kids]
                    print("generation %s of %s with %s kids and %s pairs" % (generation, len(pedigree.generationIndex.keys()),len(kids), len(gen_pairs)))
                    current_genotype = corrected_genotype.copy()
                    with concurrent.futures.ProcessPoolExecutor(max_workers=threads, 
                                            initializer=initializer,
                                            initargs=(corrected_genotype,pedigree)) as executor:
                        futures = [executor.submit(self.correctPair, sire, dam, probs_errors, 
                                  back=back, tiethreshold=tiethreshold, elimination_order="MinNeighbors", test_p=quantP_t, suspect_t=0.2
                                  ) for sire,dam in gen_pairs] #cacheIn={**cache_store[sire],**cache_store[dam]} 
                        
                        print("waiting on %s queued jobs with %s threads" % (len(futures), threads))
                        with tqdm(total=len(futures)) as pbar:
                            for future in concurrent.futures.as_completed(futures) :
                                pbar.update(1)
                                e = future.exception()
                                if e is not None:
                                    print(repr(e))
                                    raise(e)
                                
                                resultPair  = future.result()
                                del future
                                
                                if resultPair is not None:
                                    sire_blanket = blankets[resultPair.sire].copy()
                                    dam_blanket = blankets[resultPair.dam].copy()
                                    
                                    observed_sire = current_genotype.loc[resultPair.sire,:]
                                    observed_dam = current_genotype.loc[resultPair.dam,:]
                                    
                                    n_errors = 0
                                    excludedNotBest = 0
                                    #diff = np.subtract(maxprobs,joint_probs_observed)
                                    
                                    n_snps = len(list(genotypes.columns))
                                    for j,SNP_id in [(i,snp) for i, snp in enumerate(genotypes.columns) if snp in resultPair.jointprobs.keys()]:
                                        
                                        sire_probs_all_states = resultPair.prob_sire[SNP_id]
                                        dam_probs_all_states = resultPair.prob_dam[SNP_id]
                                        
                                        mendel_joint_probs = resultPair.jointprobs[SNP_id]
                                        
                                        sire_probs = 0
                                        dam_probs = 0
                                        if observed_sire[SNP_id] != 9 and observed_dam[SNP_id] != 9:
                                            sire_probs = mendel_joint_probs[observed_sire[SNP_id],observed_dam[SNP_id]]
                                            dam_probs = mendel_joint_probs[observed_sire[SNP_id],observed_dam[SNP_id]]
                                        else:
                                            if observed_sire[SNP_id] != 9:
                                                sire_probs = resultPair.observedprob_sire[SNP_id]
                                            if observed_dam[SNP_id] != 9:
                                                dam_probs = resultPair.observedprob_dam[SNP_id]
                                        sireparents = [x for x in pedigree.get_parents(resultPair.sire) if x is not None]
                                        sire_error_rank = probs_errors.loc[resultPair.sire,SNP_id]
                                        dam_error_rank = probs_errors.loc[resultPair.dam,SNP_id]
                                        sire_blanket_ranks = [probs_errors.loc[b,SNP_id] for b in sire_blanket if b != resultPair.sire]
                                        dam_blanket_ranks = [probs_errors.loc[b,SNP_id] for b in dam_blanket if b != resultPair.dam]
                                        
                                        sire_blanket_maxrank = np.nanmax([0]+sire_blanket_ranks)
                                        dam_blanket_maxrank = np.nanmax([0]+dam_blanket_ranks)
                                        
                                        damparents = [x for x in pedigree.get_parents(resultPair.dam) if x is not None]
                                        maxstates = resultPair.beststate[SNP_id]
                                        maxprobs = resultPair.bestprobs[SNP_id]
                                        
                                        if lddist != 'none': # this overwrites the probabilities with product of empirical and mendelian 
                                            snpwindow = emp.getWindow(SNP_id)
                                            observedstatesevidence = {snpid:genotypes.loc[resultPair.sire,snpid] for snpid in snpwindow}
                                            state_obs = list(emp.getCountTable(observedstatesevidence, SNP_id))
                                            sire_prob_states_normalised = np.divide(state_obs,np.nansum(state_obs))
                                            
                                            observedstatesevidence = {snpid:genotypes.loc[resultPair.dam,snpid] for snpid in snpwindow}
                                            state_obs = list(emp.getCountTable(observedstatesevidence, SNP_id))
                                            dam_prob_states_normalised = np.divide(state_obs,np.nansum(state_obs))
                                                                                    
                                            #sire_probs_all_states = np.multiply(sire_prob_states_normalised,sire_probs_all_states)
                                            #dam_probs_all_states = np.multiply(dam_prob_states_normalised,dam_probs_all_states)
                                                                                    
                                            stateProbs_empirical_joint = np.empty((3,3), dtype=float)
                                            for sire_state,dam_state in accumulateCombinations([0,1,2],2):
                                                stateProbs_empirical_joint[sire_state,dam_state] = sire_prob_states_normalised[sire_state]*dam_prob_states_normalised[dam_state]
                                            #stateProbs_empirical_joint = np.outer(sire_prob_states_normalised,dam_prob_states_normalised)
                                            stateProbs_empirical_joint = stateProbs_empirical_joint/np.sum(stateProbs_empirical_joint)
                                            
                                            maxstates_empirical = [list(x) for x in np.asarray(stateProbs_empirical_joint == np.nanmax(stateProbs_empirical_joint)).nonzero()]
                                            maxstates_empirical_joint_rank = rankdata(1-stateProbs_empirical_joint, method='max').reshape(stateProbs_empirical_joint.shape)
                                            maxstates_mendel_joint_rank = rankdata(1-resultPair.jointprobs[SNP_id], method='max').reshape(resultPair.jointprobs[SNP_id].shape)
                                            
                                            multiply_rank = rankdata(1-(np.multiply(resultPair.jointprobs[SNP_id],stateProbs_empirical_joint)), method='max').reshape(resultPair.jointprobs[SNP_id].shape)
                                            mean_rank = rankdata(1-(np.nanmean([resultPair.jointprobs[SNP_id],stateProbs_empirical_joint],0, dtype=float)), method='max').reshape(resultPair.jointprobs[SNP_id].shape)
                                            
                                            combined_probs = np.nanmean([resultPair.jointprobs[SNP_id],stateProbs_empirical_joint],0, dtype=float)
                                            if np.nansum(combined_probs) > 0:
                                                combined_probs = np.divide(combined_probs,np.sum(combined_probs)) # normalise so sum is 1
                                            combined_maxprobs = np.nanmax(combined_probs)
                                            combined_maxstates = [list(x) for x in np.asarray(combined_probs >= (combined_maxprobs-tiethreshold)).nonzero()]
                                            
                                            if observed_sire[SNP_id] != 9 and observed_dam[SNP_id] != 9:
                                                sire_probs = combined_probs[observed_sire[SNP_id],observed_dam[SNP_id]]
                                                dam_probs = combined_probs[observed_sire[SNP_id],observed_dam[SNP_id]]
                                            else:
                                                if observed_sire[SNP_id] != 9:
                                                    sire_probs_combined = np.multiply(resultPair.observedprob_sire[SNP_id],sire_prob_states_normalised)
                                                    sire_probs_combined = np.divide(sire_probs_combined,np.sum(sire_probs_combined))
                                                    sire_probs = sire_probs_combined[observed_sire[SNP_id]]     
                                                if observed_dam[SNP_id] != 9:
                                                    dam_probs_combined = np.multiply(resultPair.observedprob_dam[SNP_id],dam_prob_states_normalised)
                                                    dam_probs_combined = np.divide(dam_probs_combined,np.sum(dam_probs_combined))
                                                    dam_probs = dam_probs_combined[observed_dam[SNP_id]]                           
                                            maxstates = combined_maxstates
                                            joint_probs_observed = combined_probs
                                            
                                            #print("combined_probs %s" % combined_probs)
                                            #print("maxstates %s" % maxstates)
                                            #print("sire_probs %s" % sire_probs)
                                            #print("dam_probs %s" % dam_probs)
                                            
                                            if DEBUGDIR is not None:
                                                observed = [current_genotype.loc[resultPair.sire,SNP_id], current_genotype.loc[resultPair.dam,SNP_id]]
                                                actual = [debugreal.loc[resultPair.sire,SNP_id], debugreal.loc[resultPair.dam,SNP_id]]
                                                log_stats.pair_stats.write("\t".join(list(map(str,[",".join(map(str,observed)),
                                                                                   ",".join(map(str,actual)), 
                                                                                   ",".join(map(str,["%s:%s" % (x,y) for x,y in zip(*maxstates_empirical)])), 
                                                                                   maxstates_empirical_joint_rank[actual[0]][actual[1]],
                                                                                   -9 if 9 in observed else maxstates_empirical_joint_rank[observed[0]][observed[1]],
                                                                                   maxstates_mendel_joint_rank[actual[0]][actual[1]],
                                                                                   -9 if 9 in observed else maxstates_mendel_joint_rank[observed[0]][observed[1]],
                                                                                   multiply_rank[actual[0]][actual[1]],
                                                                                   -9 if 9 in observed else multiply_rank[observed[0]][observed[1]],
                                                                                   mean_rank[actual[0]][actual[1]],
                                                                                   -9 if 9 in observed else mean_rank[observed[0]][actual[1]],
                                                                                   rankdata(1-(dam_prob_states_normalised), method='max')[actual[1]],
                                                                                   rankdata(1-(sire_prob_states_normalised), method='max')[actual[0]],
                                                                                   rankdata(1-(np.array(dam_probs_all_states)), method='max')[actual[1]],
                                                                                   rankdata(1-(np.array(sire_probs_all_states)), method='max')[actual[0]],
                                                                                   maxstates_empirical_joint_rank[actual[1]][actual[0]],
                                                                                   maxstates_mendel_joint_rank[actual[1]][actual[0]]]
                                                                                    )))+"\n")
                                        
                                        if observed_sire[SNP_id] not in [0,1,2] or (observed_sire[SNP_id] not in maxstates[0] and ((np.nanmax(joint_probs_observed)-sire_probs) >= threshold_pair)):
                                            #print("sire %s dam %s sire %s dam %s" % (current_genotype.loc[resultPair.sire,SNP_id], current_genotype.loc[resultPair.dam,SNP_id], debugreal.loc[resultPair.sire,SNP_id], debugreal.loc[resultPair.dam,SNP_id]))
                                            #print("%s %s %s %s" % (observed_sire[SNP_id], maxstates[0], np.nanmax(sire_probs), threshold_pair))
                                            #print("jointprobs mendelian %s : %s" % (resultPair.jointprobs[SNP_id],np.sum(resultPair.jointprobs[SNP_id])))
                                            #print("jointprobs empirical %s : %s" % (stateProbs_empirical_joint,np.sum(stateProbs_empirical_joint)))
                                            #print("max empirical %s" % np.nanmax(stateProbs_empirical_joint))
                                            #print("max mendelian %s" % maxstates)
                                            #print("max states empirical %s" % "".join(map(str,[list(x) for x in np.asarray(stateProbs_empirical_joint == np.nanmax(stateProbs_empirical_joint)).nonzero()])))
                                            if sire_error_rank+err_thresh  >= sire_blanket_maxrank: #highest difference in blanket?
                                                if (len(set(maxstates[0])) == 1):
                                                    corrected_genotype.loc[int(resultPair.sire),SNP_id] = maxstates[0][0]
                                                else:
                                                    corrected_genotype.loc[int(resultPair.sire),SNP_id] = 9
                                                n_errors+=1
                                            else:
                                                excludedNotBest +=1
                                                if outputerrors and corrected_genotype.loc[int(resultPair.sire),SNP_id] != debugreal.loc[int(resultPair.sire),SNP_id]:
                                                    logoutput.debugutil._debugoutput(DEBUGDIR, "pairsire_incorrectrankexcl", SNP_id, resultPair, sireparents,corrected_genotype, debugreal, probs)
                                        else:
                                            if outputerrors and corrected_genotype.loc[int(resultPair.sire),SNP_id] != debugreal.loc[int(resultPair.sire),SNP_id]:
                                                logoutput.debugutil._debugoutput(DEBUGDIR, "pairsire_thresholdexcl", SNP_id, resultPair, sireparents, corrected_genotype, debugreal, probs)
                                        
                                        if observed_dam[SNP_id] not in [0,1,2] or (observed_dam[SNP_id] not in maxstates[1] and ((np.nanmax(joint_probs_observed)-dam_probs) >= threshold_pair)):
                                            #print("%s %s %s %s" % (observed_dam[SNP_id], maxstates[1], np.nanmax(dam_probs), threshold_pair))
                                            if dam_error_rank+err_thresh >= dam_blanket_maxrank: #highest difference in blanket?
                                                if (len(set(maxstates[1])) == 1):
                                                    corrected_genotype.loc[int(resultPair.dam),SNP_id]  = maxstates[1][0]
                                                else:
                                                    corrected_genotype.loc[int(resultPair.dam),SNP_id] = 9
                                                n_errors+=1
                                            else:
                                                excludedNotBest +=1
                                                if outputerrors and corrected_genotype.loc[int(resultPair.dam),SNP_id] != debugreal.loc[int(resultPair.dam),SNP_id]:
                                                    logoutput.debugutil._debugoutput(DEBUGDIR, "pairdam_incorrectrankexcl", SNP_id, resultPair, damparents, corrected_genotype, debugreal, probs)
                                        else:
                                            if outputerrors and corrected_genotype.loc[int(resultPair.dam),SNP_id] != debugreal.loc[int(resultPair.dam),SNP_id]:
                                                logoutput.debugutil._debugoutput(DEBUGDIR, "pairdam_thresholdexcl", SNP_id, resultPair, damparents, corrected_genotype, debugreal, probs)
        
                                    errors_all+=n_errors
                                    excludedNotBest_all += excludedNotBest
                print("n errors pairs: not rank1 excluded %s passed %s " % (excludedNotBest_all, errors_all))
                
                pairedkids = set([x[0] for x in partner_pairs]+[x[1] for x in partner_pairs])
                allkids = pedigree.males.union(pedigree.females)
                singlekids = [x for x in allkids if x not in pairedkids]
                with concurrent.futures.ProcessPoolExecutor(max_workers=threads, 
                                            initializer=initializer,
                                            initargs=(corrected_genotype,pedigree)) as executor:
                    futures = {executor.submit(self.correctSingle, kid, probs_errors,
                                                back=back, tiethreshold=tiethreshold, elimination_order="MinNeighbors", test_p=quantP_t, suspect_t=0.2):kid for kid in singlekids} #cacheIn=cache_store[kid]
                    print("waiting on %s queued jobs with %s threads" % (len(futures), threads))
                    with tqdm(total=len(futures)) as pbar:
                        for future in concurrent.futures.as_completed(futures) :
                            pbar.update(1)
                            e = future.exception()
                            if e is not None:
                                print(repr(e))
                                raise(e)
                            resultkid: ResultSingleInfer = future.result()
                            #observed_kid = current_genotype.loc[resultkid.kid,:]
                            if resultkid is not None:
                                n_errors = 0
                                
                                blanket = blankets[resultkid.kid]
                                
                                for j,SNP_id in [(i,snp) for i, snp in enumerate(genotypes.columns) if snp in resultkid.prob.keys()]:
                                    maxstates = resultkid.beststate[SNP_id]
                                    maxprobs = resultkid.bestprobs[SNP_id]
                                    observed_state = current_genotype.loc[resultkid.kid,SNP_id] 
                                    observed_prob = 0. if observed_state == 9 else resultkid.observedprob[SNP_id]
                                    
                                    if lddist != 'none': # this overwrites the probabilities with product of empirical and mendelian 
                                        snpwindow = emp.getWindow(SNP_id)
                                        observedstatesevidence = {snpid:genotypes.loc[resultkid.kid,snpid] for snpid in snpwindow}
                                        state_obs = list(emp.getCountTable(observedstatesevidence, SNP_id))
                                        prob_states_normalised =  np.divide(state_obs,np.sum(state_obs))
                                                               
                                        maxstates_empirical = [list(x) for x in np.asarray(prob_states_normalised == np.nanmax(prob_states_normalised)).nonzero()]
                                        maxstates_empirical_rank = rankdata(1-prob_states_normalised, method='max')
                                        maxstates_mendel_rank = rankdata(1-resultkid.prob[SNP_id], method='max')
                                        
                                        multiply_rank = rankdata(1-(np.multiply(resultkid.prob[SNP_id],prob_states_normalised)), method='max')
                                        mean_rank = rankdata(1-(np.nanmean([resultkid.prob[SNP_id],prob_states_normalised],0, dtype=float)), method='max')
                                        
                                        combined_probs = np.nanmean([resultkid.prob[SNP_id],prob_states_normalised],0, dtype=float)
                                        combined_probs = np.divide(combined_probs,np.sum(combined_probs)) # normalise so sum is 1
                                        combined_maxprobs = np.nanmax(combined_probs)
                                        combined_maxstates = [list(x) for x in np.asarray(combined_probs >= (combined_maxprobs-tiethreshold)).nonzero()]
                                        
                                        if observed_state != 9:
                                            observed_prob = combined_probs[observed_state]
                                        else:
                                            observed_prob = 0
                                                                     
                                        maxstates = combined_maxstates
                                        maxprobs = combined_maxprobs
                                        
                                        #print("combined_probs %s" % combined_probs)
                                        #print("maxstates %s" % maxstates)
                                        #print("sire_probs %s" % sire_probs)
                                        #print("dam_probs %s" % dam_probs)
        
                                        if DEBUGDIR is not None:
                                            observed = current_genotype.loc[resultkid.kid,SNP_id]
                                            actual = debugreal.loc[resultkid.kid,SNP_id]
                                            log_stats.single_stats.write("\t".join(list(map(str,[observed,
                                                                               actual, 
                                                                               ",".join(map(str,maxstates_empirical)), 
                                                                               maxstates_empirical_rank[actual],
                                                                               -3 if observed == 9 else maxstates_empirical_rank[observed],
                                                                               maxstates_mendel_rank[actual],
                                                                               -3 if observed == 9  else maxstates_mendel_rank[observed],
                                                                               multiply_rank[actual],
                                                                               -3 if observed == 9 else multiply_rank[observed],
                                                                               mean_rank[actual],
                                                                               mean_rank[observed],
                                                                               rankdata(1-(prob_states_normalised), method='max')[actual],
                                                                               rankdata(1-(prob_states_normalised), method='max')[actual],
                                                                               rankdata(1-(resultkid.prob[SNP_id]), method='max')[actual],
                                                                               rankdata(1-(resultkid.prob[SNP_id]), method='max')[actual], 
                                                                               0,0]                
                                                                                )))+"\n")
                                    blanket_ranks = [probs_errors.loc[b,SNP_id] for b in blanket if b != resultkid.kid]
                                    blanket_maxrank = np.nanmax([0]+blanket_ranks)
                                    kid_error_rank = probs_errors.loc[resultkid.kid,SNP_id]
                                    
                                    if (maxprobs - observed_prob) >= threshold_single and kid_error_rank+err_thresh >= blanket_maxrank:
                                        if (len(set(maxstates[0])) == 1):
                                            corrected_genotype.loc[int(resultkid.kid),SNP_id] = maxstates[0][0]
                                        else:
                                            corrected_genotype.loc[int(resultkid.kid),SNP_id] = 9
                                        n_errors+=1
                                    else:
                                        if outputerrors and corrected_genotype.loc[int(resultkid.kid),SNP_id] != debugreal.loc[int(resultkid.kid),SNP_id]:
                                            logoutput.debugutil._debugoutput_s(DEBUGDIR, "single_thresholdexcl", SNP_id, resultkid, corrected_genotype, debugreal, probs)
                                        excludedNotBest+=1
                                errors_all+=n_errors
                                excludedNotBest_all += excludedNotBest
                            
                correct = np.product(genotypes.shape)-errors_all
                print("n errors pairs+single: not rank1 excluded %s passed %s " % (excludedNotBest_all, errors_all))
                print("predicted errors %s of %s ({%1.2f}pc)" % (errors_all, correct, (errors_all/(errors_all+correct))*100))
                
                return(corrected_genotype)
#===============================================================================
#  
# print("Loading data ..")
# print("pedigree ..")
# pedigree = PedigreeDAG.from_file("/gensys/mhindle/quick_sim/poc3/allPed-sex.txt")
# print("sim_data ..")
# sim_data = pd.read_csv("/home/mhindle/simulation_correction_newalgo1/simulatedgenome.ssv", sep=" ", header=0, index_col=0)
# sim_data = sim_data[list(sim_data.columns)]
# print("sim_data_errors ..")
# sim_data_errors = pd.read_csv("/home/mhindle/simulation_correction_newalgo1/simulatedgenome_with_2.5errors.ssv", sep=" ", header=0, index_col=0)
# sim_data_errors = sim_data_errors[list(sim_data_errors.columns)]
#   
# print(sim_data)
# print(sim_data_errors)
#   
# errors = np.count_nonzero(np.not_equal(sim_data,sim_data_errors))
#   
# print("There are %s errors " % errors)
#   
# print(sim_data.shape)
#   
# #allelefrq = pd.DataFrame(np.array([sim_data[snp].value_counts().values for snp in sim_data.columns]), index=sim_data.columns, columns=["0","1","2"])
# #print(allelefrq)
#   
# c = CorrectGenotypes()
# #c.correctMatrix(sim_data, pedigree, 0.34,threads=12, DEBUG=True)
# import pathlib
# pathlib.Path("/home/mhindle/simulation_correction_newalgo2").mkdir(parents=True, exist_ok=True)
#   
# blanketnodes: Set[int] = set([3827182281448, 3827188551242])
# parents = [x for x in list(pedigree.get_parents_depth([3827182281448, 3827188551242],2)) if x is not None]
# blanketnodes.update([x for x in parents if x is not None])
#   
# kids_include = set(list(pedigree.get_kids(3827182281448))+list(pedigree.get_kids(3827188551242)))
# blanketnodes.update(kids_include)
# for kid in kids_include:
#     blanketnodes.update(set(pedigree.get_parents_depth([kid], 2)))
#   
# blanket = pedigree.get_subset(list(blanketnodes), balance_parents = True)
#   
# probs = {}
# print("pre-calculate mendel probs on all individuals")
# with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
#     futures = {executor.submit(c.probsSingle, sim_data_errors, pedigree, kid, back=2):kid for kid in map(int, blanket.nodes())}
#     print("waiting on %s queued jobs with %s threads" % (len(futures), 12))
#     for future in concurrent.futures.as_completed(futures) :
#         kid = futures[future]
#         resultkid = future.result()
#         probs[kid] = resultkid
#   
# #print(probs[3968126301727][list(sim_data_errors.columns).index('AX-75266407')])
#   
# print("Done")
#===============================================================================

#resultPair = c.correctSingle(sim_data_errors, pedigree, 3827182281448, probs, back=2)
#resultPair = c.correctSingle(sim_data_errors, pedigree, 3827188551242, probs, back=2)

#===============================================================================
# resultPair = c.correctPair(sim_data_errors, pedigree, 3827182281448, 3827188551242, probs, 2)
# 
# SNP_id = "AX-75433956"
# j = list(sim_data_errors.columns).index(SNP_id)
# sire_probs = resultPair.bestprob_sire[SNP_id]
# dam_probs = resultPair.bestprob_dam[SNP_id]
# sire_obsprobs = resultPair.observedprob_sire[SNP_id]
# dam_obsprobs = resultPair.observedprob_dam[SNP_id]
# 
# diffProbs_sire = np.nanmax(sire_probs) - sire_obsprobs
# diffProbs_dam = np.nanmax(dam_probs) - dam_obsprobs
# 
# #print("%s %s " % (SNP_id, prob_difference))
# sireparents = [x for x in pedigree.get_parents(resultPair.sire) if x is not None]
# obsProbs_sireparents = [probs[x][j,sim_data_errors.loc[x,SNP_id]] for x in sireparents] 
# maxProbs_sireparents = [np.nanmax(probs[x][j,:]) for x in sireparents] 
# diffProbs_sireparents = np.subtract(maxProbs_sireparents,obsProbs_sireparents)
# 
# damparents = [x for x in pedigree.get_parents(resultPair.dam) if x is not None]
# obsProbs_damparents = [probs[x][j,sim_data_errors.loc[x,SNP_id]] for x in damparents] 
# maxProbs_damparents = [np.nanmax(probs[x][j,:]) for x in damparents] 
# diffProbs_damparents = np.subtract(maxProbs_damparents,obsProbs_damparents)
# 
# maxstates = resultPair.beststate[SNP_id]
# maxprobs = resultPair.bestprobs[SNP_id]
# 
# if len(sireparents)  == 0 or diffProbs_sire > np.nanmax(diffProbs_sireparents): #highest difference in blanket?
#     print("condition met")
# else:
#     print("not met")
#     print(diffProbs_sire)
#     print(diffProbs_sireparents)
#     
# 
# print(list(sim_data.columns))
# snploc = list(sim_data.columns).index("AX-75433956")
#   
# print("p %s" % resultPair.bestprobs["AX-75433956"])
# print("s %s" % resultPair.beststate["AX-75433956"])
# print("sire bp %s" % resultPair.bestprob_sire["AX-75433956"])
# print("dam bp %s" % resultPair.bestprob_dam["AX-75433956"])
# print("sire p %s" % resultPair.prob_sire["AX-75433956"])
# print("dam p %s" % resultPair.prob_dam["AX-75433956"])
# print("sire _e %s" % sim_data_errors.loc[int(resultPair.sire), "AX-75433956"])
# print("dam _e %s" % sim_data_errors.loc[int(resultPair.dam), "AX-75433956"])
# print("sire _r %s" % sim_data.loc[int(resultPair.sire), "AX-75433956"])
# print("dam _r %s" % sim_data.loc[int(resultPair.dam), "AX-75433956"])
#   
# import matplotlib.pyplot as plt
# import networkx as nx
# from networkx.drawing.nx_agraph import graphviz_layout
# from bayesnet.result import ResultPairInfer, ResultSingleInfer
#   
#   
# plt.figure(figsize=(24, 24))
# plt.margins(x=0.3, y=0.3)
# model2 = nx.relabel_nodes(resultPair.model, {k:"%s_e%s[%s]|%s%s" % (k, 
#                                                                      sim_data_errors.loc[int(k), "AX-75433956"] ,
#                                                                      sim_data.loc[int(k), "AX-75433956"], 
#                                                                      np.around(probs[int(k)][snploc],4),
#                                                                      "*" if sim_data_errors.loc[int(k), "AX-75433956"] != sim_data.loc[int(k), "AX-75433956"] else ""
#                                                                      ) for i,k in enumerate(resultPair.model.nodes)}, True)
#   
# pos=graphviz_layout(model2, prog='dot', args='-s600' )
# nx.draw(model2, pos, with_labels=False, arrows=True,node_size=10,font_size=4,
#         node_color=["red" if str(node).endswith("*") else "blue" for node in model2.nodes])
# text = nx.draw_networkx_labels(model2, pos)
# for _,t in text.items():
#     t.set_rotation(45)
#   
# plt.show()
#===============================================================================
#c.correctMatrix(sim_data_errors, pedigree, 0.4, 0.9, threads=20, 
#                 DEBUGDIR="/home/mhindle/simulation_correctionT2", debugreal=sim_data)
 
             
