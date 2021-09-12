'''
Created on Aug 12, 2021

@author: mhindle

'''
from networkx.drawing.nx_agraph import graphviz_layout

from bayesnet.result import ResultPairInfer, ResultSingleInfer
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def _accumulateCombinations(states, length, current=0, thisarray=None):
    if thisarray is None:
        thisarray = np.full(length, 9,dtype=np.int8)
    elif current == length:
        yield thisarray
        return
    for state in states:
        thisarray[current] = state
        yield from _accumulateCombinations(states, length, current=current+1, thisarray=thisarray.copy())

def _debugoutput(DEBUGDIR, name, SNP_id, resultPair: ResultPairInfer, restofblanket, corrected_genotype, debugreal, probs):
    j = list(corrected_genotype.columns).index(SNP_id)
    
    observed_states = [corrected_genotype.loc[resultPair.sire,SNP_id], corrected_genotype.loc[resultPair.dam,SNP_id]]
    observed_prob_sire = resultPair.observedprob_sire[SNP_id]
    observed_prob_dam = resultPair.observedprob_dam[SNP_id]
    observed_prob = observed_prob_sire*observed_prob_dam
    max_prob = resultPair.bestprobs[SNP_id]
    max_states = resultPair.beststate[SNP_id]
    
    with open("%s/%s___%s_%s_%s_error.txt" % (DEBUGDIR, name, resultPair.sire, resultPair.dam, SNP_id), "wt") as fout:
        fout.write("sire %s dam %s\n" % (resultPair.sire, resultPair.dam))
        fout.write("observed %s\n" % observed_states)
        fout.write("joint observed prob %s\n" % observed_prob)
        fout.write("kids joint sire observed prob %s\n" % observed_prob_sire)
        fout.write("kids joint dam observed prob %s\n" % observed_prob_dam)
        fout.write("difference %s\n" % (max_prob-observed_prob) )
        fout.write("max_prob %s\n" % (max_prob) )
        fout.write("max_states %s\n" % (max_states) )
        fout.write("PREDICTED STATES %s\n" % max_states)
        fout.write("rest of blanket %s\n" % restofblanket)
        if debugreal is not None:
            fout.write("REAL states sire %s dam %s\n" % (debugreal.loc[resultPair.sire, SNP_id], debugreal.loc[resultPair.dam, SNP_id]))
    
    if resultPair.model is not None:
        plt.figure(figsize=(8, 24))
        plt.margins(x=0.2, y=0.2)
        model2 = nx.relabel_nodes(resultPair.model, {k:"%s_e%s[%s]|r%s" % (k,corrected_genotype.loc[int(k), SNP_id],
                                                                             debugreal.loc[int(k), SNP_id], 
                                                                             ",".join(list(map(str,probs[int(k)][j,:])))) for k in resultPair.model.nodes}, True)
        
        pos=graphviz_layout(model2, prog='dot', args='-s300' )
        nx.draw(model2, pos, with_labels=False, arrows=True,node_size=20,font_size=5,
                node_color=["red" if str(node).startswith(str(resultPair.sire)) or str(node).startswith(str(resultPair.dam)) else "blue" for node in model2.nodes])
        text = nx.draw_networkx_labels(model2, pos)
        for _,t in text.items():
            t.set_rotation(45)
        plt.savefig("%s/%s___%s_%s_%s_error.png" % (DEBUGDIR, name, resultPair.sire, resultPair.dam, SNP_id))
        plt.close()
    
def _debugoutput_s(DEBUGDIR, name, SNP_id, result: ResultSingleInfer, corrected_genotype, debugreal, probs):
    
    j = list(corrected_genotype.columns).index(SNP_id)
    observed_state = corrected_genotype.loc[result.kid,SNP_id]
    observed_prob = result.observedprob[SNP_id]
    max_prob = result.bestprobs[SNP_id]
    max_states = result.beststate[SNP_id]
    
    with open("%s/%s___%s_%s_error.txt" % (DEBUGDIR, name, result.kid, SNP_id), "wt") as fout:
        fout.write("kid %s\n" % (result.kid))
        fout.write("observed %s\n" % observed_state)
        fout.write("observed prob %s\n" % observed_prob)
        fout.write("difference %s\n" % (max_prob-observed_prob) )
        fout.write("max_prob %s\n" % (max_prob) )
        fout.write("max_states %s\n" % (max_states) )
        fout.write("PREDICTED STATES %s\n" % max_states)
    
    plt.figure(figsize=(8, 24))
    plt.margins(x=0.2, y=0.2)
    model2 = nx.relabel_nodes(result.model, {k:"%s_e%s[%s]|r%s" % (k,corrected_genotype.loc[int(k), SNP_id],
                                                                     debugreal.loc[int(k), SNP_id], 
                                                                     ",".join(list(map(str,probs[int(k)][j,:])))) for k in result.model.nodes}, True)
    
    pos=graphviz_layout(model2, prog='dot', args='-s300' )
    nx.draw(model2, pos, with_labels=False, arrows=True,node_size=20,font_size=5,
            node_color=["red" if str(node).startswith(str(result.kid)) else "blue" for node in model2.nodes])
    text = nx.draw_networkx_labels(model2, pos)
    for _,t in text.items():
        t.set_rotation(45)
    plt.savefig("%s/%s___%s_%s_error.png" % (DEBUGDIR, name, result.kid, SNP_id))
    plt.close()
    