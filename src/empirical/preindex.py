#!/usr/local/bin/python2.7
# encoding: utf-8
'''
empirical.preindex -- shortdesc

empirical.preindex is a description

It defines classes_and_methods

@author:     user_name

@copyright:  2022 organization_name. All rights reserved.

@license:    license

@contact:    user_email
@deffield    updated: Updated
'''

from gfutils import pickle_util

import sys
import os
import correct_genotypes2

import numpy as np
import pandas as pd
from pedigree.pedigree_dag import PedigreeDAG
import pathlib

from collections import defaultdict
import multiprocessing
import concurrent.futures

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

from empirical.jalleledist3 import JointAllellicDistribution

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

__all__ = []
__version__ = 0.1
__date__ = '2022-07-26'
__updated__ = '2022-07-26'

TESTRUN = 0
PROFILE = 0

class CLIError(Exception):
    '''Generic exception to raise and log different fatal errors.'''
    def __init__(self, msg):
        super(CLIError).__init__(type(self))
        self.msg = "E: %s" % msg
    def __str__(self):
        return self.msg
    def __unicode__(self):
        return self.msg

def main(argv=None): # IGNORE:C0111
    '''Command line options.'''

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)

    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
    program_license = '''%s

  Created by user_name on %s.
  Copyright 2022 organization_name. All rights reserved.

  Licensed under the Apache License 2.0
  http://www.apache.org/licenses/LICENSE-2.0

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))

    # Setup argument parser
    parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-g", "--genotypes", dest="genotypes_input_file", help="genotypes in kranis format 0|1|2 with header in first row starting \"id\" and first column is animal id")
    parser.add_argument("-p", "--pedigree", dest="pedigree", required=True, help="pedigree kid,sire,dam,sex,phenotype space delimited file")
    parser.add_argument("-w", "--surroundsnps", dest="surround_size", type=int, default=3, help="number of snps either side of a snp to create window for empirical")
    parser.add_argument("-o", "--outdir", dest="outdir", type=str, required=True, help="outdir to create index")
    parser.add_argument("-s", "--snps", dest="snps", required=True, help="snp map file")
    parser.add_argument("-T", "--threads", dest="threads", type=int, required=False, default=multiprocessing.cpu_count(),  help="weight of empirical vs collected medelian error when ranking snps by error probability")
    parser.add_argument("-q", "--initquantilefilter", dest="initquantilefilter", type=float, required=False, default=0.9,  help="initial filter to select upper quantile in error likelihood dist")
    
    # Process arguments
    args = parser.parse_args()
    

    out_dir = args.outdir
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    init_filter_p=args.initquantilefilter
    filter_e=args.filter_e
    
    threads= args.threads

    genotypes_input_file = args.genotypes_input_file
    pedigree = args.pedigree
    surroundsnps = args.surround_size

    pedigree = PedigreeDAG.from_file(args.pedigree)
    genomein = pd.read_csv(args.snps, sep=',', names = ["snpid", "chrom","pos", "topAllele","B"], skiprows=1, engine='c',low_memory=False, memory_map=True)
    
    snps = [row["snpid"] for _index, row in genomein.iterrows()]#
    chromosome2snp = defaultdict(set)
    chromosomes = set([row["chrom"] for _index, row in genomein.iterrows()])
    chromosomesnps = {}
    for _index, row in genomein.iterrows():
        if row["chrom"] not in chromosomesnps:
            chromosomesnps[row["chrom"]] = 1
        else:
            chromosomesnps[row["chrom"]]+=1
        chromosome2snp[row["chrom"]].add(row["snpid"])
    print("%s" % [x for x in map(str,chromosome2snp.keys())] )

    
    ###################
    #calculate blankets for mendel errors
    ####################
    blankets = {}
    for kid in pedigree.males.union(pedigree.females):
        blanket = set() #pedigree.get_kids(kid) #pedigree.get_kids(kid)
        #blanket.update(pedigree.get_partners(kid))
        blanket.update([x for x in pedigree.get_parents(kid) if x is not None])
        blanket.discard(kid)
        blankets[kid] = blanket

    print("%s blankets done " % len(blankets))
    
    ###############
    
    quant95_t, quant99_t, quantQ = None,None,None
    
    for chromosome in sorted(chromosome2snp.keys()) :
        if chromosome == "" or chromosome == "-999" or chromosome == "-9" :
            continue
        print("calculating chromosome %s" % chromosome)
        if genotypes_input_file.endswith(".gz") :
            genotypes = pd.read_csv(genotypes_input_file, sep=" ", compression='gzip', header=0, index_col=0, engine="c", dtype={snp:np.uint8 for snp in snps}, low_memory=False, memory_map=True)
        else :
            genotypes = pd.read_csv(genotypes_input_file, sep=" ", header=0, index_col=0, engine="c", dtype={snp:np.uint8 for snp in snps}, low_memory=False, memory_map=True)
        print("Loaded genotype matrix with %s individuals X %s snps " %genotypes.shape)
        
        candidatesForEval = list()
        for kid in genotypes.index :
            sire, dam = pedigree.get_parents(kid)
            if sire in genotypes.index and dam in genotypes.index:
                candidatesForEval.append(kid)
            
        genotypes = genotypes.loc[candidatesForEval,chromosome2snp[chromosome]]
        print("genotype matrix for eval is %s individuals X %s snps " %genotypes.shape)
        probs = {}
        probs_errors = pd.DataFrame(np.zeros(genotypes.shape), columns=genotypes.columns, index=genotypes.index)
        cache_store = {}
        

        
        #populate_base_probs
        print("pre-calculate mendel probs on all individuals")
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads, 
                                                    initializer=correct_genotypes2.initializer,
                                                    initargs=(genotypes,pedigree, None)) as executor:
            #for kid in tqdm(pedigree.males.union(pedigree.females)):
            #    x, b = self.mendelProbsSingle(corrected_genotype, pedigree, kid, back)
            #    print("kid %s done" % kid)
            futures = {executor.submit(correct_genotypes2.CorrectGenotypes.mendelProbsSingle, kid, 2, elimination_order="MinNeighbors"):kid for kid in candidatesForEval}
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
                    #blanket of partners parents and kids
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
        probs_errors[genotypes == 9] = 1
        
        distribution_of_ranks = probs_errors.to_numpy().flatten()
        
        if quant95_t != None :
            quantQ_L = np.nanquantile(distribution_of_ranks, [init_filter_p], method='linear')
            quant95_t, quant99_t, quantQ = np.nanquantile(distribution_of_ranks, [0.95,0.99, init_filter_p], method='interpolated_inverted_cdf')
            ax = sns.distplot(distribution_of_ranks)
            ax.set(xlabel='sum difference in observed vs expected', ylabel='count')
            plt.axvline(quant95_t, 0,1, color="blue", alpha=0.5, linestyle="--")
            plt.axvline(quant99_t, 0,1, color="red", alpha=0.5, linestyle="--")
            plt.axvline(quantQ, 0,1, color="black")
            plt.axvline(quantQ_L, 0,1, color="yellow")
            plt.savefig("%s/distribution_of_sum_error_ranks_histogram_preld_based_on_chromosome_%s.png" % (out_dir, chromosome), dpi=300)
            plt.clf()
        
        print("initial P of errors calculated with 95%q = %s, 99q = %s, and cuttoff %sq = %s" % (quant95_t, quant99_t, init_filter_p, quantQ))
        
        print("calculating LDDist ")
        empC = JointAllellicDistribution(list(genotypes.columns),
                                        surround_size=surroundsnps,
                                        chromosome2snp=chromosome2snp)
        print("create mask")
        mask = np.array(probs_errors.to_numpy() <= quantQ, dtype=bool)
        print("calc empirical ld on genotype with %s of %s (%6.2f pc) over under cuttoff %6.6f mendel errors after removing > %s quantile of mendel errors" % (np.count_nonzero(mask), mask.size, (np.count_nonzero(mask)/mask.size)*100, filter_e, quantQ))
        #print("calc empirical ld on genotype with %s of %s (%6.2f pc) over under cuttoff 0 mendel errors" % (np.count_nonzero(mask), mask.size, (np.count_nonzero(mask)/mask.size)*100,))
        empC.countJointFrqAll(genotypes, mask)
        
        pathlib.Path("%s/%s" % (out_dir, chromosome)).mkdir(parents=True, exist_ok=True)
        pickle_util.dumpToPickle("%s/%s/empiricalIndex.idx" % (out_dir, chromosome), empC)
        
    
if __name__ == "__main__":
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = 'empirical.preindex_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())