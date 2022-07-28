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
    
    threads= args.threads

    genotypes_input_file = args.genotypes_input_file
    pedigree = args.pedigree
    surroundsnps = args.surround_size

    pedigree = PedigreeDAG.from_file(args.pedigree)
    genomein = pd.read_csv(args.snps, sep=',', names = ["snpid", "chrom","pos", "topAllele","B"], skiprows=1, engine='c',low_memory=False, memory_map=True)
    genomein = genomein.sort_values(by=["chrom", "pos"], ascending=[True, True])
    print(genomein)
    
    snps = [row["snpid"] for _index, row in genomein.iterrows()]#
    chromosome2snp = defaultdict(list)
    #chromosomes = set([row["chrom"] for _index, row in genomein.iterrows()])
    chromosomesnps = {}
    for _index, row in genomein.iterrows():
        if row["chrom"] not in chromosomesnps:
            chromosomesnps[row["chrom"]] = 1
        else:
            chromosomesnps[row["chrom"]]+=1
        if row["snpid"] not in chromosome2snp[row["chrom"]] :
            chromosome2snp[row["chrom"]].append(row["snpid"])
        else :
            raise Exception("Error in map file %s appears multiple times in chromosome %s " % (row["snpid"],row["chrom"]))
    print("%s" % [x for x in map(str,chromosome2snp.keys())] )
    
    ###############
    
    quant95_t, quant99_t, quantQ = None,None,None
    maxsumprobs = 0
    
    for chromosome in sorted(chromosome2snp.keys()) :
        if chromosome == "" or chromosome == "-999" or chromosome == "-9" :
            continue
        snpsToImport = chromosome2snp[chromosome]
        filtercolumns = ["id"]+snpsToImport
        print("calculating chromosome %s: importing %s snps" % (chromosome, len(snpsToImport)))
        datatypes = {snp:np.uint8 for snp in snps} | {"id":np.int64}
        if genotypes_input_file.endswith(".gz") :
            genotypes = pd.read_csv(genotypes_input_file, usecols=filtercolumns,
                                    sep=" ", compression='gzip', header=0, index_col=0, engine="c", dtype=datatypes, low_memory=False, memory_map=True)
        else :
            genotypes = pd.read_csv(genotypes_input_file, usecols=filtercolumns,
                                     sep=" ", header=0, index_col=0, engine="c", dtype=datatypes, low_memory=False, memory_map=True)
        print("Loaded genotype matrix with %s individuals X %s snps " %genotypes.shape)
        
        candidatesForEval = list()
        for kid in genotypes.index :
            sire, dam = pedigree.get_parents(kid)
            if sire in genotypes.index and dam in genotypes.index:
                candidatesForEval.append(kid)
            
        genotypes = genotypes.loc[candidatesForEval,chromosome2snp[chromosome]]
        print("genotype matrix for eval is %s individuals X %s snps after only trio candidates retained" %genotypes.shape)
        probs_errors = pd.DataFrame(np.zeros(genotypes.shape), columns=genotypes.columns, index=genotypes.index)
        
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
                    __, probsErrors, __ = future.result()
                    probs_errors.loc[kid,:] = np.squeeze(probsErrors)
                    #blanket of partners parents and kids
                    del futures[future]
                    del future
            print("mendel precalc done")
        
        #minnonzeroprobs = np.min(maxsumprobs)
        maxsumprobs = np.nanmax((maxsumprobs, np.nanmax(probs_errors)))
        print("maximum ranking for error = %s" % maxsumprobs)
        
        probs_errors = np.log(np.log(probs_errors+1)+1)
        maxsumprobs = np.nanmax(probs_errors)
        probs_errors = probs_errors/np.nanmax(maxsumprobs)
        probs_errors[genotypes == 9] = 1
        probs_errors[genotypes == -9] = 1
        
        individualSumProbs = probs_errors.sum(axis=1).to_numpy()
        individualSumProbs = individualSumProbs/np.nanmax(individualSumProbs)
        
        quantQ_chromosome_individual = np.nanquantile(individualSumProbs, [init_filter_p], method='interpolated_inverted_cdf')
        ax = sns.distplot(individualSumProbs)
        ax.set(xlabel='sum difference in observed vs expected', ylabel='count')
        plt.axvline(quantQ_chromosome_individual, 0,1, color="black")
        plt.savefig("%s/%s/individuals_dist_histogram_preld_based_on_chromosome_%s.png" % (out_dir, chromosome, chromosome), dpi=300)
        plt.clf()
        
        distribution_of_ranks = probs_errors.to_numpy().flatten()
        
        if quant95_t is None :
            print("calculating quantiles")
            quant95_t, quant99_t, quantQ = np.nanquantile(distribution_of_ranks, [0.95,0.99, init_filter_p], method='interpolated_inverted_cdf')
            ax = sns.distplot(distribution_of_ranks)
            ax.set(xlabel='sum difference in observed vs expected', ylabel='count')
            plt.axvline(quant95_t, 0,1, color="blue", alpha=0.5, linestyle="--")
            plt.axvline(quant99_t, 0,1, color="red", alpha=0.5, linestyle="--")
            plt.axvline(quantQ, 0,1, color="black")
            plt.savefig("%s/distribution_of_sum_error_ranks_histogram_preld_based_on_chromosome_%s.png" % (out_dir, chromosome), dpi=300)
            plt.clf()
            
        quantQ_chromosome = np.nanquantile(distribution_of_ranks, [init_filter_p], method='interpolated_inverted_cdf')
        ax = sns.distplot(distribution_of_ranks)
        ax.set(xlabel='sum difference in observed vs expected', ylabel='count')
        plt.axvline(quantQ_chromosome, 0,1, color="black")
        plt.axvline(quantQ, 0,1, color="cyan")
        plt.savefig("%s/%s/distribution_of_sum_error_ranks_histogram_preld_based_on_chromosome_%s.png" % (out_dir, chromosome, chromosome), dpi=300)
        plt.clf()
        
        print("initial P of errors calculated with 95pc-quantile = %s, 99pc-quantile = %s, and cuttoff %s-quantile = %s" % (quant95_t, quant99_t, init_filter_p, quantQ))
        
        print("calculating LDDist ")
        empC = JointAllellicDistribution(list(genotypes.columns),
                                        surround_size=surroundsnps,
                                        chromosome2snp=chromosome2snp)
        print("create mask")
        mask = np.array(probs_errors.to_numpy() <= quantQ, dtype=bool)
        print("calc empirical ld on genotype with %s of %s (%6.2f pc) under cuttoff %6.6f mendel errors after removing > %s quantile of mendel errors" % 
              (np.count_nonzero(mask), mask.size, (np.count_nonzero(mask)/mask.size)*100,quantQ, init_filter_p))
        
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