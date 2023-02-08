#!/usr/local/bin/python2.7
# encoding: utf-8
'''
pedigree.error_checker -- highlights pedigree errors in pops

pedigree.error_checker is a highlights possible pedigree errors in pops by mendel stats

@author:     mhindle

@copyright:  2022 aviagen All rights reserved.

@license:    license

@contact:    matthew.hindle@aviagen.com
@deffield    updated: 10/2022
'''

import sys
import os
import pathlib
import multiprocessing
from collections import defaultdict

from pedigree.pedigree_dag import PedigreeDAG
from gfutils.gensrandaccess import GensCache

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

import numpy as np
import pandas as pd

from tqdm import tqdm
import multiprocessing
import concurrent.futures

import correct_genotypes2

__all__ = []
__version__ = 0.1
__date__ = '2022-10-24'
__updated__ = '2022-10-24'

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

    parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-g", "--genotypes", dest="genotypes_input_file", help="genotypes in kranis format 0|1|2 with header in first row starting \"id\" and first column is animal id")
    parser.add_argument("-p", "--pedigree", dest="pedigree", required=True, help="pedigree kid,sire,dam,sex,phenotype space delimited file")
    parser.add_argument("-k", "--kids", dest="kids", required=True, help="kids to check uses first column of space delimited file (i.e. you can reuse pedigree file if you want)")
    parser.add_argument("-s", "--snps", dest="snps", required=True, help="snp map file")
    parser.add_argument("-T", "--threads", dest="threads", type=int, required=False, default=multiprocessing.cpu_count(),  help="weight of empirical vs collected medelian error when ranking snps by error probability")
    parser.add_argument("-c", "--chromosome", dest="chromosome", type=str, required=True, default="1",  help="Chromosome to build statistics on")
    parser.add_argument("-o", "--outdir", dest="outdir", type=str, required=True, help="outdir to create index")
    parser.add_argument("-d", "--delimiter", dest="delimiter_pedigree", default=' ', type=str, required=False, help="delimiter")
    
    # Process arguments
    args = parser.parse_args()
    
    out_dir = args.outdir
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    threads= args.threads
    delimiter_pedigree = args.delimiter_pedigree
 
    genotypes_input_file = args.genotypes_input_file
    pedigree = args.pedigree
    kids_file = args.kids
    kids = pd.read_csv(args.kids, sep=delimiter_pedigree).iloc[:, 0].to_list()
    print(kids)
    
    chromosome2test = args.chromosome
    print("statistics will be based on chromosome: %s" % chromosome2test)
    
    print("building cache index from %s: this may take some time" % genotypes_input_file)
    g_cache = GensCache(genotypes_input_file, header=True)
    print("DONE !!! building cache index from %s" % genotypes_input_file)
    
    print("Detected %s individuals and %s snps in input file" % (len(g_cache.all_ids), len(g_cache.snps)-1))
    
    pedigree = PedigreeDAG.from_file(args.pedigree, delimiter=delimiter_pedigree)
    genomein = pd.read_csv(args.snps, sep=",", names = ["snpid", "chrom","pos", "topAllele","B"], skiprows=1, engine='c',low_memory=False, memory_map=True)
    genomein = genomein.sort_values(by=["chrom", "pos"], ascending=[True, True])
    #print(genomein)
    
    snps = [sys.intern(row["snpid"]) for _index, row in genomein.iterrows()]#
    chromosome2snp = defaultdict(list)
    #chromosomes = set([row["chrom"] for _index, row in genomein.iterrows()])
    chromosomesnps = {}
    for _index, row in genomein.iterrows():
        if row["chrom"] not in chromosomesnps:
            chromosomesnps[row["chrom"]] = 1
        else:
            chromosomesnps[row["chrom"]] += 1
        if row["snpid"] not in chromosome2snp[row["chrom"]] :
            chromosome2snp[row["chrom"]].append(row["snpid"])
        else :
            raise Exception("Error in map file %s appears multiple times in chromosome %s " % (row["snpid"],row["chrom"]))
    
    print("chromsome has %s snps" % chromosomesnps[chromosome2test] )

    animalswithparents = list()
    for kid in [x for x in g_cache.all_ids if x in kids]  :
        sire, dam = pedigree.get_parents(kid)
        if sire in g_cache.all_ids and dam in g_cache.all_ids:
            animalswithparents.append(kid)
    
    print("Found %s out of %s kids have genotypes (%s kids) and trios" % (len(animalswithparents), len(kids), len([x for x in g_cache.all_ids if x in kids]) ))

    candidate_kids = set()
    for kid in animalswithparents:
        candidate_kids.add(kid)
        sire, dam = pedigree.get_parents(kid)
        if sire in g_cache.all_ids:
            candidate_kids.add(sire)
        if dam in g_cache.all_ids:
            candidate_kids.add(dam)

    genotypes = pd.DataFrame(data=g_cache.getMatrix(list(candidate_kids)), index=list(candidate_kids), columns=g_cache.snps)
    print("Loaded genotype matrix with %s individuals X %s snps " % genotypes.shape)
    genotypes = genotypes.loc[list(candidate_kids),chromosome2snp[chromosome2test]]
    print("Filtered to genotype matrix with %s individuals X %s snps " % genotypes.shape)
               
    probs_errors = pd.DataFrame(np.zeros(genotypes.shape, dtype=np.float16), columns=genotypes.columns, index=genotypes.index)
                
    #populate_base_probs
    print("pre-calculate mendel probs on %s individuals" % len(candidate_kids))
    with concurrent.futures.ProcessPoolExecutor(max_workers=threads, 
                                                initializer=correct_genotypes2.initializer,
                                                initargs=(genotypes,pedigree, None)) as executor:
        #for kid in tqdm(pedigree.males.union(pedigree.females)):
        #    x, b = self.mendelProbsSingle(corrected_genotype, pedigree, kid, back)
        #    print("kid %s done" % kid)
        futures = {executor.submit(correct_genotypes2.CorrectGenotypes.mendelProbsSingle, kid, 2, elimination_order="MinNeighbors"):kid for kid in candidate_kids}
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
                probs_errors.loc[kid,:] = np.squeeze(probsErrors).astype(np.float16) # 
                #blanket of partners parents and kids
                del probsErrors
                del futures[future]
                del future
                del __
        print("mendel precalc done")
    
    #minnonzeroprobs = np.min(maxsumprobs)
    maxsumprobs = np.nanmax(probs_errors)
    print("maximum ranking for error = %s" % maxsumprobs)
    
    probs_errors = np.log(np.log(probs_errors+1)+1)
    maxsumprobs = np.nanmax(probs_errors)
    probs_errors = probs_errors/np.nanmax(maxsumprobs)
    probs_errors[genotypes == 9] = -1
    probs_errors[genotypes == -9] = -1
    
    probs_errors.loc[animalswithparents,:].to_csv("%s/probs_errors.csv" % out_dir, index=True, header=True)  
    
    individualSumProbs = probs_errors.loc[animalswithparents,:].sum(axis=1).to_numpy()
    individualSumProbs = individualSumProbs/np.nanmax(individualSumProbs)
    
    np.savetxt("%s/individual_sum_probs_errors.csv" % out_dir, individualSumProbs, delimiter=",")  
    
if __name__ == "__main__":
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = 'pedigree.error_checker_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())