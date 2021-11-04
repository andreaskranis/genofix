#!/usr/local/bin/python3
# encoding: utf-8
'''
genofix -- shortdesc

genofix correct a genotype matrix

It defines classes_and_methods

@author:     mhindle

@copyright:  2021 Ediburgh University. All rights reserved.

@license:    GPL

@contact:    matthew.hindle@roslin.ed.ac.uk
@deffield    updated: Updated
'''
import sys,os
import time
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

from pedigree.pedigree_dag import PedigreeDAG
import pathlib
import pandas as pd
import quickgsim as gsim
from collections import Counter, defaultdict
import numpy as np
from correct_genotypes2 import CorrectGenotypes
from typing import Tuple, List, Dict, Union
import multiprocessing

__all__: List[str] = []
__version__ = 0.1
__date__ = '2021-07-26'
__updated__ = '2021-07-26'

DEBUG = 0
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
  Copyright 2021 organization_name. All rights reserved.

  Licensed under the Apache License 2.0
  http://www.apache.org/licenses/LICENSE-2.0

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))

    try:
        # Setup argument parser
        parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument("-p", "--pedigree", dest="pedigree", required=True, help="pedigree file")
        parser.add_argument("-s", "--snps", dest="snps", required=True, help="snp map file")
        parser.add_argument("-l", "--thresholdsingles", dest="threshold_singles", type=float, default=0.7, help="threshold prob for singles")
        parser.add_argument("-m", "--thresholdpairs", dest="threshold_pairs", type=float, default=0.5, help="threshold prob for mating pairs")
        parser.add_argument("-w", "--surroundsnps", dest="surround_size", type=int, default=2, help="number of snps either side of a snp to create window for empirical")
        parser.add_argument("-o", "--outdir", dest="outdir", required=True , help="outputdir")
        parser.add_argument("-d", "--lddisttype", dest="lddisttype",  default='global', type=str, choices=['none', 'global', 'local'], help="ld distance calculation type to use")
        parser.add_argument("-b", "--lookback", dest="lookback", type=int, default=2,  help="generations to look up/back")
        parser.add_argument("-t", "--tiethreshold", dest="tiethreshold", type=float, default=0.05,  help="error tolerance between probabilies to declare a tie")
        parser.add_argument("-e", "--errorrate", dest="error_rate", type=float, default=1,  help="simulated error rate")
        parser.add_argument("-i", "--input", dest="input", type=str, required=False,  help="genotype input file (SSV file, first column int ids, first row snp ids)")
        parser.add_argument("-n", "--firstnsnps", dest="first_n_snps", type=int, required=False, default=None,  help="only use the first n snps of the genome")
        parser.add_argument("-q", "--initquantilefilter", dest="initquantilefilter", type=float, required=False, default=0.9,  help="initial filter to select upper quantile in error likelihood dist")
        parser.add_argument("-Q", "--filter_e", dest="filter_e", type=float, required=False, default=0.8,  help="filter to select upper quantile to exclude from empirical ld calculation")
        parser.add_argument("-W", "--weightempirical", dest="weight_empirical", type=float, required=False, default=2,  help="weight of empirical vs collected medelian error when ranking snps by error probability")
        parser.add_argument("-M", "--minimum_cluster_size", dest="minimum_cluster_size", type=float, required=False,  help="set to turn on ld partitioning of pedigree (recomend 10-15 clusters with min 150)")
        parser.add_argument("-E", "--elimination_order", dest="elimination_order", type=str, required=False, default="weightedminfill", choices=["weightedminfill","minneighbors","minweight","minfill"], help="elimination order in mendel prob calculation")
        parser.add_argument("-T", "--threads", dest="threads", type=int, required=False, default=multiprocessing.cpu_count(),  help="weight of empirical vs collected medelian error when ranking snps by error probability")
        parser.add_argument('-V', '--version', action='version', version=program_version_message)
        
        # Process arguments
        args = parser.parse_args()

        out_dir = args.outdir
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        
        pedigree = PedigreeDAG.from_file(args.pedigree)
        elimination_order = str(args.elimination_order)
        first_n_snps  = args.first_n_snps
        
        threshold_singles = float(args.threshold_singles)
        threshold_pairs = float(args.threshold_pairs)
        surround_size = int(args.surround_size)
        tiethreshold = float(args.tiethreshold)
        genomein = pd.read_csv(args.snps, sep=' ', names = ["chrom", "snpid", "cm", "pos"], engine='c',low_memory=False, memory_map=True)
        init_filter_p = float(args.initquantilefilter) 
        weight_empirical = float(args.weight_empirical)
        filter_e = float(args.filter_e)
        threads = int(args.threads)
        
        if args.minimum_cluster_size is not None:
            minimum_cluster_size = int(args.minimum_cluster_size)
            partition_pedigree = True
            print("partitions for ld with min size of %s" % minimum_cluster_size)
        else:
            partition_pedigree = False
            minimum_cluster_size = None
            print("partitions for ld disabled")

        input_file = args.input
        
        lddist = args.lddisttype
        print("Will use ld distance type %s " % lddist)
        
        error_rate = args.error_rate
        print("Will use error rate %s: "  % error_rate)
        
        lookback = args.lookback
        print("Will %s generations up/back: " % lookback)
        
        snps = [row["snpid"] for _index, row in genomein.iterrows()]
        chromosome2snp = defaultdict(set)
        chromosomes = set([row["chrom"] for _index, row in genomein.iterrows()])
        chromosomesnps = {}
        for _index, row in genomein.iterrows():
            if row["chrom"] not in chromosomesnps:
                chromosomesnps[row["chrom"]] = 1
            else:
                chromosomesnps[row["chrom"]]+=1
            chromosome2snp[row["chrom"]].add(row["snpid"])
        print(chromosomesnps)
                
        genome = gsim.genome.Genome()
        
        for _index, row in genomein.iterrows():
            if row["cm"] > 0:
                genome.add_variant(int(row["chrom"]), snpid=row["snpid"], cm_pos=float(row["cm"])+1)
        
        for chrom in genome.chroms.values():
            chrom.finalise_chrom_configuration()
            
        for chrom in genome.chroms.keys():
            genome.chroms[chrom].cm_pos = genome.chroms[chrom].cm_pos+abs(min(genome.chroms[chrom].cm_pos))
            genome.chroms[chrom].cm_pos =  genome.chroms[chrom].cm_pos/sum(genome.chroms[chrom].cm_pos) 
        
        chromosomes = sorted([int(chro) for chro in genome.chroms.keys()])
        print("chromosomes: %s " % chromosomes)
        
        genotypes_with_errors = pd.read_csv(input_file, sep=" ", compression='gzip', header=0, index_col=0, engine="c", dtype={snp:np.uint8 for snp in snps}, low_memory=False, memory_map=True)
        print("loaded genome matrix of size %s animals by %s snps" % genotypes_with_errors.shape)
        if first_n_snps is not None:
            genotypes_with_errors = genotypes_with_errors.iloc[:,0:first_n_snps]
            print("reduced genome matrix to size %s animals by %s snps" % genotypes_with_errors.shape)
    
         
        #allelefrq = pd.DataFrame(np.array([genomematrix[snp].value_counts().values for snp in genomematrix.columns]), index=genomematrix.columns, columns=["0","1","2"])
        print("chromosomes found: %s " % chromosomes) 
             
        c = CorrectGenotypes(chromosome2snp=chromosome2snp, surround_size=surround_size, min_cluster_size=minimum_cluster_size, elimination_order=elimination_order)
        
        print("starting correct matrix") 
        
        counter = Counter(genotypes_with_errors.to_numpy().flatten())
        print("pre-corrected array: %s" % counter)
        
        tic = int(time.time())
        corrected_genotype = pd.DataFrame(data=np.full(genotypes_with_errors.shape, 10, dtype=np.uint8),index=genotypes_with_errors.index, columns=genotypes_with_errors.columns)
        
        def chunk(l, n, b=0, m=2):
            n = max(1, n)
            groups = filter( lambda x: len(x) > m, (l[i:] if len(l)-(i+n+b) < m else l[i:i+n+b] for i in range(0, len(l), n)))
            return list(groups)
        chunks = chunk(genotypes_with_errors.columns, 1000, (surround_size*2)+1, (surround_size*2)+1)
        for i, snps in enumerate(chunks):
            print("correcting chunk %s of %s with %s snps in chunk" % ( i, len(chunks), len(snps)))
            result = c.correctMatrix(genotypes_with_errors.loc[:,snps], 
                                                            pedigree, 
                                                            threshold_pairs, threshold_singles,
                                                            lddist, back=lookback, tiethreshold=tiethreshold, 
                                                            init_filter_p=init_filter_p, filter_e=filter_e,
                                                            weight_empirical=weight_empirical,partition_pedigree=partition_pedigree,
                                                            threads=threads, DEBUGDIR=out_dir)
            if i == 0:
                corrected_genotype.loc[:,snps] = result
            else :
                corrected_genotype.loc[:,snps[surround_size+1:]] = result[surround_size+1:]
            
        toc = int(time.time())
        print("done correct matrix in %s minutes" % ((toc-tic)/60) )            
        corrected_genotype.to_csv("%s/simulatedgenome_corrected_errors_threshold.ssv" % (out_dir), sep=" ")
                 
        counter = Counter(corrected_genotype.to_numpy().flatten())
        print("corrected array: %s" % counter)
        
        return(0)
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception as e:
        raise(e)
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use --help")
        return 2

if __name__ == "__main__":
    if DEBUG:
        sys.argv.append("-h")
        sys.argv.append("-v")
        sys.argv.append("-r")
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = 'simulate.simulate_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())