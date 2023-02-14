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
import gzip
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

from pedigree.pedigree_dag import PedigreeDAG
import pathlib
import pandas as pd
import quickgsim as gsim
from collections import Counter, defaultdict
import numpy as np
from correct_genotypes3 import CorrectGenotypes
from typing import Tuple, List, Dict, Union
import multiprocessing
import pickle

from gfutils.gensrandaccess import GensCache

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
    '''
    @TODO specify preindex empirical as parameter
    @TODO specify list of indifividuals in gens.txt to correct
    @TODO correct individuals with parents first.
    '''

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
        parser.add_argument("-w", "--surroundsnps", dest="surround_size", type=int, default=3, help="number of snps either side of a snp to create window for empirical")
        parser.add_argument("-o", "--outdir", dest="outdir", required=True , help="outputdir")
        parser.add_argument("-d", "--lddisttype", dest="lddisttype",  default='global', type=str, choices=['none', 'global', 'local'], help="ld distance calculation type to use")
        parser.add_argument("-b", "--lookback", dest="lookback", type=int, default=2,  help="generations to look up/back")
        parser.add_argument("-t", "--tiethreshold", dest="tiethreshold", type=float, default=0.05,  help="error tolerance between probabilies to declare a tie")
        parser.add_argument("-i", "--genotypes_input_file", dest="genotypes_input_file", type=str, required=True,  help="genotype input file (SSV file, first column int ids, first row snp ids)")
        parser.add_argument("-n", "--firstnsnps", dest="first_n_snps", type=int, required=False, default=None,  help="only use the first n snps of the genome")
        parser.add_argument("-q", "--initquantilefilter", dest="initquantilefilter", type=float, required=False, default=0.9,  help="initial filter to select upper quantile in error likelihood dist")
        parser.add_argument("-Q", "--filter_e", dest="filter_e", type=float, required=False, default=0.8,  help="filter to select upper quantile to exclude from empirical ld calculation")
        parser.add_argument("-W", "--weightempirical", dest="weight_empirical", type=float, required=False, default=2,  help="weight of empirical vs collected medelian error when ranking snps by error probability")
        parser.add_argument("-E", "--elimination_order", dest="elimination_order", type=str, required=False, default="weightedminfill", choices=["weightedminfill","minneighbors","minweight","minfill"], help="elimination order in mendel prob calculation")
        parser.add_argument("-T", "--threads", dest="threads", type=int, required=False, default=multiprocessing.cpu_count(),  help="weight of empirical vs collected medelian error when ranking snps by error probability")
        parser.add_argument('-P', '--empC', dest="empC", required=True, help="folder with prior empirical disribution for ld snps /chr/*.idx.gz")
        parser.add_argument('-V', '--version', action='version', version=program_version_message)
        
        # Process arguments
        args = parser.parse_args()

        out_dir = args.outdir
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        
        pedigree = PedigreeDAG.from_file(args.pedigree)
        
        elimination_order = str(args.elimination_order)
        first_n_snps  = args.first_n_snps
        
        empCFile = args.empC
        
        threshold_singles = float(args.threshold_singles)
        threshold_pairs = float(args.threshold_pairs)
        surround_size = int(args.surround_size)
        tiethreshold = float(args.tiethreshold)
        genomein = pd.read_csv(args.snps, sep=' ', names = ["chrom", "snpid", "cm", "pos"], engine='c',low_memory=False, memory_map=True)
        init_filter_p = float(args.initquantilefilter) 
        weight_empirical = float(args.weight_empirical)
        filter_e = float(args.filter_e)
        threads = int(args.threads)
        
        genotypes_input_file = args.genotypes_input_file
        
        print("building cache index from %s: this may take some time" % genotypes_input_file)
        g_cache = GensCache(genotypes_input_file, header=True)
        print("DONE !!! building cache index from %s" % genotypes_input_file)
        print("Detected %s individuals and %s snps in input file" % (len(g_cache.all_ids), len(g_cache.snps)-1))
    
        lddist = args.lddisttype
        print("Will use ld distance type %s " % lddist)
        
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
        
#        if input_file.endswith(".gz"):
#            genotypes_with_errors = pd.read_csv(input_file, sep=" ", compression='gzip', header=0, index_col=0, engine="c", dtype={snp:np.uint8 for snp in snps}, low_memory=False, memory_map=True)
#        else :
#            genotypes_with_errors = pd.read_csv(input_file, sep=" ", header=0, index_col=0, engine="c", dtype={snp:np.uint8 for snp in snps}, low_memory=False, memory_map=True)
        
        #print("loaded genome matrix of size %s animals by %s snps" % genotypes_with_errors.shape)
        #if first_n_snps is not None:
        #    genotypes_with_errors = genotypes_with_errors.iloc[:,0:first_n_snps]
        #    print("reduced genome matrix to size %s animals by %s snps" % genotypes_with_errors.shape)
    
         
        #allelefrq = pd.DataFrame(np.array([genomematrix[snp].value_counts().values for snp in genomematrix.columns]), index=genomematrix.columns, columns=["0","1","2"])
        print("chromosomes found: %s " % chromosomes) 
             
        c = CorrectGenotypes(chromosome2snp=chromosome2snp, surround_size=surround_size, elimination_order=elimination_order)
        
        print("starting correct matrix") 
        
        tic = int(time.time())
        corrected_genotype = pd.DataFrame(data=g_cache.getMatrix(list(g_cache.all_ids)), 
                                         index=list(g_cache.all_ids),
                                         columns=g_cache.snps)
        
        counter = Counter(corrected_genotype.to_numpy().flatten())
        print("pre-corrected array: %s" % counter)
        
        def chunk(l, n, b=0, m=2):
            n = max(1, n)
            groups = filter( lambda x: len(x) > m, (l[i:] if len(l)-(i+n+b) < m else l[i:i+n+b] for i in range(0, len(l), n)))
            return list(groups)
        
        for chromosome, snpsOnchrom in sorted(chromosome2snp.items(), key=lambda x:len(x)): #EXCLUDE ZERO @TODO
            if chromosome == "0" or chromosome == "Z" :
                print("chromosome skipped %s" % chromosome)
                continue
            print("Chromosome %s with %s snps" % (chromosome, len(snpsOnchrom)))
            empC = pickle.load(gzip.open("%s/%s/empiricalIndex.idx.gz" % (empCFile,chromosome)))
            
            found_snps = [x for x in snps if x in g_cache.snps and x in chromosome2snp[chromosome]]
            chunks = chunk(found_snps, 1000, (surround_size*2)+1, (surround_size*2)+1)
            for i, snps in enumerate(chunks):
                print("correcting chunk %s of %s with %s snps in chunk" % ( i, len(chunks), len(snps)))
                
                genotypes = pd.DataFrame(data=g_cache.getMatrix(list(g_cache.all_ids)), 
                                         index=list(g_cache.all_ids),
                                         columns=g_cache.snps).loc[:,found_snps]
                
                result = c.correctMatrix(genotypes, pedigree, empC,
                                            threshold_pairs, threshold_singles,
                                            back=lookback, tiethreshold=tiethreshold, 
                                            init_filter_p=init_filter_p, filter_e=filter_e,
                                            weight_empirical=weight_empirical,
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