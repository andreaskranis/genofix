#!/usr/local/bin/python3
# encoding: utf-8
'''
simulate.simulate -- shortdesc

simulate.simulate is a description

It defines classes_and_methods

@author:     user_name

@copyright:  2021 organization_name. All rights reserved.

@license:    license

@contact:    user_email
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
import math
from itertools import chain
from correct_genotypes2 import CorrectGenotypes
from typing import Tuple, List, Dict, Union
import multiprocessing
from tqdm import tqdm
import gzip

from quickgsim import importers

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
        parser.add_argument("-H", "--founders_haplo", dest="founders_haplo_file", required=False, help="founders haplotype file..space delimited, two lines per individual mat/pat, columns SNPs, non-founders will be ignored")
        parser.add_argument("-G", "--founders_geno", dest="founders_geno_file", required=False, help="founders genotype file..space delimited, one lines per individual mat/pat random assigned, columns SNPs, non-founders will be ignored")
        parser.add_argument("-p", "--pedigree", dest="pedigree", required=True, help="pedigree file")
        parser.add_argument("-s", "--snps", dest="snps", required=True, help="snp map file")
        parser.add_argument("-c", "--chunk", dest="chunk", required=False,default=1000,type=int, help="chunk size to process snps in..decrease if you have memory issues")
        parser.add_argument("-l", "--thresholdsingles", dest="threshold_singles", type=float, default=0.7, help="threshold prob for singles")
        parser.add_argument("-m", "--thresholdpairs", dest="threshold_pairs", type=float, default=0.5, help="threshold prob for mating pairs")
        parser.add_argument("-w", "--surroundsnps", dest="surround_size", type=int, default=2, help="number of snps either side of a snp to create window for empirical")
        parser.add_argument("-o", "--outdir", dest="outdir", required=True , help="outputdir")
        parser.add_argument("-d", "--lddisttype", dest="lddisttype",  default='global', type=str, choices=['none', 'global', 'local'], help="ld distance calculation type to use")
        parser.add_argument("-b", "--lookback", dest="lookback", type=int, default=2,  help="generations to look up/back")
        parser.add_argument("-t", "--tiethreshold", dest="tiethreshold", type=float, default=0.05,  help="error tolerance between probabilies to declare a tie")
        parser.add_argument("-e", "--errorrate", dest="error_rate", type=float, default=1,  help="simulated error rate")
        parser.add_argument("-et", "--error_thresh", dest="err_thresh", type=float, default=0.1,  help="error tolerance to co-rank top states in blanket")
        parser.add_argument("-g", "--popgenome", dest="prior_pop_genome", type=str, required=False,  help="skip population genomes generation and use specified file")
        parser.add_argument("-z", "--popgenomeerrors", dest="prior_genome_errors", type=str, required=False,  help="skip insertion of errors and use specified file. NB error_rate option will be ignored")
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
        
        founders_haplo_file = args.founders_haplo_file
        founders_geno_file = args.founders_geno_file
        elimination_order = str(args.elimination_order)
        first_n_snps  = args.first_n_snps
        chunk_n = args.chunk
    
        threshold_singles = float(args.threshold_singles)
        threshold_pairs = float(args.threshold_pairs)
        surround_size = int(args.surround_size)
        tiethreshold = float(args.tiethreshold)
        genomein = pd.read_csv(args.snps, sep=' ', names = ["chrom", "snpid", "cm", "pos"], engine='c',low_memory=False, memory_map=True)
        init_filter_p = float(args.initquantilefilter) 
        weight_empirical = float(args.weight_empirical)
        filter_e = float(args.filter_e)
        threads = int(args.threads)
        err_thresh = float(args.err_thresh)
        if args.minimum_cluster_size is not None:
            minimum_cluster_size = int(args.minimum_cluster_size)
            partition_pedigree = True
            print("partitions for ld with min size of %s" % minimum_cluster_size)
        else:
            partition_pedigree = False
            minimum_cluster_size = None
            print("partitions for ld disabled")

        prior_genome = args.prior_pop_genome
        prior_genome_errors = args.prior_genome_errors

        lddist = args.lddisttype
        print("Will use ld distance type %s " % lddist)
        
        error_rate = args.error_rate
        print("Will use error rate %s: "  % error_rate)
        
        lookback = args.lookback
        print("Will %s generations up/back: " % lookback)
        
        snps = [row["snpid"] for index, row in genomein.iterrows()]
        chromosome2snp = defaultdict(set)
        chromosomes = set([row["chrom"] for index, row in genomein.iterrows()])
        chromosomesnps = {}
        for index, row in genomein.iterrows():
            if row["chrom"] not in chromosomesnps:
                chromosomesnps[row["chrom"]] = 1
            else:
                chromosomesnps[row["chrom"]]+=1
            chromosome2snp[row["chrom"]].add(row["snpid"])
        print(chromosomesnps)
                
        genome = gsim.genome.Genome()
        
        for index, row in genomein.iterrows():
            if row["cm"] > 0:
                genome.add_variant(int(row["chrom"]), snpid=row["snpid"], cm_pos=float(row["cm"])+1)
        
        for chrom in genome.chroms.values():
            chrom.finalise_chrom_configuration()
            
        for chrom in genome.chroms.keys():
            genome.chroms[chrom].cm_pos = genome.chroms[chrom].cm_pos+abs(min(genome.chroms[chrom].cm_pos))
            genome.chroms[chrom].cm_pos =  genome.chroms[chrom].cm_pos/sum(genome.chroms[chrom].cm_pos) 
        
        rs = np.random.Generator(np.random.PCG64(1234)) 
        
        if prior_genome is None:
            if founders_haplo_file is not None: # read in haplotypes
                gens = importers.read_real_haplos(founders_haplo_file, 
                                   genome, first_haplo='maternal', 
                                   mv=9, sep=' ', header=False, random_assign_missing=True)
                print("imported %s haplotypes" % len(gens.keys()))
            elif founders_geno_file is not None: 
                gens = importers.read_real_genos(founders_geno_file, 
                                   genome, mv=9, sep=' ', header=False)
                print("imported %s genotypes" % len(gens.keys()))
            else: #generate founders from random haplotypes
                founder_no_sire = [x for x in pedigree.males.union(pedigree.females) if x not in pedigree.kid2sire.keys()]
                founder_no_dam = [x for x in pedigree.males.union(pedigree.females) if x not in pedigree.kid2dam.keys()]
                
                gens = {int(x):None for x in list(founder_no_sire) + list(founder_no_dam)}
                
                for founder in list(founder_no_sire) + list(founder_no_dam):
                    g = gsim.Genotype(chromosomes)
                    for chromosome in chromosomes:
                        g.data[chromosome][0] = rs.integers(size=genome.chroms[chromosome].nvars, low=0, high=2)
                        g.data[chromosome][1] = rs.integers(size=genome.chroms[chromosome].nvars, low=0, high=2)
                    gens[founder] = g
                print("generated %s haplotypes" % len(gens.keys()))
            
            genders = {x:1 for x in pedigree.males if x in gens.keys()}
            genders.update({x:2 for x in pedigree.females if x in gens.keys()})        
            
            founders = gsim.create_founders(genders,gens,genome)
            
            genotypes = founders.copy()
            
            print("genotypes %s" % len(genotypes.keys()))
            
            def addMissingKids(trios:list):
                created=0
                for kid, sire, dam, sex in trios:
                    if int(kid) not in genotypes.keys() and int(sire) in genotypes.keys() and int(dam) in genotypes.keys():
                        genotypes[kid] = gsim.mate(genotypes[int(sire)], genotypes[int(dam)], int(kid), genome, int(sex))
                        created+=1
                print("created %s" % created)
                return(created)
            
            #mate to fill missing untill there are no more kids to make....
            while addMissingKids(pedigree.as_ped()) != 0: pass
        
            print("%s genotypes generated for %s individuals" % (len(genotypes.keys()), len(pedigree.males)+len(pedigree.females)))
        
            if len(genotypes.keys()) != len(pedigree.males)+len(pedigree.females):
                print("males missing and cannot be generated %s" % [kid for kid in pedigree.males if kid not in genotypes.keys()])
                print("females missing and cannot be generated %s" % [kid for kid in pedigree.males if kid not in genotypes.keys()])
            #when the founders aren't the top nodes in the pedigree we need to trim these
            #pedigree = pedigree.get_subset(list(genotypes.keys()), balance_parents=False)
            
            chromosomes = sorted([int(chro) for chro in genome.chroms.keys()])
            print(chromosomes)
            
            snpids = list(chain(*[genome.chroms[chro].snpids for chro in chromosomes]))
            
            with gzip.open("%s/simulated_genotype_crossovers.txt.gz" % out_dir, "wt") as fout:
                def xtoString(chro, data):
                    switch, points = data
                    return(str(chro)+":"+"-".join(map(str,switch))+":"+",".join(map(str,points)))
                
                for kid in genotypes.keys():
                    fout.write(str(kid))
                    if len(genotypes[kid].mcr) > 0:
                        for chro in chromosomes:
                            fout.write("\t"+xtoString(chro,genotypes[kid].mcr[chro]))
                    else:
                        fout.write("\t")
                    fout.write(str(kid))
                    if len(genotypes[kid].pcr) > 0:
                        for chro in chromosomes:
                            fout.write("\t"+xtoString(chro,genotypes[kid].pcr[chro]))
                    fout.write("\n")
            #data_crossovers_P = {kid:[[xtoString(chro, data) for data in genotypes[kid].pcr[chro]] for chro in chromosomes if chro < len(genotypes[kid].pcr in genotypes[kid].pcr] for kid in genotypes.keys()}
            #data_crossovers_M = {kid:np.concatenate([genotypes[kid].mcr[chro] for chro in chromosomes]) for kid in genotypes.keys()}
            
            dataM = {kid:np.concatenate([genotypes[kid].genotype[chro][genotypes[kid].genotype.maternal_strand] for chro in chromosomes]) for kid in genotypes.keys()}
            for kid in dataM.keys():
                if np.greater(dataM[kid], 1).any():
                    print(dataM[kid][np.greater(dataM[kid], 1)])
                    raise Exception("maternal haplotype greater than 1????")
            dataP = {kid:np.concatenate([genotypes[kid].genotype[chro][genotypes[kid].genotype.paternal_strand] for chro in chromosomes]) for kid in genotypes.keys()}
            for kid in dataP.keys():
                if np.greater(dataP[kid], 1).any():
                    print(dataP[kid][np.greater(dataP[kid], 1)])
                    raise Exception("paternal haplotype greater than 1????")
            data = {kid:np.concatenate([np.add(genotypes[kid].genotype[chro][genotypes[kid].genotype.maternal_strand], genotypes[kid].genotype[chro][genotypes[kid].genotype.paternal_strand]) for chro in chromosomes]) for kid in genotypes.keys()}
            for kid in data.keys():
                if np.greater(data[kid], 2).any():
                    print(data[kid][np.greater(data[kid], 2)])
                    raise Exception("genotype greater than 2????")

            
            del genotypes
            print("building genotype matrix")
            
            genomematrix = pd.DataFrame.from_dict(data, orient='index', columns= snpids, dtype=np.uint8)
            genomematrixM = pd.DataFrame.from_dict(dataM, orient='index', columns= snpids, dtype=np.uint8)
            genomematrixP = pd.DataFrame.from_dict(dataP, orient='index', columns= snpids, dtype=np.uint8)
            
            del data
            del dataM
            del dataP
            
            print("save genotype matrix of size %s animals by %s snps" % genomematrix.shape)
            genomematrix.to_csv("%s/simulated_genotype_genome.ssv.gz" % out_dir, sep=" ", compression='gzip')
            print("save maternal haplotype matrix of size %s animals by %s snps" % genomematrix.shape)
            genomematrixM.to_csv("%s/simulated_maternal_haplotype_genome.ssv.gz" % out_dir, sep=" ", compression='gzip')
            print("save paternal haplotype matrix of size %s animals by %s snps" % genomematrix.shape)
            genomematrixP.to_csv("%s/simulated_paternal_haplotype_genome.ssv.gz" % out_dir, sep=" ", compression='gzip')
            del genomematrixM
            del genomematrixP
            
            if first_n_snps is not None:
                print("restricting snps to first %s" % first_n_snps)
                genomematrix = genomematrix.iloc[:,0:first_n_snps] ##first 50 snps

        else :
            genomematrix = pd.read_csv(prior_genome, sep=" ", compression='gzip', header=0, index_col=0, engine="c", dtype={snp:np.uint8 for snp in snps}, low_memory=False, memory_map=True)
            print("loaded genome matrix of size %s animals by %s snps" % genomematrix.shape)
            if first_n_snps is not None:
                genomematrix = genomematrix.iloc[:,0:first_n_snps]
                print("reduced genome matrix to size %s animals by %s snps" % genomematrix.shape)
        
         
        #allelefrq = pd.DataFrame(np.array([genomematrix[snp].value_counts().values for snp in genomematrix.columns]), index=genomematrix.columns, columns=["0","1","2"])
        print("chromosomes found: %s " % chromosomes) 
        
        if prior_genome_errors is None:
            errors = math.ceil(genomematrix.size*(error_rate/100))
            print("will insert %spc errors or %s snps " % (error_rate, errors))
            individuals = rs.choice(genomematrix.index, size=errors, replace=True)
            positionsmutate = rs.choice(genomematrix.columns, size=errors, replace=True)
            genotypes_with_errors = genomematrix.copy()
            
            print("inserting errors into simulated genotypes...")
            for kid, SNP_id in tqdm(list(zip(individuals, positionsmutate))):
                actual = genotypes_with_errors.loc[kid, SNP_id]
                target = actual
                while target == actual:
                    target = rs.integers(size=1, low=0, high=3)
                genotypes_with_errors.loc[kid, SNP_id] = target
            
            genotypes_with_errors.to_csv("%s/simulatedgenome_with_%serrors.ssv.gz" % (out_dir, error_rate), sep=" ", compression='gzip')
        else:
            genotypes_with_errors = pd.read_csv(prior_genome_errors, sep=" ", compression='gzip', header=0, index_col=0, engine="c", dtype={snp:np.uint8 for snp in snps}, low_memory=False, memory_map=True)
            print("loaded error matrix of size %s animals by %s snps" % genomematrix.shape)
            if first_n_snps is not None:
                genotypes_with_errors = genotypes_with_errors.iloc[:,0:first_n_snps]
                print("reduced error matrix to size %s animals by %s snps" % genotypes_with_errors.shape)
            
        difference = genomematrix - genotypes_with_errors
        print("%s errors %1.2fpc in array" % (np.count_nonzero(difference), (np.count_nonzero(difference)/difference.size)*100))
        
        with open("%s/statistics.tsv" % out_dir, "wt") as statout:
            headers = ("threshold_pair",
                       "threshold_single",
                       "prob_tie_threshold",
                       "init_filter_p",
                       "weight_empirical",
                       "prefilter_empirical",
                       "ld_distance_mode",
                             "total_observations",
                             "positives",
                              "positive_nine", 
                               "negatives",
                               "old_error_n",
                               "new_error_n",
                               "false_positives",
                               "true_positives",
                               "false_negatives",
                               "true_negatives",
                               "precision", 
                               "recall", 
                               "fscore", 
                               "true_positives_nine",
                               "precision_9", 
                               "recall_9", 
                               "fscore_9",
                               "time")
            statout.write('\t'.join(headers)+'\n')
            statout.flush()
            
            c = CorrectGenotypes(chromosome2snp=chromosome2snp, surround_size=surround_size, min_cluster_size=minimum_cluster_size, elimination_order=elimination_order)
            
            statout.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % 
                      (threshold_pairs, threshold_singles, tiethreshold, init_filter_p, weight_empirical,filter_e,lddist,
                       np.product(genomematrix.shape) ))
            statout.flush()

            
            print("starting correct matrix") 
            
            counter = Counter(genotypes_with_errors.to_numpy().flatten())
            print("pre corrected array: %s" % counter)
            
            #pedigree = pedigree.get_subset(list(genotypes_with_errors.index), balance_parents=False)
            
            tic = int(time.time())
            corrected_genotype = pd.DataFrame(data=np.full(genotypes_with_errors.shape, 10, dtype=np.uint8),index=genotypes_with_errors.index, columns=genotypes_with_errors.columns)
            
            def chunk(l, n, b=0, m=2):
                n = max(1, n)
                groups = filter( lambda x: len(x) > m, (l[i:] if len(l)-(i+n+b) < m else l[i:i+n+b] for i in range(0, len(l), n)))
                return list(groups)
            chunks = chunk(genotypes_with_errors.columns, chunk_n, (surround_size*2)+1, (surround_size*2)+1)
            for i, snps in enumerate(chunks):
                print("correcting chunk %s of %s with %s snps in chunk" % ( i+1, len(chunks), len(snps)))
                result = c.correctMatrix(genotypes_with_errors.loc[:,snps], 
                                                                pedigree, 
                                                                threshold_pairs, threshold_singles,
                                                                lddist, back=lookback, tiethreshold=tiethreshold, 
                                                                init_filter_p=init_filter_p, filter_e=filter_e,
                                                                weight_empirical=weight_empirical,partition_pedigree=partition_pedigree,
                                                                threads=threads, err_thresh=err_thresh, 
                                                                DEBUGDIR=out_dir, debugreal=genomematrix)
                if i == 0:
                    corrected_genotype.loc[:,snps] = result.loc[:,snps]
                else :
                    corrected_genotype.loc[:,snps[surround_size+1:]] = result.loc[:,snps[surround_size+1:]]
            
            toc = int(time.time())
            print("done correct matrix in %s minutes" % ((toc-tic)/60) )            
            corrected_genotype.to_csv("%s/simulatedgenome_corrected_errors_threshold.ssv" % (out_dir), sep=" ")
                     
            counter = Counter(corrected_genotype.to_numpy().flatten())
            print("corrected array: %s" % counter)
            
            difference_after_correction = genotypes_with_errors == corrected_genotype
            difference_true_errors = genomematrix == genotypes_with_errors
            difference_error_post_correction = genomematrix == corrected_genotype
            
            old_error_n = np.count_nonzero(np.logical_not(difference_true_errors))
            new_error_n = np.count_nonzero(np.logical_not(difference_error_post_correction))
            
            n_notchanging = np.count_nonzero(np.bitwise_and(genomematrix == genotypes_with_errors,genomematrix == corrected_genotype))
            
            positives = np.count_nonzero(difference_after_correction)
            positive_nine = np.count_nonzero(corrected_genotype == 9)
            negatives = np.count_nonzero(np.logical_not(difference_after_correction))
            
            correct_in_errormatrix = np.count_nonzero(difference_true_errors)
            correct_in_corrected = np.count_nonzero(difference_after_correction)
            error_in_errormatrix = np.count_nonzero(np.logical_not(difference_true_errors))
            error_in_corrected = np.count_nonzero(np.logical_not(difference_after_correction))
            
            still_wrong = np.logical_and(np.logical_not(difference_error_post_correction), np.logical_not(difference_true_errors))
            nolonger_wrong = np.logical_and(difference_error_post_correction, np.logical_not(difference_true_errors))
            new_wrong = np.logical_and(np.logical_not(difference_error_post_correction), difference_true_errors)
            still_right = np.logical_and(difference_error_post_correction, difference_true_errors)
            
            nolonger_wrong_nine = np.logical_and(np.logical_or(difference_error_post_correction,corrected_genotype == 9), np.logical_not(difference_true_errors))
            
            print("correct_in_errormatrix "+str(correct_in_errormatrix))
            print("correct_in_corrected "+str(correct_in_corrected))
            print("error_in_errormatrix "+str(error_in_errormatrix))
            print("error_in_corrected "+str(error_in_corrected))
            
            false_positives  = np.count_nonzero(new_wrong)
            true_positives = np.count_nonzero(nolonger_wrong)
            false_negatives = np.count_nonzero(still_wrong)
            true_negatives = np.count_nonzero(still_right)
            
            print("false_positives "+str(false_positives))
            print("true_positives "+str(true_positives))
            print("false_negatives "+str(false_negatives))
            print("true_negatives "+str(true_negatives))
            
            print("sum stats: "+str(false_positives+true_positives+false_negatives+true_negatives)+" size genotype array: "+str(len(genomematrix)))
            
            true_positives_nine = np.count_nonzero(nolonger_wrong_nine)
            
            if (true_positives + false_positives) > 0:
                precision = true_positives / (true_positives + false_positives)
            else:
                precision = 0
            
            if (true_positives + false_negatives) > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0
            
            if (precision+recall) > 0:
                fscore = (2*precision*recall)/(precision+recall)
            else:
                fscore = 0
            
            if (true_positives_nine + false_positives) > 0:
                precision_nine = true_positives_nine / (true_positives_nine + false_positives)
            else:
                precision_nine = 0
            if (true_positives_nine + false_negatives) > 0:
                recall_nine = true_positives_nine / (true_positives_nine + false_negatives)
            else:
                recall_nine = 0
            if (precision_nine+recall_nine) > 0:
                fscore_nine = (2*precision_nine*recall_nine)/(precision_nine+recall_nine)
            else:
                fscore_nine = 0
            
            statout.write('\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%1.4f\t%1.4f\t%1.4f\t%s\t%1.4f\t%1.4f\t%1.4f\n' % 
                          (positives,
                           positive_nine, 
                           negatives,
                           old_error_n,
                           new_error_n,
                           false_positives,
                           true_positives,
                           false_negatives,
                           true_negatives,
                           precision, 
                           recall, 
                           fscore, 
                           true_positives_nine,
                           precision_nine,
                           recall_nine,
                           fscore_nine, 
                           (toc - tic)/60 ))
            statout.flush()
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