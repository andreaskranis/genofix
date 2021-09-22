#!/usr/local/bin/python2.7
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
sys.path.append(os.getcwd())
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
from correct_genotypes import CorrectGenotypes
from typing import Tuple, List, Dict, Union

from tqdm import tqdm

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
        parser.add_argument("-d", "--nolddist", dest="nolddist",  action='store_true', help="do not use ld distance in calculation")
        parser.add_argument("-b", "--lookback", dest="lookback", type=int, default=2,  help="generations to look up/back")
        parser.add_argument("-e", "--errorrate", dest="error_rate", type=float, default=1,  help="simulated error rate")
        parser.add_argument('-V', '--version', action='version', version=program_version_message)
        
        # Process arguments
        args = parser.parse_args()

        out_dir = args.outdir
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        
        pedigree = PedigreeDAG.from_file(args.pedigree)
        
        threshold_singles = float(args.threshold_singles)
        threshold_pairs = float(args.threshold_pairs)
        surround_size = int(args.surround_size)
        
        genomein = pd.read_csv(args.snps, sep=' ', names = ["chrom", "snpid", "cm", "pos"])

        lddist = (not args.nolddist)
        print("Will use ld distance? %s: " % lddist)
        
        error_rate = args.error_rate
        print("Will use error rate %s: "  % error_rate)
        
        lookback = args.lookback
        print("Will %s generations up/back: " % lookback)
        
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
            
        foundersires = pedigree.males.difference(pedigree.kid2sire.keys())
        founderdams = pedigree.females.difference(pedigree.kid2dam.keys())

        genders = {x:1 for x in pedigree.males}
        genders.update({x:2 for x in pedigree.females})

        gens = {x:None for x in list(foundersires) + list(founderdams)}
        
        rs = np.random.Generator(np.random.PCG64(1234)) 
        
        for founder in list(foundersires) + list(founderdams):
            g = gsim.Genotype(chromosomes)
            for chromosome in chromosomes:
                g.data[chromosome][0] = rs.integers(size=genome.chroms[chromosome].nvars, low=0, high=2)
                g.data[chromosome][1] = rs.integers(size=genome.chroms[chromosome].nvars, low=0, high=2)
            gens[founder] = g
        founders = gsim.create_founders(genders,gens,genome)
        
        genotypes = founders.copy()
        while(len(genotypes.keys()) < len(pedigree.males)+len(pedigree.females)):
            for kid, sire, dam, sex in pedigree.as_ped():
                if kid not in genotypes.keys() and sire in genotypes.keys() and dam in genotypes.keys():
                    genotypes[kid] = gsim.mate(genotypes[sire], genotypes[dam], kid, genome, sex)
        
        print("%s genotypes generated for %s individuals" % (len(genotypes.keys()), len(pedigree.males)+len(pedigree.females)))
        
        chromosomes = sorted([int(chro) for chro in genome.chroms.keys()])
        print(chromosomes)
        
        snpids = list(chain(*[genome.chroms[chro].snpids for chro in chromosomes]))
        
        #this is messy np.add(genotypes[kid].genotype[chro].values()) work?
        data = {kid:np.concatenate([np.add(genotypes[kid].genotype[chro][0], genotypes[kid].genotype[chro][1]) for chro in chromosomes]) for kid in genotypes.keys()}
        print("building genotype matrix")
        genomematrix = pd.DataFrame.from_dict(data, orient='index', columns= snpids, dtype=np.uint8).iloc[:,0:50] ##first 50 snps
        del data
        
        
        errors = math.ceil(genomematrix.size*(error_rate/100))
        
        print("will insert %spc errors or %s snps " % (error_rate, errors))

        genomematrix.to_csv("%s/simulatedgenome.ssv" % out_dir, sep=" ")
        
        allelefrq = pd.DataFrame(np.array([genomematrix[snp].value_counts().values for snp in genomematrix.columns]), index=genomematrix.columns, columns=["0","1","2"])
        
        individuals = rs.choice(genomematrix.index, size=errors, replace=True)
        positionsmutate = rs.choice(genomematrix.columns, size=errors, replace=True)
        print("chromosomes found: %s " % chromosomes) 
        
        genotypes_with_errors = genomematrix.copy()
        
        print("inserting errors into simulated genotypes...")
        for kid, SNP_id in tqdm(zip(individuals, positionsmutate)):
            actual = genotypes_with_errors.loc[kid, SNP_id]
            target = actual
            while target == actual:
                target = rs.integers(size=1, low=0, high=3)
            genotypes_with_errors.loc[kid, SNP_id] = target
        
        genotypes_with_errors.to_csv("%s/simulatedgenome_with_%serrors.ssv" % (out_dir, error_rate), sep=" ")
        
        difference = genomematrix - genotypes_with_errors
        print("%s errors %1.2fpc in array" % (np.count_nonzero(difference), (np.count_nonzero(difference)/difference.size)*100))
        
        with open("%s/statistics.tsv" % out_dir, "wt") as statout:
            headers = ("threshold_pair",
                       "threshold_single",
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
                               "fscore_9")
            statout.write('\t'.join(headers)+'\n')
            statout.flush()
            
            print("Threshold singles %s threshold pairs %s" % (threshold_singles, threshold_singles))
            c = CorrectGenotypes(chromosome2snp=chromosome2snp, surround_size=surround_size)
            #corrected_genotype = c.correctMatrix(genomematrix, 
            #                                            pedigree, 
            #                                            thresholdpairs, thresholdsingles,
            #                                            threads=12, DEBUGDIR=error_dir)
            #corrected_genotype.to_csv("%s/simulatedgenome_corrected_virgin_threshold_%s_%s.ssv" % (out_dir,thresholdpairs, thresholdsingles), sep=" ")
            
            #differenceFP = genomematrix - corrected_genotype
            
            statout.write('%s\t%s\t%s' % 
                          (threshold_pairs, threshold_singles,
                           np.product(genomematrix.shape) ))
            statout.flush()
            
            corrected_genotype = c.correctMatrix(genotypes_with_errors, 
                                                        pedigree, 
                                                        threshold_pairs, threshold_singles,
                                                        lddist, back=lookback,
                                                        threads=20, DEBUGDIR=out_dir, debugreal=genomematrix)
            corrected_genotype.to_csv("%s/simulatedgenome_corrected_errors_threshold_%s_%s.ssv" % (out_dir,threshold_pairs, threshold_singles), sep=" ")
            
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
            
            statout.write('\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%1.4f\t%1.4f\t%1.4f\t%s\t%1.4f\t%1.4f\t%1.4f\n' % 
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
                           fscore_nine))
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