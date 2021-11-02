#!/usr/local/bin/python2.7
# encoding: utf-8
'''
crosssim.hotcross -- shortdesc

crosssim.hotcross is a description

It defines classes_and_methods

@author:     user_name

@copyright:  2021 organization_name. All rights reserved.

@license:    license

@contact:    user_email
@deffield    updated: Updated
'''

import sys
import os
import psutil
from utils import *

import gzip
import importphase
import cloudpickle, pickle

import pandas

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from typing import List, Dict, Tuple

import quicksim
import multiprocessing as mp

from scipy.stats import chisquare
import scipy.sparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

__all__ = []
__version__ = 0.1
__date__ = '2021-04-23'
__updated__ = '2021-04-23'

DEBUG = 1
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
        parser.add_argument("-v", "--verbose", dest="verbose", action="count", help="set verbosity level [default: %(default)s]")
        parser.add_argument("-s", "--snp_map", dest="snp_map", help="tab (\"\t\") separated file with four columns \"chr\" \"id\" \"cm\" \"bp\" ")
        parser.add_argument("-o", "--snp_list_include", dest="snp_list", help="ordered file with each line an snp id that corresponds the genotype data in input")
        parser.add_argument("-i", "--haplotype_input_file", dest="haplotype_file", help="haplotype_input file in specified format")
        parser.add_argument("-f", "--haplotype_input_format", dest="haplotype_format", help="haplotype input file format", choices=["AlphaImpute2"])
        parser.add_argument('-p', '--pedigree', dest="pedigree", help="pedigree file")
        parser.add_argument("-I", "--indirectory", dest="in_dir", help="directory with simulation input")
        parser.add_argument("-O", "--outdirectory", dest="out_dir", help="directory to place output")
        parser.add_argument("-t", "--threads", dest="threads", help="threads to run parallel simulations in", default=mp.cpu_count())
        parser.add_argument('-V', '--version', action='version', version=program_version_message)

        # Process arguments
        args = parser.parse_args()
        
        haplotype_file = args.haplotype_file
        haplotype_format = args.haplotype_format
        out_dir  = args.out_dir
        in_dir  = args.in_dir
        verbose = args.verbose
        threads = args.threads
        snpsFile = args.snp_map
        snp_list = args.snp_list
        pedigree_file = args.pedigree
        
        process = psutil.Process(os.getpid())
        print("memory usage %0.2f MB (%0.2f pc)" % (process.memory_info().rss / 1024 ** 2, process.memory_percent()))  # in bytes 

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        
        if pedigree_file is not None:
            pedigree = pandas.read_csv(pedigree_file, delimiter=' ', header=0, index_col="kid", names=["kid","sire","dam","sex"])
            with gzip.open('%s/pedigree.pickle.gz' % out_dir, "wb") as sim_pickle:
                cloudpickle.dump(pedigree, sim_pickle, protocol=pickle.HIGHEST_PROTOCOL)
            genders: Dict[str,int] = {}
            for index, row in pedigree.iterrows():
                genders[str(index)] = int(row["sex"])
            with gzip.open('%s/genders.pickle.gz' % out_dir, "wb") as sim_pickle:
                cloudpickle.dump(genders, sim_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with gzip.open('%s/pedigree.pickle.gz' % in_dir, "rb") as pickleped:
                pedigree = pickle.load(pickleped, protocol=pickle.HIGHEST_PROTOCOL)
            with gzip.open('%s/genders.pickle.gz' % in_dir, "rb") as picklegen:
                genders = pickle.load.dump(picklegen, sim_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with openfile(snpsFile) as file_handle:
            snpdetails = {bla['id']:bla for bla in [dict(zip(["chr", "id", "cm","bp"],x.strip().split("\t"))) for x in file_handle]}
            print("%s snp location details" % len(snpdetails))
            print("%s chr2 snp location details" % len([x for x in snpdetails.values() if x['chr'] == "2"]))
        
        with openfile(snp_list, mode='rt') as file_handle:
            snpIds: List[str] = [x.strip() for x in file_handle]
            print("%s ordered snpids" % len(snpIds))
            print("%s chr2 ordered snpids" % len([x for x in snpIds if snpdetails[x]['chr'] == "2"]))
        
        chr2snp2pos, snp2chromosome, gm = indexSnps(snpdetails, snpIds)
        
                ## genome info
        if quicksim.validate_genome(gm):
            genome = quicksim.Genome(gm)
        else :
            raise Exception("invalid genome ABORT ABORT")
        
        if haplotype_file is not None:
            if haplotype_format.upper() == "AlphaImpute2".upper():
                haplotypes = importphase.importAlphaImpute2(haplotype_file, snpids=snpIds, snpid2chr=snp2chromosome)
            with gzip.open('%s/haplotypes.pickle.gz' % out_dir, "wb") as sim_pickle:
                cloudpickle.dump(haplotype_format, sim_pickle, protocol=pickle.HIGHEST_PROTOCOL)
                    #real data into Andreas format 
            gens = {}
            for tag,chrdict in haplotypes.items():
                gens[tag] = {c:{0:chrdict[c].phap,1:chrdict[c].mhap} for c in genome.chroms.keys()}
    
            with gzip.open('%s/gens.pickle.gz' % out_dir, "wb") as sim_pickle:
                cloudpickle.dump(gens, sim_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        else :
            with gzip.open('%s/haplotypes.pickle.gz' % in_dir, "rb") as picklehap:
                haplotypes = pickle.load(picklehap, protocol=pickle.HIGHEST_PROTOCOL)
            with gzip.open('%s/gens.pickle.gz' % in_dir, "rb") as picklegens:
                gens = pickle.load(picklegens, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("%s individuals with haplotypes" % len(haplotypes))
        
        founders = quicksim.create_founders(genders,gens,genome)
        
        validChildren = [(x,y) for x,y in pedigree.iterrows() if str(x).startswith("4004")]
        
        realCrossPoints = np.empty(len(validChildren), dtype=object)
        chromosomes = list(founders[list(founders.keys())[0]].genotype.keys())
        genotypelengths = {chromosome:len(founders[list(founders.keys())[0]].genotype[chromosome][0]) for chromosome in chromosomes}
        nchildren = len(realCrossPoints)
       
        print("create crossover matrix")
        chr2matcross = {} 
        chr2patcross = {} 
        for chromosome in chromosomes :
            chr2matcross[chromosome] = np.empty((genotypelengths[chromosome],nchildren), dtype=np.float32)
            chr2patcross[chromosome] = np.empty((genotypelengths[chromosome],nchildren), dtype=np.float32)
        for i, (index, row) in enumerate(validChildren):
            kid, sire, dam, sex = (str(index), str(row["sire"]),str(row["dam"]),int(row["sex"]))
            simprogeny_cross: Dict[int, Dict[int, Tuple]] = {}
            if str(index).startswith("4004") :
                kid = founders[kid]
                sireobj = founders[sire]
                damobj = founders[dam]
                for chrom in kid.genotype.keys():
                    patcr = predictcrosspoints(
                        kid.genotype[chrom][0], #paternal strand
                        sireobj.genotype[chrom][0], #paternal paternal
                        sireobj.genotype[chrom][1]) #paternal maternal
                    runvalue, runstart, runlength = find_runs(patcr)
                    patcregion = (runstart[runvalue == 3], runlength[runvalue == 3])
                    crossprobs_pat = chr2patcross[chromosome]
                    for xstart, xend in zip(runstart, (runstart+runlength)):
                        crossprobs_pat[xstart:xend,i] = 1./float(xend-xstart)
                    
                    matcr = predictcrosspoints(
                        kid.genotype[chrom][1],#maternal strand
                        damobj.genotype[chrom][0], #maternal paternal
                        damobj.genotype[chrom][1]) #maternal maternal
                    runvalue, runstart, runlength = find_runs(matcr)
                    matcregion = (runstart[runvalue == 3], runlength[runvalue == 3])
                    simprogeny_cross[chrom] = {0:patcregion, 1:matcregion}
                    realCrossPoints[i] = simprogeny_cross
                    crossprobs_mat = chr2matcross[chromosome]
                    for xstart, xend in zip(runstart, (runstart+runlength)):
                        crossprobs_mat[xstart:xend,i] = 1./float(xend-xstart)
                        
        if not os.path.exists("%s/stats" % (out_dir)):
            os.mkdir("%s/stats" % (out_dir))        
        if not os.path.exists("%s/stats/chr%s" % (out_dir,chromosome)):
            os.mkdir("%s/stats/chr%s" % (out_dir,chromosome))
        
        for chromosome in chromosomes :
            crossprobs_mat = chr2matcross[chromosome]
            crossprobs_pat = chr2patcross[chromosome]
            
            with PdfPages('%s/stats/chr%s/plot.pdf' %  (out_dir,chromosome) ) as pdf:
                fig =  plt.figure(figsize=(12,4), dpi= 200)
                plt.hist([x/nchildren for x in (crossprobs_pat > 0).sum(axis=1).tolist()], bins='auto')
                plt.title('Distribution of N crossover (> 0 pÌ)‚\nin paternal haplotype')
                plt.xlabel('crossover pÌ‚ \n(sum p at position/n simulations)')
                plt.ylabel('count N')
                pdf.savefig()
                plt.close(fig)
                
                fig =  plt.figure(figsize=(12,4), dpi= 200)
                plt.hist([x/nchildren for x in (crossprobs_mat > 0).sum(axis=1).tolist()], bins='auto')
                plt.title('Distribution of N crossover (> 0 pÌ)‚\nin maternal haplotype')
                plt.xlabel('crossover pÌ‚ \n(sum p at position/n simulations)')
                plt.ylabel('count N')
                pdf.savefig()
                plt.close(fig)
                
                fig =  plt.figure(figsize=(12,4), dpi= 200)
                plt.hist([x/nchildren for x in crossprobs_pat.sum(axis=1).tolist()], bins='auto')
                plt.title('Distribution of crossover sum pÌ‚\nin paternal haplotype')
                plt.xlabel('crossover pÌ‚ \n(sum p at position/n simulations)')
                plt.ylabel('count N')
                pdf.savefig()
                plt.close(fig)
                
                fig =  plt.figure(figsize=(12,4), dpi= 200)
                plt.hist([x/nchildren for x in crossprobs_mat.sum(axis=1).tolist()], bins='auto')
                plt.title('Distribution of crossover sum pÌ‚\nin maternal haplotype')
                plt.xlabel('crossover pÌ‚ \n(sum p at position/n simulations)')
                plt.ylabel('count N')
                pdf.savefig()
                plt.close(fig)
        
       # chisquare(f_obs, f_exp)
        
        

        return 0
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception as e:
        if DEBUG or TESTRUN:
            raise(e)
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use --help")
        return 2

if __name__ == "__main__":
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = 'crosssim.hotcross_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())