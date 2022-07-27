#!/usr/local/bin/python2.7
# encoding: utf-8
'''
crosssim.crosssim -- shortdesc

crosssim.crosssim is a description

It defines classes_and_methods

@author:     user_name

@copyright:  2021 organization_name. All rights reserved.

@license:    license

@contact:    user_email
@deffield    updated: Updated
'''
import sys, gc
import os, traceback
import pathlib
from visualize import Visualize
import importphase
from gfutils import *
import crossoverdetection

import mgzip
import pandas
import numpy as np
import multiprocessing as mp
import concurrent.futures
import threading, multiprocessing
import quicksim
import psutil
import cloudpickle, pickle
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
import copy
from typing import List, Dict, Tuple
import cProfile
import scipy.sparse
from scipy.sparse import lil_matrix, dok_matrix

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import atexit

import tempfile
from datetime import datetime
from pathlib import Path
import itertools 
import math

__all__ = []
__version__ = 0.1
__date__ = '2021-04-13'
__updated__ = '2021-04-13'

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

    # Setup argument parser
    parser = ArgumentParser(description=program_license, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--snp_map", dest="snp_map", required=True, help="tab (\"\t\") separated file with four columns \"chr\" \"id\" \"cm\" \"bp\" ")
    parser.add_argument("-o", "--snp_list_include", dest="snp_list", required=True, help="ordered file with each line an snp id that corresponds the genotype data in input")
    parser.add_argument('-p', '--pedigree', dest="pedigree", help="pedigree file")
    parser.add_argument("-O", "--outdirectory", dest="out_dir", help="directory to place output", default="%s/%s_crosssim_out" % (tempfile.gettempdir(), datetime.now()))
    parser.add_argument("-n", "--nsim", dest="nsim", help="number of simulations per trio", default=10000, type=int)
    parser.add_argument("-t", "--threads", dest="threads", help="threads to run parallel simulations in", type=int, default=psutil.cpu_count(logical = False))
    parser.add_argument("-g", "--printgenotype", dest="printgenotype", help="threads to run parallel simulations in", default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument("-i", "--haplotype_input_file", dest="haplotype_file", help="haplotype_input file in specified format")
    parser.add_argument("-f", "--haplotype_input_format", dest="haplotype_format", help="haplotype input file format", choices=["AlphaImpute2"])
    parser.add_argument("-I", "--input_directory", dest="in_dir", help="directory with pregenerated inputfiles")
    parser.add_argument("-v", "--plot_visualisation", dest="viz", help="plot_visualisation", action='store_true', default=False)
    parser.add_argument('-V', '--version', action='version', version=program_version_message)

    # Process arguments
    args = parser.parse_args()

    in_dir = args.in_dir
    if in_dir != None:
        print("Data input directory %s has been provided other parameters will be ignored when pre-generated data is found" % args.in_dir)
    
    out_dir  = args.out_dir
    print("Crossim data output directory will be %s" % out_dir)
    
    viz = args.viz
    threads = args.threads
    snpsFile = args.snp_map
    snp_list = args.snp_list
    pedigree_file = args.pedigree
    nsim = int(args.nsim)
    printgenotype = args.printgenotype
    haplotype_file = args.haplotype_file
    haplotype_format = args.haplotype_format
    out_dir  = args.out_dir
    
    print(nsim)
    
    process = psutil.Process(os.getpid())
    print("memory usage %0.2f MB (%0.2f pc)" % (process.memory_info().rss / 1024 ** 2, process.memory_percent()))  # in bytes 

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    ped_p_f = '%s/pedigree.pickle.gz' % in_dir
    print("%s %s %s" % (in_dir, os.path.exists(ped_p_f), os.path.isfile(ped_p_f)))
    if in_dir == None or not (os.path.exists(ped_p_f) and os.path.isfile(ped_p_f)):
        pedigree = pandas.read_csv(pedigree_file, delimiter=' ', header=0, index_col="kid", names=["kid","sire","dam","sex"])
        with mgzip.open('%s/pedigree.pickle.gz' % out_dir, "wb", thread=threads) as sim_pickle:
            cloudpickle.dump(pedigree, sim_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    else :
        with mgzip.open(ped_p_f, "rb", thread=threads) as sim_pickle:
            pedigree = pickle.load(sim_pickle)
        print("preloaded %s from pickle" % ped_p_f)
    
    genders_p_f = '%s/genders.pickle.gz' % in_dir
    if in_dir == None or not (os.path.exists(genders_p_f) and os.path.isfile(genders_p_f)):
        genders: Dict[str,int] = {}
        for index, row in pedigree.iterrows():
            genders[str(index)] = int(row["sex"])
        with mgzip.open('%s/genders.pickle.gz' % out_dir, "wb", thread=threads) as gen_pickle:
            cloudpickle.dump(genders, gen_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    else :
        with mgzip.open(genders_p_f, "rb", thread=threads) as gen_pickle:
            genders = pickle.load(gen_pickle)
        print("preloaded %s from pickle" % genders_p_f)
    
    with openfile(snpsFile) as file_handle:
        snpdetails: Dict[str, Dict[str, str]] = {bla['id']:bla for bla in [dict(zip(["chr", "id", "cm","bp"],x.strip().split("\t"))) for x in file_handle]}
        print("%s snp location details" % len(snpdetails))
        print("%s chr2 snp location details" % len([x for x in snpdetails.values() if x['chr'] == "2"]))
        
    with openfile(snp_list, mode='rt') as file_handle:
        snpIds: List[str] = [x.strip() for x in file_handle]
        print("%s ordered snpids" % len(snpIds))
        print("%s chr2 ordered snpids" % len([x for x in snpIds if snpdetails[x]['chr'] == "2"]))
    
    chr2snp2pos, snp2chromosome, gm = indexSnps(snpdetails, snpIds)
    
    genome_p_f = '%s/genome.pickle.gz' % in_dir
    if in_dir == None or not (os.path.exists(genome_p_f) and os.path.isfile(genome_p_f)):
        ## genome info
        if quicksim.validate_genome(gm):
            genome = quicksim.Genome(gm)
            with mgzip.open('%s/genome.pickle.gz' % out_dir, "wb", thread=threads) as genome_pickle:
                cloudpickle.dump(genome, genome_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        else :
            raise Exception("invalid genome ABORT ABORT")
    else :
        with mgzip.open(genome_p_f, "rb", thread=threads) as gen_pickle:
            genome = pickle.load(gen_pickle)
            print("preloaded %s from pickle" % genome_p_f)
    
    haplotypes_p_f = '%s/haplotypes.pickle.gz' % in_dir
    if in_dir == None or not (os.path.exists(haplotypes_p_f) and os.path.isfile(haplotypes_p_f)):
        if haplotype_format.upper() == "AlphaImpute2".upper():
            haplotypes = importphase.importAlphaImpute2(haplotype_file, snpids=snpIds, snpid2chr=snp2chromosome)
        with mgzip.open('%s/haplotypes.pickle.gz' % out_dir, "wb", thread=threads) as sim_pickle:
            cloudpickle.dump(haplotypes, sim_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    else :
        with mgzip.open(haplotypes_p_f, "rb", thread=threads) as gen_pickle:
            haplotypes = pickle.load(gen_pickle)
            print("preloaded %s from pickle" % haplotypes_p_f)
    
    print("dataset has %s individuals with haplotypes" % len(haplotypes))
    
    gens_p_f = '%s/gens.pickle.gz' % in_dir
    if in_dir == None or not (os.path.exists(gens_p_f) and os.path.isfile(gens_p_f)):
        #real data into Andreas format 
        gens = {}
        for tag,chrdict in haplotypes.items():
            gens[tag] = {c:{0:chrdict[c].phap,1:chrdict[c].mhap} for c in genome.chroms.keys()}
        with mgzip.open('%s/gens.pickle.gz' % out_dir, "wb", thread= threads) as sim_pickle:
            cloudpickle.dump(gens, sim_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    else : 
        with mgzip.open(gens_p_f, "rb", thread=threads) as gen_pickle:
            gens = pickle.load(gen_pickle)
        print("preloaded %s from pickle" % gens_p_f)
    
    print(genome.chroms.keys())
    print(genome.chroms[2])   
    
    founders_p_f = '%s/founders.pickle.gz' % in_dir
    if in_dir == None or not (os.path.exists(founders_p_f) and os.path.isfile(founders_p_f)):
        founders = quicksim.create_founders(genders,gens,genome)
        with mgzip.open('%s/founders.pickle.gz' % out_dir, "wb", thread=threads) as founders_pickle:
            cloudpickle.dump(founders, founders_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    else : 
        with mgzip.open(founders_p_f, "rb", thread=threads) as founders_pickle:
            founders = pickle.load(founders_pickle)
        print("preloaded %s from pickle" % founders_p_f)
    
    print(founders.keys())
    os.makedirs("%s/crossovers" % (out_dir),exist_ok=True)
    
    chromosomes = sorted(list(founders[list(founders.keys())[0]].genotype.keys())) #don't like this way of doing it
    genotypelengths = {chromosome:len(founders[list(founders.keys())[0]].genotype[chromosome][0]) for chromosome in chromosomes}
    print("chromosomes: %s" % chromosomes)
    
    genpattern = "40"
    
    simulatechildren = {}
    extantTrios = {}
    for index, row in pedigree.iterrows():
            kidId = str(index)
            sireId = str(row["sire"])
            damId = str(row["dam"])
            if kidId in founders and sireId in founders and damId in founders :
                kidobj = founders[kidId]
                sireobj = founders[sireId]
                damobj = founders[damId]
                if kidId.startswith(genpattern) :
                        simulatechildren[kidId] = (str(row["sire"]),str(row["dam"]),int(row["sex"]))
                extantTrios[kidId] = (str(row["sire"]),str(row["dam"]),int(row["sex"]))
    
    xover_file = '%s/crossovers/crossovers.pickle.gz' % out_dir
    if in_dir == None or not (os.path.exists(xover_file) and os.path.isfile(xover_file)):
        countxovers = {}
        xovers = {}
        for kidId,(sireId, damId, sex) in extantTrios.items():
            trioID = "%s~%s~%s" % (kidId,sireId,damId)
            print(trioID)
            kidobj = founders[kidId]
            sireobj = founders[sireId]
            damobj = founders[damId]
            xovers[kidId] = crossoverdetection.predictCrossoverRegions(kidobj, sireobj, damobj)
            countxovers[kidId] = [len(xovers[kidId][chr][0][0]+xovers[kidId][chr][0][1]) if chr in xovers[kidId] else 0 for chr in chromosomes]
            crossoverdetection.writeCrossoverRegionsShortFormat(xovers[kidId], kidId, '%s/crossovers/%s.txt.gz' % (out_dir,trioID))
            crossoverdetection.writeCrossoverRegionsBedFormat(xovers[kidId], kidId, '%s/crossovers/%s.bed.gz' % (out_dir,trioID))
        with mgzip.open(xover_file, "wb", thread=threads) as xovers_pickle:
            print("write xovers to %s" % xover_file)
            cloudpickle.dump(xovers, xovers_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("loading %s" % xover_file)
        with mgzip.open(xover_file, "rb", thread=threads) as xovers_pickle:
            xovers = pickle.load(xovers_pickle)
        countxovers = {}
        for kidId in xovers.keys():
            countxovers[kidId] = [len(xovers[kidId][chr][0][0]+xovers[kidId][chr][0][1]) if chr in xovers[kidId] else 0 for chr in chromosomes]
    
    with openfile('%s/crossovers/xovercounts.txt.gz' % (out_dir), "wt") as fout:
        fout.write("\t".join(map(str,["#ID"]+chromosomes))+"\n")
        for idk,counts in sorted(countxovers.items()):
            fout.write("\t".join(map(str,[idk]+counts))+"\n")
    
    vis = Visualize(snpdetails, snpIds, genotypelengths)
    vis.realxoverStatistics(xovers,countxovers, chromosomes, '%s/crossovers/xovercounts.pdf' % (out_dir))
    del countxovers
    
    print("children to simulate: %s" % (len(simulatechildren)))
    
    kids = np.array(list(simulatechildren.keys()), dtype=str)
    
    sim_dir:str = in_dir # default load from input
    kids_pickles = ['%s/sim_crossovers/pickle/chr%s/sim_actual_%s.pickle.gz' % (in_dir,chro,kid) for kid,chro in itertools.product(kids,chromosomes)]
    
    pathnotexists = [(not os.path.exists(x)) for x in kids_pickles]
    pathisnotfile = [(not os.path.isfile(x)) for x in kids_pickles]
    
    if in_dir == None or all(pathnotexists) or all(pathisnotfile):
        if in_dir != None and (all(pathnotexists) or all(pathisnotfile)):
            print("some kids/chromsomes are missing simulation cache so sim will be rerun")
        print("running simulations...")
        sim_dir:str = out_dir # we are generating so placed in ouptut dir
        
        def init_worker():
            os.nice(1)
            thisTL = multiprocessing.current_process()
            setattr(thisTL, 'genome', copy.deepcopy(genome))
            setattr(thisTL, 'founders', copy.deepcopy(founders))
            if printgenotype :
                genotypeout = mgzip.open('%s/sim_crossovers/sim_genotypes/%s.txt.gz' % (out_dir,thisTL.name),
                                     mode="wt", thread=threads)
                setattr(thisTL, 'genotypeout', genotypeout)
                thisTL.genotypeout.write("#")
                example = list(thisTL.founders.values())[-1].genotype
                for chromosome, genotype in list(example.items()):
                    thisTL.genotypeout.write("#%s\n" % "\t".join([str(chromosome)]*len(genotype)))
                def savegenout():
                    thisTL.genotypeout.close()
                    print("nice close %s" % thisTL.genotypeout.name)
                atexit.register(savegenout)
            else :
                setattr(thisTL, 'genotypeout', None)
                #print("init %s" % thisTL.name)
        
        
        os.makedirs("%s/sim_crossovers/sim_genotypes" % (out_dir),exist_ok=True)
        os.makedirs("%s/sim_crossovers/npz" % (out_dir), exist_ok=True)
        os.makedirs("%s/sim_crossovers/pickle" % (out_dir),exist_ok=True)
        os.makedirs("%s/sim_crossovers/plot_individuals" % (out_dir),exist_ok=True)
        
        with ProcessPoolExecutor(max_workers=threads, initializer=init_worker) as executor:
            simulatedData = {executor.submit(crossoverdetection.simulatePopulation, nsim, kid, sire, dam, sex) : kid
                                              for ni, (kid, (sire,dam,sex)) in enumerate(simulatechildren.items())}
            print("waiting on %s queued jobs with %s threads" % (len(simulatedData), threads))
            for ni, future in enumerate(concurrent.futures.as_completed(simulatedData)) :
                kid = simulatedData[future]
                e = future.exception()
                if e is not None:
                    print(repr(e))
                actualcrosses, ambiguityregions = future.result()
                for chromosome in actualcrosses.keys() :
                    os.makedirs("%s/sim_crossovers/pickle/chr%s" % (out_dir,chromosome),exist_ok=True)
                    with mgzip.open('%s/sim_crossovers/pickle/chr%s/sim_actual_%s.pickle.gz' % (out_dir,chromosome,kid), "wb", thread=threads) as act_pickle:
                        cloudpickle.dump(actualcrosses[chromosome], act_pickle, protocol=pickle.HIGHEST_PROTOCOL)
                    #print(actualcrosses)
                    #print("actual pat %s %s %s " % (actualcrosses[chromosome][0][0],chromosome,kid))
                    #print("actual mat %s %s %s " % (actualcrosses[chromosome][0][1],chromosome,kid))
                    with mgzip.open('%s/sim_crossovers/pickle/chr%s/sim_ambigregions_%s.pickle.gz' % (out_dir,chromosome,kid), "wb", thread=threads) as amb_pickle:
                        cloudpickle.dump(ambiguityregions[chromosome], amb_pickle, protocol=pickle.HIGHEST_PROTOCOL)
                    #print("ambigu pat %s %s %s " % (ambiguityregions[chromosome][0][0],chromosome,kid))
                    #print("ambigu mat %s %s %s " % (ambiguityregions[chromosome][0][1],chromosome,kid))
                    #for i in range(1,5):
                    #    print("%s %s %s %s %s" % (i, kid, chromosome, actualcrosses[chromosome][i], ambiguityregions[chromosome][i]))
                del simulatedData[future]
                del future
                del actualcrosses
                del ambiguityregions
                print("%s of %s: memory usage %0.2f MB (%0.2f pc)" % (ni+1, len(simulatechildren), process.memory_info().rss / 1024 ** 2, process.memory_percent()))
    
    print("memory usage post gc %0.2f MB (%0.2f pc)" % (process.memory_info().rss / 1024 ** 2, process.memory_percent()))  # in bytes 
    
    if viz:
        print("Visualize simulations on individuals from %s" % sim_dir)
        print(chromosomes)
        print(kids)
        vis.individualsStatistics(chromosomes[0:3],kids[0:2], sim_dir, out_dir, threads=threads)
        print("Visualize simulations across population from %s" % sim_dir)
        vis.populationStatistics(chromosomes,kids, sim_dir, out_dir)
    
    print("memory usage %0.2f MB (%0.2f pc)" % (process.memory_info().rss / 1024 ** 2, process.memory_percent()))  # in bytes 
    gc.collect()
    print("memory usage post gc %0.2f MB (%0.2f pc)" % (process.memory_info().rss / 1024 ** 2, process.memory_percent()))  # in bytes 

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
        profile_filename = 'crosssim.crosssim_profile2.profile'
        cProfile.run('main()', "/mnt/md0/mhindle/gensys/chr3sim2/profile.out")
        sys.exit(0)
    sys.exit(main())