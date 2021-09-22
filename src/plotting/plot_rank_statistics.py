#!/usr/local/bin/python2.7
# encoding: utf-8
'''
plotting.plot -- shortdesc

plotting.plot is a description

It defines classes_and_methods

@author:     user_name

@copyright:  2021 organization_name. All rights reserved.

@license:    license

@contact:    user_email
@deffield    updated: Updated
'''

import sys
import os

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from collections import Counter

__all__ = []
__version__ = 0.1
__date__ = '2021-09-15'
__updated__ = '2021-09-15'

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
        parser.add_argument("-s", "--rankstatsfile", required=True, dest="rank_stats_file", help="location of rank stats file produced by simulate")
        parser.add_argument("-o", "--outputdir", required=True, dest="output_dir", help="directory location of output files")
        
        # Process arguments
        args = parser.parse_args()

        rank_stats_file = args.rank_stats_file
        output_dir = args.output_dir
        
        print("reading %s " % rank_stats_file)
        
        inputdata =  pd.read_table(rank_stats_file, delimiter="\t", header=None)
        inputdata.columns = ["obs_state","real_state","maxp_state","rank_real_emp","rank_obs_emp","rank_real_mendel","rank_obs_mendel", "rank_real_combproduct","rank_obs_combproduct", "rank_real_combmean","rank_obs_combmean"]
        print(inputdata)
        
        print("plot: empirical vs mendel bayes model")
        heatmap, xedges, yedges = np.histogram2d(inputdata["rank_real_emp"], inputdata["rank_real_mendel"], bins=(9,9))
        
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.xlabel("rank of real genotype in empirical model")
        plt.ylabel("rank of real genotype in mendel bayes model")
        total = np.sum(heatmap.T)
        for (j,i),label in np.ndenumerate(heatmap.T):
            if ((label/total)*100) > 0:
                plt.text(1+(i*0.89),1+(j*0.88),"%.1f" % ((label/total)*100),ha='left',va='bottom', color="grey")
        plt.colorbar()
        
        plt.savefig("%s/hist_2d_empvsmendel.png" % output_dir, bbox_inches='tight')
        
        print("plot: mean combined vs mendel bayes model")
        heatmap, xedges, yedges = np.histogram2d(inputdata["rank_real_combproduct"], inputdata["rank_real_mendel"], bins=(9,9))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.xlabel("rank of real genotype in product combined model")
        plt.ylabel("rank of real genotype in mendel bayes model")
        total = np.sum(heatmap.T)
        for (j,i),label in np.ndenumerate(heatmap.T):
            if ((label/total)*100) > 0:
                plt.text(1+(i*0.89),1+(j*0.88),"%.1f" % ((label/total)*100),ha='left',va='bottom', color="grey")
        plt.colorbar()
        
        plt.savefig("%s/hist_2d_combproductvsmendel.png" % output_dir, bbox_inches='tight')
        
        print("plot: product combined vs mendel bayes model")
        heatmap, xedges, yedges = np.histogram2d(inputdata["rank_real_combmean"], inputdata["rank_real_mendel"], bins=(9,9))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.xlabel("rank of real genotype in mean combined model")
        plt.ylabel("rank of real genotype in mendel bayes model")
        total = np.sum(heatmap.T)
        for (j,i),label in np.ndenumerate(heatmap.T):
            if ((label/total)*100) > 0:
                plt.text(1+(i*0.89),1+(j*0.88),"%.1f" % ((label/total)*100),ha='left',va='bottom', color="grey")
        plt.colorbar()
        
        plt.savefig("%s/hist_2d_combmeanvsmendel.png" % output_dir, bbox_inches='tight')
        
        print("plot: stacked bar chart of ranks")
        
        rank_real_emp = Counter(inputdata["rank_real_emp"])
        rank_real_mendel = Counter(inputdata["rank_real_mendel"])
        rank_real_combproduct = Counter(inputdata["rank_real_combproduct"])
        rank_real_combmean = Counter(inputdata["rank_real_combmean"])
        
        counter = [rank_real_emp, rank_real_mendel, rank_real_combproduct, rank_real_combmean]
        
        N = len(counter)
  
        series = {}
        for key in {key for keys in counter for key in keys}:
            series[key] = [(0 if key not in item else item[key]) for item in counter]
         
        fig, ax = plt.subplots()
        bottom, x = np.zeros(N), ["empirical","mendal","product","mean"]
         
        for key in sorted(series.keys())[::-1]:
            ax.bar(x, series[key], label=key, bottom=bottom)
            bottom += np.array(series[key])
        handles, labels = ax.get_legend_handles_labels()
        
        ax.legend(handles[::-1], labels[::-1], title='Rank of real value', loc='upper left', framealpha=0.2)
        
        plt.savefig("%s/stackedbarchartrank.png" % output_dir, dpi=200)
        
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
        profile_filename = 'plotting.plot_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())