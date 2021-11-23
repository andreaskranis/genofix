#!/usr/local/bin/python2.7
# encoding: utf-8
'''
plotting.plot_precision_recall -- shortdesc

plotting.plot_precision_recall is a description

It defines classes_and_methods

@author:     user_name

@copyright:  2021 organization_name. All rights reserved.

@license:    license

@contact:    user_email
@deffield    updated: Updated
'''

import sys
import os
import csv
import pandas as pd
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

__all__ = []
__version__ = 0.1
__date__ = '2021-11-22'
__updated__ = '2021-11-22'

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
    parser.add_argument("-s", "--statistics_input_files", required=True, nargs='+', dest="statistics_input_files", help="statistics input files in tsv with header")
    
    # Process arguments
    args = parser.parse_args()

    statistics_input_files = args.statistics_input_files

    stats = list()
    for file in statistics_input_files:
        with open(file, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            for row in reader:
                stats.append(row)

    df = pd.DataFrame(stats)
    print(df.columns)
    
    df['old_error'] = np.array(list(map(int,df.loc[:,'old_error_n'])))
    df['new_error'] = np.array(list(map(int,df.loc[:,'new_error_n'])))
    df['total_observations'] = np.array(list(map(int,df.loc[:,'total_observations'])))
    df['precision_9pc'] = np.array(list(map(float,df.loc[:,'precision_9'])))*100
    df['recall_9pc'] = np.array(list(map(float,df.loc[:,'recall_9'])))*100
    df['fscore_9pc'] = np.array(list(map(float,df.loc[:,'fscore_9'])))*100
    
    df['old_error_pc'] = (df.loc[:,'old_error']/df.loc[:,'total_observations'])*100
    df['new_error_pc'] = (df.loc[:,'new_error']/df.loc[:,'total_observations'])*100
    
    print(df.loc[:,'old_error_pc'])
    print(df.loc[:,'new_error_pc'] )

    print(df.transpose())
    sns.set_theme()
    sns.set(style="ticks")
    
    line1 = sns.lineplot(data=df, x='old_error_pc', y='precision_9pc',dashes=False, marker='+', ms=4, markeredgecolor='black', legend='brief', label='precision')
    line2 = sns.lineplot(data=df, x='old_error_pc', y='recall_9pc',dashes=False, marker='x', ms=4, markeredgecolor='black', legend='brief', label='recall')
    line3 = sns.lineplot(data=df, x='old_error_pc', y='fscore_9pc',dashes=False, marker='.', ms=6, markeredgecolor='black', legend='brief', label='fscore')
    line1.set_xlabel('simulated error (%)')
    line1.set_ylabel('%')
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    if TESTRUN:
        import doctest
        doctest.testmod()
    if PROFILE:
        import cProfile
        import pstats
        profile_filename = 'plotting.plot_precision_recall_profile.txt'
        cProfile.run('main()', profile_filename)
        statsfile = open("profile_stats.txt", "wb")
        p = pstats.Stats(profile_filename, stream=statsfile)
        stats = p.strip_dirs().sort_stats('cumulative')
        stats.print_stats()
        statsfile.close()
        sys.exit(0)
    sys.exit(main())