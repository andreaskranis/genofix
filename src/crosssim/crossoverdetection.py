import sys
import threading, multiprocessing
from crosssim.utils import predictcrosspoints, find_runs, openfile
import numpy as np
from typing import Dict, Tuple, List, Set

def simulatePopulation(nsim,kid,sire,dam,sex):
    '''
    thread safe implementation deep copies founders and genome to local thread
    '''
    #genome = getattr(thisTL, 'genome', None)
    #founders = getattr(thisTL, 'founders', None)
    #if genome is None:
    #    setattr(thisTL, 'genome', copy.deepcopy(genome_nts))
    #    genome = getattr(thisTL, 'genome', None)
        #print("init genome on thread %s" % threading.current_thread().name)
    
    #if founders is None:
    #    setattr(thisTL, 'founders', copy.deepcopy(founders_nts))
    #   founders = getattr(thisTL, 'founders', None)
    thisTL = multiprocessing.current_process()
    
    founders = thisTL.founders
    genome = thisTL.genome

    oldkid = founders[kid]
    sireobj = founders[sire]
    damobj = founders[dam]
    
    simulatedCrossPoints = {chro:[None]*nsim for chro in oldkid.genotype.keys()}
    actualCrossPoints = {chro:[None]*nsim for chro in oldkid.genotype.keys()}
    genome.randomise(seed=int(np.prod(list(map(ord,kid)))))
    for i in range(0,nsim) :
        newkid = "newkid-%s-%s" % (nsim,kid)
        simprogeny = genome.mate(sireobj,damobj,newkid,sex)
        
        if thisTL.genotypeout is not None :
            for chrom in oldkid.genotype.keys():
                thisTL.genotypeout.write("%s\t%s\n" % (newkid, np.array2string(simprogeny.genotype[chrom][0],separator='\t', threshold=sys.maxsize)) )
            for chrom in oldkid.genotype.keys():
                thisTL.genotypeout.write("%s\t%s\n" % (newkid, np.array2string(simprogeny.genotype[chrom][1],separator='\t', threshold=sys.maxsize)) )
        
        #form {chromosome:{0:patcregion, 1:matcregion}}
        detectedCrossoverRegions:Dict[int,Dict[int,Tuple]] = predictCrossoverRegions(simprogeny, sireobj, damobj)
        for chrom in oldkid.genotype.keys():   
            patcross = simprogeny.paternal_crossovers[chrom][1]
            matcross = simprogeny.maternal_crossovers[chrom][1]
            actualCrossPoints[chrom][i] = {0:patcross, 1:matcross}
            simulatedCrossPoints[chrom][i] = detectedCrossoverRegions[chrom]
            patpredict_starts, patpredict_lengths = detectedCrossoverRegions[chrom][0]
            #patpredict_ends = patpredict_starts+patpredict_lengths
            # for cross in patcross:
                # withintarget = [cross >= x for x in patpredict_starts] and [cross <= x for x in patpredict_ends]
                # if sum(withintarget) == 0 :
                    # print("chr %s crosses %s starts %s lengths %s" % (chrom, patcross, patpredict_starts, patpredict_ends))
                    # print("cross %s does not have a prediction?" % (cross))
                    # print("%s\n%s\n%s" % (sireobj.genotype[chrom][0][cross-10:cross+10], 
                                      # sireobj.genotype[chrom][1][cross-10:cross+10],
                                      # simprogeny.genotype[chrom][0][cross-10:cross+10]))
                    # print("sum of genotypes on chr pp %s pm %s" % (sum(sireobj.genotype[chrom][0]), sum(sireobj.genotype[chrom][1])))
    return(actualCrossPoints, simulatedCrossPoints)

def predictCrossoverRegions(kid, sireobj, damobj, paternal_strand=0,maternal_strand=1) -> Dict[int,Dict[int,Tuple]]:
    detectedCrossoverRegions: Dict[int,Dict[int,Tuple]] = {}
    for chrom in kid.data.keys():
        patcr = predictcrosspoints(
            kid.data[chrom][kid.paternal_strand], #paternal strand
            sireobj.data[chrom][kid.paternal_strand], #paternal paternal
            sireobj.data[chrom][kid.maternal_strand], ignorevalue=9, 
            paternal_strand=kid.paternal_strand,maternal_strand=kid.maternal_strand) #paternal maternal
        runvalue, runstart, runlength = find_runs(patcr)
        exactTransition = [True if i > 0 and runvalue[i-1] < 3 and x < 3 else False for i,x in enumerate(runvalue)]
        starts = runstart[np.logical_or(runvalue == 3,exactTransition)]
        xoverlength = runlength
        xoverlength[exactTransition] = 0
        lengths = xoverlength[np.logical_or(runvalue == 3,exactTransition)]
        patcregion = (starts, lengths)
        
        matcr = predictcrosspoints(
            kid.data[chrom][kid.maternal_strand],#maternal strand        
            damobj.data[chrom][kid.paternal_strand], #maternal paternal
            damobj.data[chrom][kid.maternal_strand], ignorevalue=9,
            paternal_strand=kid.paternal_strand,maternal_strand=kid.maternal_strand) #maternal maternal
        runvalue, runstart, runlength = find_runs(matcr)
        exactTransition = [True if i > 0 and runvalue[i-1] < 3 and x < 3 else False for i,x in enumerate(runvalue)]
        starts = runstart[np.logical_or(runvalue == 3,exactTransition)]
        xoverlength = runlength
        xoverlength[exactTransition] = 0
        lengths = xoverlength[np.logical_or(runvalue == 3,exactTransition)]
        matcregion = (starts, lengths)
        
        detectedCrossoverRegions[chrom] = {maternal_strand:matcregion, paternal_strand:patcregion}
    return(detectedCrossoverRegions)

def writeCrossoverRegionsShortFormat(detectedregions:Dict[int,Dict[int,Tuple]], idkid, file, writemode="wt"):
    with openfile(file, writemode) as fout:
        for chrom, xovers in detectedregions.items():
            #print("ShortFormat %s %s" % (chrom, xovers))
            xoversstr = []
            if len(xovers[0][0]) > 0: #you can't do len on a zip so this is easier
                patcregion = list(zip(xovers[0][0], xovers[0][1])) #structure is ([starts], [lengths]) so we zip to start,length tuples
                xoversstr.extend(["p\t%s\t%s" % (r_starts,r_length) for r_starts,r_length in patcregion])
            if len(xovers[1][0]) > 0:
                matcregion = list(zip(xovers[1][0], xovers[1][1]))
                xoversstr.extend(["p\t%s\t%s" % (r_starts,r_length) for r_starts,r_length in matcregion])
            line = "%s\t%s\t%s\n" % (idkid, chrom, "\t".join(xoversstr))
            fout.write(line)

def writeCrossoverRegionsBedFormat(detectedregions:Dict[int,Dict[int,Tuple]], idkid, file, writemode="wt"):
    with openfile(file, writemode) as fout:
        for chrom, xovers in detectedregions.items():
            #print("BedFormat %s %s" % (chrom, xovers))
            if len(xovers[0][0]) > 0: #you can't do len on a zip so this is easier
                patcregion = list(zip(xovers[0][0], xovers[0][1])) #structure is ([starts], [lengths]) so we zip to start,length tuples
                for r_starts,r_length in patcregion:
                        fout.write("%s\t%s\t%s\tpat_%s\t%.3f\n" % (chrom, r_starts, r_starts+r_length, idkid, 1/r_length))
            if len(xovers[1][0]) > 0:
                matcregion = list(zip(xovers[1][0], xovers[1][1]))
                for r_starts,r_length in matcregion:    
                        fout.write("%s\t%s\t%s\tmat_%s\t%.3f\n" % (chrom, r_starts, r_starts+r_length, idkid, 1/r_length))


