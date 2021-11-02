'''
Created on May 17, 2021

@author: mhindle
'''
import sys
import mgzip
import pickle, gzip
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import numpy as np
from matplotlib.lines import Line2D
import re
from scipy.ndimage.filters import uniform_filter1d
import functools
import cloudpickle
spre = re.compile("\s+")

with open('/mnt/md0/mhindle/gensys/etarsani/newdataimputation/chr2miss.map', mode='rt')  as file_handle:
    snpdetails = {bla['id']:bla for bla in [dict(zip(["chr", "id", "cm","bp"],x.strip().split("\t"))) for x in file_handle]}
    print("%s snp location details" % len(snpdetails))
    print("%s chr2 snp location details" % len([x for x in snpdetails.values() if x['chr'] == "2"]))
    
with open('/mnt/md0/mhindle/gensys/etarsani/newdataimputation/chr2miss.snps', mode='rt') as file_handle:
    snpIds= [x.strip() for x in file_handle]
    print("%s ordered snpids" % len(snpIds))
    print("%s chr2 ordered snpids" % len([x for x in snpIds if snpdetails[x]['chr'] == "2"]))

chr2snps = [x for x in snpIds if snpdetails[x]['chr'] == "2"]
#chr2snp2pos, snp2chromosome, gm = indexSnps(snpdetails, snpIds)

with open("/mnt/md0/mhindle/gensys/etarsani/plink.frq", "rt") as file_handle:
    header = spre.split(file_handle.readline().strip())
    mafvalues = {bla['SNP']:float(bla['MAF']) for bla in [dict(zip(header,spre.split(x.strip()))) for x in file_handle.readlines()]}

smooth_maf = uniform_filter1d([mafvalues[x] for x in chr2snps], size=20)

LDUmapvalues = {}
kbmapvalues = {}

with open("/mnt/md0/mhindle/gensys/etarsani/newdataimputation/ldmap/chr2miss.map", "rt") as file_handle:
    header = ["ID","Locus","kb map","LDU map"]
    for line in file_handle.readlines():
        if not line.startswith('#'):
            content = spre.split(line.strip())
            if len(content) == 4 :
                linedetails = dict(zip(header,content))
                LDUmapvalues[linedetails["Locus"]] = float(linedetails["LDU map"])
                kbmapvalues[linedetails["Locus"]] = float(linedetails["LDU map"])
print(LDUmapvalues)

out_dir = "/mnt/md0/mhindle/gensys/chr3sim9b/sim_crossovers/population_sim"

with gzip.open("/mnt/md0/mhindle/gensys/chr3sim9/haplotypes.pickle.gz", "rb") as gen_pickle:
    haplotypes = pickle.load(gen_pickle)


phapArrays = np.array([x[2].phap for x in haplotypes.values()], dtype=int)
mhapArrays = np.array([x[2].mhap for x in haplotypes.values()], dtype=int)

width = 50
observations = phapArrays.shape[1]
haplotypediversity = np.zeros(observations,dtype=int)
haplotypediversity_phap = np.zeros(observations,dtype=int)
haplotypediversity_mhap = np.zeros(observations,dtype=int)

print("windows")
for i, end in enumerate(range(width,phapArrays.shape[1],1)) :
    if i % 100 == 0 :
        print("%s done of %s" % (i,observations))
    unique_phap = set([functools.reduce(lambda total, d: 10 * int(total) + d, row, 0) for row in phapArrays[:,end-width:end]])
    unique_mhap = set([functools.reduce(lambda total, d: 10 * int(total) + d, row, 0) for row in mhapArrays[:,end-width:end]])
    unique = unique_phap.union(unique_mhap)
    haplotypediversity_phap[i] = len(unique_phap)
    haplotypediversity_mhap[i] = len(unique_mhap)
    haplotypediversity[i] = len(unique)

with mgzip.open('%s/haplotypediversity.pickle.gz' % out_dir, "wb", thread=12) as gen_pickle:
    cloudpickle.dump(haplotypediversity, gen_pickle, protocol=pickle.HIGHEST_PROTOCOL)
with mgzip.open('%s/haplotypediversity_phap.pickle.gz' % out_dir, "wb", thread=12) as gen_pickle:
    cloudpickle.dump(haplotypediversity, gen_pickle, protocol=pickle.HIGHEST_PROTOCOL)
with mgzip.open('%s/haplotypediversity_mhap.pickle.gz' % out_dir, "wb", thread=12) as gen_pickle:
    cloudpickle.dump(haplotypediversity, gen_pickle, protocol=pickle.HIGHEST_PROTOCOL)

print("done")
with PdfPages('/mnt/md0/mhindle/gensys/chr3sim9b/sim_crossovers/population_sim/chr2/haplotypediversitydensity.pdf' ) as pdf:
    fig =plt.figure(figsize=(16,9), dpi= 300, facecolor='w', edgecolor='k')
    plt.plot(range(1,len(haplotypediversity_phap)+1,1),
                             haplotypediversity_phap, color="blue", lw=0.2)
    pdf.savefig()
    plt.close(fig)
    fig =plt.figure(figsize=(16,9), dpi= 300, facecolor='w', edgecolor='k')
    plt.plot(range(1,len(haplotypediversity_phap)+1,1),
                             haplotypediversity_phap, color="pink", lw=0.2)
    pdf.savefig()
    plt.close(fig)
    fig =plt.figure(figsize=(16,9), dpi= 300, facecolor='w', edgecolor='k')
    plt.plot(range(1,len(haplotypediversity)+1,1),
                             haplotypediversity, color="black", lw=0.2)
    pdf.savefig()
    plt.close(fig)
    
    fig =plt.figure(figsize=(16,9), dpi= 300, facecolor='w', edgecolor='k')
    plt.plot(range(1,len(haplotypediversity_phap)+1,1),
                             np.cumsum(haplotypediversity_phap), color="blue", lw=0.2)
    plt.plot(range(1,len(haplotypediversity_mhap)+1,1),
                             np.cumsum(haplotypediversity_mhap), color="pink", lw=0.2)
    pdf.savefig()
    plt.close(fig)
    
    fig =plt.figure(figsize=(16,9), dpi= 300, facecolor='w', edgecolor='k')
    plt.plot(range(1,len(haplotypediversity_phap)+1,1),
                             np.cumsum(haplotypediversity_phap), color="blue", lw=0.2)
    pdf.savefig()
    plt.close(fig)
    
    

with PdfPages('/mnt/md0/mhindle/gensys/chr3sim9b/sim_crossovers/population_sim/chr2/lengths_plotsims.pdf' ) as pdf:
    densitylengthposition = pickle.load(gzip.open('/mnt/md0/mhindle/gensys/chr3sim9b/sim_crossovers/population_sim/chr2/densitylengthposition.pickle.gz'))
    
    maxlengths = densitylengthposition.shape[1]
    
    fig =plt.figure(figsize=(16,9), dpi= 300, facecolor='w', edgecolor='k')
    pos = plt.imshow(densitylengthposition.astype(float).transpose(), 
                     interpolation='none', cmap=plt.cm.get_cmap('inferno').reversed(), origin='lower', aspect=3,norm=LogNorm())
    plt.plot(range(0, len(smooth_maf)),
                         smooth_maf*200, color="blue", lw=1)
    secaxy = plt.gca().secondary_yaxis('right', functions=((lambda a: a/maxlengths), (lambda a: a*maxlengths)))
    secaxy.set_ylabel('MAF')
    secaxy.set_color('blue')
    plt.xlabel('N genomic position')
    plt.ylabel('length of ambiguity region')
    plt.title('Location density by length of ambiguity x-over region')
    cb = fig.colorbar(pos, pad=0.1)
    cb.set_label('log(N)')
    sns.despine(top=True, right=True, left=False, bottom=False)
    pdf.savefig()
    plt.close(fig)

    fig =plt.figure(figsize=(16,9), dpi= 300, facecolor='w', edgecolor='k')
    plt.yscale('log')
    pos = plt.imshow(densitylengthposition.astype(float).transpose(), 
                     interpolation='none', cmap=plt.cm.get_cmap('inferno').reversed(), origin='lower', aspect=1000,norm=LogNorm())
    plt.plot(range(0, len(smooth_maf)),
                         smooth_maf*200, color="blue", lw=1)
    secaxy = plt.gca().secondary_yaxis('right', functions=((lambda a: a/maxlengths), (lambda a: a*maxlengths)))
    secaxy.set_ylabel('MAF')
    secaxy.set_color('blue')
    plt.xlabel('N genomic position')
    plt.ylabel('length of ambiguity region')
    plt.title('Location density by length of ambiguity x-over region')
    cb = fig.colorbar(pos, pad=0.1)
    cb.set_label('log(N)')
    sns.despine(top=True, right=True, left=False, bottom=False)
    pdf.savefig()
    plt.close(fig)
    
    densitylengthposition = pickle.load(gzip.open('/mnt/md0/mhindle/gensys/chr3sim9b/sim_crossovers/population_sim/chr2/densitylengthposition.pickle.gz'))
    
    fig =plt.figure(figsize=(16,9), dpi= 500, facecolor='w', edgecolor='k')
    pos = plt.imshow(densitylengthposition.astype(float).transpose(), 
                     interpolation='none', cmap=plt.cm.get_cmap('inferno').reversed(), origin='lower', aspect=15,norm=LogNorm())
    plt.plot(range(0, len(smooth_maf)),
                         smooth_maf*200, color="blue", lw=1)
    secaxy = plt.gca().secondary_yaxis('right', functions=((lambda a: a/200), (lambda a: a*200)))
    secaxy.set_ylabel('MAF')
    secaxy.set_color('blue')
    plt.ylim(0,200)
    plt.xlabel('N genomic position')
    plt.ylabel('length of ambiguity region')
    plt.title('Location density by length of ambiguity x-over region')
    cb = fig.colorbar(pos, pad=0.1)
    cb.set_label('log(N)')
    sns.despine(top=True, right=True, left=False, bottom=False)
                
    pdf.savefig()
    plt.close(fig)
    
    smooth_all = uniform_filter1d(densitylengthposition.astype(float).transpose().sum(axis=0).tolist(), size=20)
    smooth_above50 = uniform_filter1d(densitylengthposition.astype(float).transpose()[50:,:].sum(axis=0).tolist(), size=20)
    smooth_above20 = uniform_filter1d(densitylengthposition.astype(float).transpose()[25:,:].sum(axis=0).tolist(), size=20)
    smooth_below20 = uniform_filter1d(densitylengthposition.astype(float).transpose()[0:25,:].sum(axis=0).tolist(), size=20)
   
    print(len((densitylengthposition.astype(float).transpose().sum(axis=0).tolist())))
    print(len(smooth_all))
         
    def scale_number(unscaled, to_min, to_max, from_min, from_max):
        return (to_max-to_min)*(unscaled-from_min)/(from_max-from_min)+to_min     

    def scale_list(l, to_min, to_max, from_min, from_max):
        return [scale_number(i, to_min, to_max, from_min, from_max) for i in l]
         
    fig =plt.figure(figsize=(16,9), dpi= 500, facecolor='w', edgecolor='k')
    asProbs = [x/sum(smooth_all) for x in smooth_all]
    asProbs_above20 = [x/sum(smooth_above20) for x in smooth_above20]
    asProbs_below20 = [x/sum(smooth_below20) for x in smooth_below20]
    asProbs_above50 = [x/sum(smooth_above50) for x in smooth_above50]
    print(max(asProbs))
    plt.plot(range(0, len(asProbs)),
                         asProbs, color="red", lw=1)
    plt.plot(range(0, len(asProbs_above20)),
                         asProbs_above20, color="green", lw=1)
    plt.plot(range(0, len(asProbs_below20)),
                         asProbs_below20, color="grey", lw=1)
    plt.plot(range(0, len(smooth_maf)),
                         scale_list(smooth_maf,0,max(asProbs),0, max(smooth_maf)), color="blue", lw=0.5)
    secaxy = plt.gca().secondary_yaxis('right', 
                                       functions=(lambda a: scale_number(a,min(asProbs),max(asProbs), min(smooth_maf),max(smooth_maf)), 
                                                  lambda a: scale_number(a,min(smooth_maf),max(smooth_maf),min(asProbs),max(asProbs))))
    secaxy.set_ylabel('MAF')
    secaxy.set_color('blue')
    plt.xlabel('N genomic position')
    plt.ylabel('p of region at position')
    plt.title('MAF vs p of region at position at different thresholds')
    plt.legend([
        Line2D([0], [0], color="red", lw=1),
        Line2D([0], [0], color="green", lw=1),
        Line2D([0], [0], color="grey", lw=1)] , ["all regions","regions > 20 snps","regions < 20 snps"])
    
    sns.despine(top=True, right=True, left=False, bottom=False)
                
    pdf.savefig()
    plt.close(fig)
    
    fig =plt.figure(figsize=(16,9), dpi= 500, facecolor='w', edgecolor='k')
    asProbs = [x/sum(smooth_all) for x in smooth_all]
    asProbs_above20 = [x/sum(smooth_above20) for x in smooth_above20]
    asProbs_below20 = [x/sum(smooth_below20) for x in smooth_below20]
    asProbs_above50 = [x/sum(smooth_above50) for x in smooth_above50]
    print(max(asProbs))
    ldu = [x for x in [LDUmapvalues[x] for x in chr2snps if x in LDUmapvalues.keys()]]
    kbmap = [x for x in [kbmapvalues[x] for x in chr2snps if x in kbmapvalues.keys()]]
    plt.plot(range(0, len(asProbs)),
                         asProbs, color="red", lw=1)
    #plt.plot(range(0, len(asProbs_above20)),
    #                    asProbs_above20, color="green", lw=1)
    #plt.plot(range(0, len(asProbs_below20)),
    #                     asProbs_below20, color="grey", lw=1)
    plt.plot(scale_list([kbmapvalues[x] for x in chr2snps if x in LDUmapvalues.keys()],0,len(asProbs),0,max(kbmap)),
                         scale_list(ldu,0,max(asProbs), 0, max(ldu)), color="blue", lw=1)
    plt.plot(range(width,phapArrays.shape[1],1),
                         np.cumsum(haplotypediversity), color="green", lw=1)
        
    secaxy = plt.gca().secondary_yaxis('right', 
                                           functions=(lambda a: scale_number(a,min(asProbs),max(asProbs), min(ldu),max(ldu)), 
                                                      lambda a: scale_number(a,min(ldu),max(ldu),min(asProbs),max(asProbs))))
    
    secaxy.set_ylabel('LDU')
    secaxy.set_color('blue')
    plt.xlabel('N genomic position')
    plt.ylabel('p of region at position')
    plt.title('LDU vs p of region at position at different thresholds')
    # plt.legend([
        # Line2D([0], [0], color="red", lw=1),
        # Line2D([0], [0], color="green", lw=1),
        # Line2D([0], [0], color="grey", lw=1)] , ["all regions","regions > 20 snps","regions < 20 snps"])
    plt.legend([
        Line2D([0], [0], color="red", lw=1),
        Line2D([0], [0], color="green", lw=1),
        Line2D([0], [0], color="blue", lw=1)] , ["all regions","haplotype diversity","LDU"])
    
    sns.despine(top=True, right=True, left=False, bottom=False)
                
    pdf.savefig()
    plt.close(fig)
    
    # fig =plt.figure(figsize=(16,9), dpi= 500, facecolor='w', edgecolor='k')
    # #plt.yscale('log')
    # #plt.xscale('log')
    # #plt.plot(np.cumsum(smooth_maf),
    # #                     np.cumsum(smooth_maf), color="black", lw=1)
    # plt.plot(np.cumsum(smooth_maf),
                         # np.cumsum(asProbs), color="red", lw=1)
    # plt.plot(np.cumsum(smooth_maf),
                         # np.cumsum(asProbs_above20), color="green", lw=1)   
    # plt.plot(np.cumsum(smooth_maf),
                         # np.cumsum(asProbs_below20), color="grey", lw=1)   
    # plt.plot(np.cumsum(smooth_maf),
                         # np.cumsum(asProbs_above50), color="cyan", lw=1)     
    # plt.xlabel('cumulative sum of p of region at position')
    # plt.ylabel('cumulative sum of MAF')
    # plt.title('MAF vs p of region at position at different thresholds')
    # plt.legend([
        # #Line2D([0], [0], color="black", lw=1),
        # Line2D([0], [0], color="red", lw=1),
        # Line2D([0], [0], color="cyan", lw=1),
        # Line2D([0], [0], color="green", lw=1),
        # Line2D([0], [0], color="grey", lw=1)
        # ] , [#"exact correlation",
             # "all regions","regions > 50 snps","regions > 20snps","regions < 20 snps"])
             #
    # sns.despine(top=True, right=True, left=False, bottom=False)
    #
    # pdf.savefig()
    # plt.close(fig)
    #
    fig =plt.figure(figsize=(10,10), dpi= 300, facecolor='w', edgecolor='k')
    plt.plot(range(width,phapArrays.shape[1],1),
                         np.cumsum(haplotypediversity), color="black", lw=1)
    maxldmap = sum(haplotypediversity)
    plt.plot(range(0, len(asProbs)),
                         scale_list(np.cumsum(asProbs),0 ,maxldmap, 0, sum(asProbs)), color="red", lw=1)
    plt.plot(range(0, len(asProbs_above20)),
                         scale_list(np.cumsum(asProbs_above20),0,maxldmap, 0, sum(asProbs_above20)), color="green", lw=0.5)   
    plt.plot(range(0, len(asProbs_below20)),
                         scale_list(np.cumsum(asProbs_below20),0,maxldmap, 0, sum(asProbs_below20)), color="grey", lw=0.5)   
    plt.plot(range(0, len(asProbs_above50)),
                         scale_list(np.cumsum(asProbs_above50),0,maxldmap, 0, sum(asProbs_above50)), color="cyan", lw=0.5)
    pdf.savefig()
    plt.close(fig)
    
    