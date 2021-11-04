# genofix
Using genomic data to fix pedigree and genotypes

## requirements
* [python3](https://www.python.org/download/releases/3.0/)
* [pandas 1.3.4](https://pandas.pydata.org/)
* [matplotlib 3.4.3](https://matplotlib.org/)
* [seaborn 0.11.2](https://seaborn.pydata.org/)
* [pgmpy 0.1.16](https://github.com/pgmpy/pgmpy)
* [networkx 2.6.2](https://networkx.org/)
* [numpy 1.21](https://numpy.org/)

## main entry points
`./genofix.py`
Correct a genotype matrix 

`./simulate/simulate.py`
Generate a gold standard genotype matrix (using random or by mating known founders and offspring to fill a pedigree), add known quantity of genotyping errors, and correct the genotype matrix (with statistics on precision and recall) 

`./utils/extract_founders.py`
Create a list of founders from a pedigree, which can be contrained with a genotype file when a more complete pedigree is part sequenced

`./phase/evaluate.py`
Compare two phase prediction outputs in AlphaImpute2 format

`./plotting/plot_rank_statistics.py`
Create histograms and heat maps for rank statistics created by simulate.py
