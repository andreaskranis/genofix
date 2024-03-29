# GenoFix
Using empirical and Bayesian probabilistic models of Mendelian inheritance to fix pedigree and genotypes

## lib requirements
* [python3](https://www.python.org/download/releases/3.0/)
* [pandas 1.3.4](https://pandas.pydata.org/) `pip3 install pandas`
* [matplotlib 3.4.3](https://matplotlib.org/) `pip3 install matplotlib`
* [seaborn 0.11.2](https://seaborn.pydata.org/) `pip3 install seaborn`
* [pgmpy 0.1.16](https://github.com/pgmpy/pgmpy) `pip3 install pgmpy`
* [networkx 2.6.2](https://networkx.org/) `pip3 install networkx`
* [numpy 1.21](https://numpy.org/) `pip3 install numpy`
* [tqdm 4.62.3](https://github.com/tqdm/tqdm) `pip3 install tqdm`
* [cloudpickle 2.0.0](https://github.com/cloudpipe/cloudpickle) `pip3 install cloudpickle`
* [pyarrow 8.0.0]() `pip3 install pyarrow`
* mgzip `pip3 install mgzip`
* psutil `pip3 install psutil`
* sortedcontainers `pip3 install sortedcontainers`

or with conda...

`conda create -n genofix python=3.10.0`

`source activate genofix`

`pip3 install pandas matplotlib seaborn pgmpy networkx numpy tqdm cloudpickle mgzip psutil sortedcontainers`

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
