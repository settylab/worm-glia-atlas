# worm-glia-atlas
This is a repository with tutorials and reproducibility notebooks for the worm glia scRNA-seq atlas available at https://wormglia.org.

### Pairwise differential analysis 
The tutorial notebook for pairwise differential analysis is available [here](https://github.com/settylab/worm-glia-atlas/blob/main/notebooks/pairwise-differential-results.ipynb).
Pairwise differential analysis performs pairwise differential analysis to identify cluster enriched genes rather than one-vs-all approach. 

The following packages need to be installed for running this notebook:
1. `scanpy`: https://scanpy.readthedocs.io/en/stable/installation.html
2. `plotly`: https://plotly.com/python/getting-started/
3. `tqdm` : https://github.com/tqdm/tqdm#installation

Input is `anndata` object with normalized, log-transformed data and an `obs` variable containing information about clusters for pairwise comparison. The `anndata` object is updated with the following informatio
1. `adata.varm['pairwise_cluster_count']`: Gene X Cluster matrix indicating how many comparisons the gene is differential in.
2. `adata.varm['cluster_means']`: Gene X Cluster matrix of mean expression of gene per cluster.

### Citations
Worm glia atlas manuscript is available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.03.21.533668v1). Please cite our paper if you use these tutorials for your analyses:

```
@article {Purice2023.03.21.533668,
	author = {Maria D. Purice and Elgene J.A. Quitevis and R. Sean Manning and Liza J. Severs and Nina-Tuyen Tran and Violet Sorrentino and Manu Setty and Aakanksha Singhvi},
	title = {Molecular heterogeneity of C. elegans glia across sexes},
	elocation-id = {2023.03.21.533668},
	year = {2023},
	doi = {10.1101/2023.03.21.533668},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/03/24/2023.03.21.533668},
	eprint = {https://www.biorxiv.org/content/early/2023/03/24/2023.03.21.533668.full.pdf},
	journal = {bioRxiv}
}

```
