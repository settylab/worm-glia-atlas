# worm-glia-atlas
This is a repository with tutorials and reproducibility notebooks for the worm glia scRNA-seq atlas available at https://wormglia.org.

## Environment Setup
------------------------------------------
To run the tutorial, certain packages need to be installed which  `requirements.txt` 
1. `conda=4.8.2`
2. `Python 3.8.10`
3. `pip=21.1.3`

installation using pip:<br>

```pip install -r envs/requirements.txt```

with conda
```
envName=worm-glia-atlas

conda env create -n "$envName" --file envs/environment.yaml

conda activate "$envName"
```



### Important Dependencies
The following packages need to be installed for running the notebooks (are also be installed as described above):
1. `scanpy`: https://scanpy.readthedocs.io/en/stable/installation.html
4. `sklearn`: https://scikit-learn.org/stable/install.html
2. `plotly`: https://plotly.com/python/getting-started/
3. `tqdm` : https://github.com/tqdm/tqdm#installation

<!-- ## $${\color{red}Pairwise\spacedifferential analysis}$$ -->
## Pairwise Differential Expression Analysis
------------------------------------------
The tutorial notebook for <b><i>pairwise differential expression analysis</i></b> is available <b>[here](https://github.com/settylab/worm-glia-atlas/blob/main/notebooks/pairwise-differential-results.ipynb)</b>.
Pairwise differential analysis performs pairwise differential analysis to identify cluster enriched genes rather than one-vs-all approach. 

##### <b>Inputs</b>
The input is an `anndata` object with normalized, log-transformed data and an `obs` variable containing information about clusters for pairwise comparison.
1. `anndata.obs[LEIDEN_NAME]`: The `.obs` field containing the groups to be used for the analysis

##### <b>Outputs</b>
The `anndata` object is updated with the following information
1. `adata.varm['pairwise_cluster_count']`: Gene X Cluster matrix indicating how many comparisons the gene is differential in.
2. `adata.varm['cluster_means']`: Gene X Cluster matrix of mean expression of gene per cluster.
3. HTML files of the pairwise analyses results can saved using the `plot_pairwise_results()` function by specifying a path to the `save` parameter.

## Feature Ranking Analysis 
------------------------------------------
The tutorial notebook for <b><i>Sheath/Socket</i> & <i>Pan-Glia</i> marker analysis</b> is available <b>here</b>. A `logistic regression` model is trained and employed for `binary classification` of cells using gene expression. Subsequently, a ranking of the learned features within the model is then performed with the objective being to rank features that are highly informative and correlated with the specified target classes or cell type. The tutorial demonstrates the analyses that is performed as described in the manuscript, where training of the model and ranking of the learned features is performed on Male only samples but can easily be extended to other datasets provided the appropriate inputs as detailed below.

##### <b>Inputs</b>
The key inputs for this analyses is the `anndata` object with normalized & log-transformed counts, imputed gene expression values as well as the following anndata attribute fields below:

- `anndata.obs[CLASS_LABELS]`: Where `CLASS_LABELS` is the name of the column in `anndata.obs` containing the ground truth labels for each cells in the anndata object.
- `anndata.obs[CLUSTER_LABELS]`: Where `CLUSTER_LABELS` is the name of the column in `anndata.obs` containing the cluster labels for each cells in the anndata object.
- `anndata.var[USE_GENES]`: Where `USE_GENES` is the name of the column in `anndata.var` containing boolean values that specifies whether a gene is to be used for analysis or ignored (default is `highly_variable` genes columns). 
- `anndata.layers[USE_LAYER]`: Where `USE_LAYER` is a key in `anndata.layers` dictionary corresponding to the imputed Cell X Gene count matrix. if not specified, will use the normalized and log-transformed counts as values for the constructed feature matrix & feature ranking analysis.

##### <b>Outputs</b>
The output of the analysis is a trained logistic regression model and an updated anndata object as follows:
- `anndata.obs['exclude_cells']`: A new column in the `anndata.obs`, added containing boolean values denoting which cells are excluded during model training and feature/gene ranking analysis if specific groups are specified to be ignored.
- `anndata.var['identified_genes']`: A new column in the `anndata.var`, added containing boolean values denoting which genes are used within the analysis after pre-filtering of the gene expression.
- `anndata.obs['data_splits']`: A new column in `anndata.obs`, added containing labels whether a cell belongs to training, validation or test datasets after splitting the the feature matrix and target vector accordingly.
- `anndata.uns['model_selection_metrics']`: A dataframe object containing mean scores of trained regularized models on training/validation/test datasets stored in `anndata.uns`.
- `anndata.uns['<target_class>_marker_results']`: A dictionary object containing results of the feature ranking analysis for a specified target class stored in `anndata.uns`. 
- `anndata.uns['<target_class>_probEst_Summary']`: A dataframe object containing the mean probability estimates for each 
- `anndata.uns['<target_class>_AUROCC_Summary']`: A dataframe object


## Citations
------------------------------------------
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
