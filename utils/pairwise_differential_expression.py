# IMPORTS
# data related
import anndata as AnnData
import numpy as np
import pandas as pd
import scanpy as sc

# plotly plots -- heatmaps
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# utils -- loading bar
import tqdm


# imports needed for traininng logistic regression model
# imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_precision_recall_curve, plot_confusion_matrix, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics.pairwise import paired_distances
from sklearn.feature_selection import RFE

# copy models
from copy import deepcopy

# saving and loading models
from joblib import dump, load


# PLOT AESTHETIC RELATED/PRESESTS
PLOTLY_COLOR_TEMPLATE = 'plotly_white'
HEATMAP_COLOR = [[0,'rgb(250,250,250)'], [1,'rgb(102,0,204)']]
HEATMAP_MIN_COLOR = 0
HEATMAP_MAX_COLOR = 4


# UTILITY FUNCTIONS
# function to perform pairwise differential expression analysis
def pairwise_differential_analysis(adata, target_cluster, leiden_name, min_logfc, pval_cutoff):
    """
    [Summary]
        Performs a pairwise comparison for a single target cluster. If there are N clusters this 
        function will perform a pairwise comparison N-1 times iteratively. 
    [Parameters]
            adata          : your anndata object
            target_cluster : A target cluster/group of interest within the columns containing
                             clusters/groups within the AnnData.obs metadata of an anndata object
            leiden_name    : The column name or key containing cluster/groups
            min_logfc      : Minimum fold change for differential expression
            pval_cutoff    : P-value cutoff for differential expression 
    [Returns]
        Returns a list of dataframes of every comparison made during pairwise 
        for the target cluster
    """
    reference_clusters = adata.obs.loc[adata.obs[leiden_name] != target_cluster][leiden_name].unique()
    target_cluster_results = []
    for cluster_reference in reference_clusters:
        # key labels for the differential expression analysis -- used for accesing the
        # current pairwise results
        comparison_key = f'target_cluster_{target_cluster}vs{cluster_reference}'
        
        # perform differential expression analysis to get cluster specific genes for target cluster vs
        # a reference cluster (another cluster not the target cluster)
        sc.tl.rank_genes_groups(
            adata,
            use_raw=False,
            key_added=comparison_key,
            groupby=leiden_name,
            groups=[target_cluster], 
            reference=cluster_reference,
        )
        
        # extract the result -- get the dataframe
        de_results = sc.get.rank_genes_groups_df(
            adata, 
            key=comparison_key, 
            group=[target_cluster], 
            log2fc_min=min_logfc,
            pval_cutoff=pval_cutoff,
        )
        
        # add a versus columns to the dataframe
        de_results['group'] = de_results.shape[0] * [target_cluster]
        de_results['compared_against'] = de_results.shape[0] * [cluster_reference]
        
        # append the results
        target_cluster_results.append(de_results)
        
        # delete the key that was added -- this should declutter the anndat.uns object
        del adata.uns[comparison_key]
        
    return target_cluster_results


# function that adds pairwise analysis results to anndata
def add_pairwise_results_to_anndata(adata, pairwise_results, leiden_name):

    # ##########
    # Pairwise results 
    
    # create a pandas dataframe to store pairwise counts -- gene by cluster
    pairwise_counts_matrix = pd.DataFrame(0, index=adata.var_names, columns=adata.obs[leiden_name].values.categories)
    cluster_list = adata.obs[leiden_name].values.categories.tolist()
    df_mp_GroupCounts = pairwise_results.loc[:,['names','group']].groupby(['group','names']).agg({'names':'count'}).rename(columns={'names':'gene_counts'})
    
    for cluster in tqdm.tqdm(cluster_list, total=len(cluster_list), desc='Adding count values'):
        clust_specific_genes = df_mp_GroupCounts.loc[cluster,:].index.tolist()
        clust_specific_genes_count = df_mp_GroupCounts.loc[cluster].values.ravel()
        pairwise_counts_matrix.loc[clust_specific_genes,cluster] = clust_specific_genes_count

    # add the pairwise counts matrix to adata varm
    adata.varm['pairwise_cluster_count'] = pairwise_counts_matrix.copy()


    # ##########
    # Mean expression per cluster
    
    # create mean expression matrix
    expression_matrix = pd.DataFrame(adata.X.toarray().copy(), index=adata.obs_names, columns=adata.var_names)
    
    # add a cluster column to groupby and compute the means matrix with
    expression_matrix.loc[:,'leiden_cluster'] = adata.obs.loc[:,leiden_name].copy()

    # compute th mean matrix -- mean expression of genes per cluster
    mean_expression_matrix = expression_matrix.groupby('leiden_cluster').mean()

    # gene by clusters with entries being mean expression 
    adata.varm['cluster_means'] = mean_expression_matrix.T.copy()

    
# Plotting function to visualize pairwise analysis results
def plot_pairwise_results(adata, target_cluster, save=None):
    """
    [Summary]
        Plots pairwise results for the target cluster
    [Parameters]
            adata          : Anndata object with the pairwise results 
            target_cluster : A target cluster/group of interest within the columns containing
                             clusters/groups within the AnnData.obs metadata of an anndata object
    """
    
    # Pairwise results for the target cluster 
    cluster_res = adata.varm['pairwise_cluster_count'].loc[:,target_cluster].sort_values(ascending=False).to_frame().copy().rename(columns={target_cluster:'counts'})
    cluster_res = cluster_res.loc[cluster_res['counts'] > 0]
    exp_mat = adata.varm['cluster_means'].loc[cluster_res.index]

    # Histogram Plots
    Hist_Counts = go.Bar(
        x=cluster_res.index,
        y=cluster_res['counts'],
        marker=dict(
            color='slateblue'
        ),
        opacity=0.6
    )

    # Heatmap Plot
    Matrix_Heatmap = go.Heatmap(
        z=exp_mat.T,
        x=exp_mat.index,
        y=exp_mat.columns,
        colorscale=HEATMAP_COLOR,
        zmin=HEATMAP_MIN_COLOR,
        zmax=HEATMAP_MAX_COLOR,
        colorbar={
            'len': 0.68,
            'y': 0.40
        }
    )

    # Put the plot together
    subplot = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.3,0.9])
    plots = [Hist_Counts, Matrix_Heatmap]
    
    # add plotly graph objects onto the subplots
    for row in range(len(plots)):
        subplot.add_trace(
            plots[row],
            row=row + 1,
            col=1
        )
        
    # customize/update layouts
    subplot.update_layout(
        template=PLOTLY_COLOR_TEMPLATE,
        height=600,
        width=1000,
        title=dict(
            text=f'Cluster {target_cluster} | Pairwise Analysis Results -- Cluster Specific Genes'
        ),
        xaxis=dict(
            showgrid=False
        ),
        yaxis=dict(
            showgrid=False,
            title='Pairwise Comparison<br>Gene Counts'
        ),
        xaxis2=dict(
            showgrid=False,
        ),
        yaxis2=dict(
            showgrid=False,
            title='Mean Gene Expression<br>Across Leiden Clusters'
        ),
        xaxis3=dict(
            showgrid=False,
            title='Cluster Specific Genes'
        ),
        yaxis3=dict(
            visible=False,
            showgrid=False
        ),
    )

    # Save and show 
    if save is not None:
        subplot.write_image(f'{save}.png')
        subplot.write_html(f'{save}.html')

    return subplot


