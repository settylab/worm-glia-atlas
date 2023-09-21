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
        sc.tl.rank_genes_groups(
            adata,
            use_raw=False,
            key_added=f'target_cluster_{target_cluster}vs{cluster_reference}',
            groupby=leiden_name,
            groups=[target_cluster], 
            reference=cluster_reference,
        )
        
        de_results = sc.get.rank_genes_groups_df(
            adata, 
            key=f'target_cluster_{target_cluster}vs{cluster_reference}', 
            group=[target_cluster], 
            log2fc_min=min_logfc,
            pval_cutoff=pval_cutoff,
        )
        
        # add a versus columns to the dataframe
        de_results['group'] = de_results.shape[0] * [target_cluster]
        de_results['compared_against'] = de_results.shape[0] * [cluster_reference]
        
        target_cluster_results.append(de_results)
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
        subplot.write_image(save)

    return subplot

# Computing linkage for hierarchical clustering
def compute_linkage(data_obsm, data_obs_index, data_obs_label, feature_names, 
                    dist_metric='cosine', linkage_dist_metric='euclidean', linkage_method='average'):
    # params: data, feature_names, obs_label, metric, linkage_method
    '''
        [Summary]
            Computes a linkages to be used for constructing dendrograms. Use linkage returned
            with scipy.cluster.hierarchy.dendrogram function to visualize results
        [Parameters]
            data_obsm           : The features and values to be used to compute linkages e.g. gene expression values, PCs
            data_obs_index      : The designated name or index for each of the sample rows in data_obsm
            data_obs_label      : An added categorical label column to -- used for grouping the rows together
            feature_names       : The column names or names of the features
            dist_metric         : The distance metric to use e.g. cosine/euclidean etc. see sklearn.metrics.pairwise_distances documentation
            linkage_dist_metric : The distance metric used for linkage method see scipy.cluster.hierarchy.linkage documentation 
            linkage_method      : Linkage method used to compute linkages e.g. single, average, complete etc.
        [Returns]
            Returns the computed linkages, group_labels to be used for dendrogram as well as the 
            paired distance matrix for each group (unordered/unclustered matrix)
    '''
    # build the data matrix and add associated group labels
    features_matrix = pd.DataFrame(data_obsm, index=data_obs_index, columns=feature_names)
    features_matrix.loc[:, 'clust_labels'] = data_obs_label
    
    # compute the mean of the values per label in the data set and create a new matrix
    features_mean = features_matrix.groupby('clust_labels').mean()
    
    # compute the pairwise distances of the groups in the data
    features_dist = pd.DataFrame(
        pairwise_distances(features_mean, metric=dist_metric), 
        index=features_mean.index, columns=features_mean.index)
    
    # compute the linkages 
    compute_linkage = scipy.cluster.hierarchy.linkage(features_dist, method=linkage_method, metric=linkage_dist_metric)
    
    return compute_linkage, features_mean.index, features_dist

