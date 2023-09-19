# important imports
import anndata as AnnData
import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# distance metrics related imports
import scipy
from scipy.stats import spearmanr
from scipy.stats import zscore
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances

# for t-test to determine if 
from scipy.stats import ttest_ind


# utils
import tqdm
import os 
import glob

# multiprocessing/parallelizing tasks
from joblib import Parallel, delayed, dump, load

# imports needed for marker selection
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


def filter_features(ad_data, use_feature_set='highly_variable', frac_expr_thresh=0.45, cluster_expr_thresh=1, 
                      cluster_label='Cluster_Celltype', ignore_clusters=None):
    '''
        [Summary]
            Gene/feature pre-filtering function that identifies genes that will be used for gene selection
            analyses. Modifies anndata objects inplace with labels for identified genes and indcluded/excluded 
            clusters. Labels are added as new columns in anndata.var and anndata.obs respectively.
        [Parameters]
            ad_data              : an anndata object
            use_feature_set      : the set of features to filter, checks adata.var (by default highly_variable)
            fract_expr_thresh    : threshold for determining whether a gene is expressed in a cluster, default 0.45
            cluster_expr_thresh  : threshold for determining whether a gene is expressed across cluster, default 1
            cluster_label        : clusters/group label, checks adata.obs
            ignore_clusters      : label/group in cluster_label to ignore when filtering genes, list of group label
    '''
    # check if the specified use_features_set column is in the dataset -- anndata.var
    if use_feature_set not in ad_data.var.columns:
        raise Exception(f'\'{feature_set}\' is not a valid anndata.var column in the provided anndata.')
    
    # check if cluster_label column is in the dataset -- anndata.obs
    if cluster_label not in ad_data.obs.columns:
        raise Exception(f'\'{cluster_label}\' is not a valid anndata.obs columns in the provided anndata.')
    else:
        # add labels denoting which cells/cell group to exclude
        if ignore_clusters is not None:
            ad_data.obs.loc[:,'exclude_cells'] = ad_data.obs[cluster_label].isin(ignore_clusters)
        else:
            # if ignore_cluster is not none
            ad_data.obs.loc[:,'exclude_cells'] = ad_data.obs[cluster_label].isin([])
        
    # create meta_data with excluded cluster -- subset data that will be used for filtering/genes
    meta_data = ad_data[~ad_data.obs[f'exclude_cells'],ad_data.var_names[ad_data.var[use_feature_set]]].copy()
    
    # quantify expression of gene for each cluster
    fraction_hvg_exp = pd.DataFrame(0.0, index=meta_data.obs[cluster_label].unique().tolist(), columns=meta_data.var_names)
    for gene in tqdm.tqdm(meta_data.var_names, total=len(meta_data.var_names), desc='Identifying genes'):
        fraction_hvg_exp.loc[:,gene] = list(meta_data.obs[cluster_label][np.ravel(meta_data[:, gene].X.todense()) > 0]. \
                                            value_counts() / meta_data.obs[cluster_label].value_counts())
    
    # actually performing the filtering
    selected_features = fraction_hvg_exp.columns[(fraction_hvg_exp > frac_expr_thresh).sum() > cluster_expr_thresh]
    print(f'Number of genes identified: {len(selected_features)}')
    
    # add labels in the ad_data.var of boolean values used 
    ad_data.var.loc[:,'identified_genes'] = ad_data.var_names.isin(selected_features)
    

def split_data(ad_data, cluster_label):
    '''
        [Summary]
            Splits the data -- 70/20/10% data splits for each group within a provided anndata.obs column.
            Modifies anndata.obs in place, adds 'data_splits' column within anndata.obs specifying which
            cells are to be used for training/validation/test
        [Parameters]
            ad_data        : an anndata object data to use to create input/output labels for training
            cluster_label  : .obs columns denoting the leiden clusters, column to split with respect to
                             to ensure proper representation of each categories 
    '''    
    data = pd.DataFrame(index=ad_data.obs_names)
    data.loc[:,'clusters'] = ad_data.obs.loc[data.index, cluster_label].values.copy()

    # Create the train/validation/test splits (70/20/10)
    train, validation, test = [], [], []

    data = data.groupby('clusters')

    # create test/validation/test splits by looping through each cluster
    for cluster in tqdm.tqdm(data.groups.keys(), total=len(data.groups.keys()), desc='Splitting Data'):
        data_meta = data.get_group(cluster).iloc[:,:-1].copy() # remove the last 'clusters' columns

        # splitting process
        train_initMeta, testMeta = train_test_split(data_meta, test_size=0.1, random_state=42)
        trainMeta, validationMeta = train_test_split(train_initMeta, test_size=0.2, random_state=42)

        # append to the train/validation/test sets
        train.append(trainMeta)
        validation.append(validationMeta)
        test.append(testMeta)

    # add training/val/test labels
    train_idx = pd.DataFrame('train', index=pd.concat(train).index, columns=['data_label'])
    val_idx = pd.DataFrame('validation', index=pd.concat(validation).index, columns=['data_label'])
    test_idx = pd.DataFrame('test', index=pd.concat(test).index, columns=['data_label'])

    # add the dataset label into the anndata
    ad_data.obs.loc[:,'data_splits'] = pd.concat([train_idx, val_idx, test_idx]).loc[ad_data.obs_names,'data_label'].values.ravel()
    
    
def retrieve_IO_data(ad_data, class_label, use_layer, data_splits_label='data_splits', 
                     use_features='identified_genes', split=True):
    '''
        [Summary]
            A function to be called inside the training/model selection step. It 
            retrieves the training and validation and test data based on preexisting
            labels in ad_data
        [Parameters]
            ad_data           : anndata object
            class_label       : anndata.obs columns containing the ground truth labels for each observation
                                in ad_data
            use_layer         : anndata.layers, key containing the imputed gene expression, if not provided,
                                normalized-logtransformed counts will be used to construct the expression mantrix
            exclude_cells     : anndata.obs columns, containing boolean values, denoting cells to be excluded
                                from the analysis -- does not include these cells into the training/validation/test
                                sets.
            data_splits_label : anndata.obs columns, containig labels (train, validation, test) for each cells pre 
                                computed prior
        [Returns]
            train, validation, test sets -- containing feature matrix and 
            target vectors for classifier training packaged as tuples respectively
    '''
    # check if class_label exists in the anndata
    if class_label not in ad_data.obs.columns:
        raise Exception(f'\'{class_label}\' is not a valid anndata.obs column or \
                                    is not present in the input anndata object.')
    
    # check if data_splits exists in the anndata
    # if data_splits_label not in ad_data.obs.columns:
    #    raise Exception(f'\'{data_splits_label}\' is not a valid anndata.obs column or'
    #                                'is not present in the input anndata object.')
    
    # Remove any excluded cells from the columns and only select the specific filtered genes
    ad_data_meta = ad_data[~ad_data.obs['exclude_cells'], \
                           ad_data.var[use_features]].copy()
    
    # create the feature matrix -- if use_layer is not specified, 
    # use the default log normalized countmatrix
    if use_layer:
        input_data = pd.DataFrame(
            ad_data_meta.layers[use_layer],
            columns=ad_data_meta.var_names,
            index=ad_data_meta.obs_names
        )
    else:
        input_data = pd.DataFrame(
            ad_data_meta.X.toarray().copy(),
            columns=ad_data_meta.var_names,
            index=ad_data_meta.obs_names
        )
        
    # add the data_splits label onto the anndata
    # input_data.loc[:,data_splits_label] = ad_data_meta.obs.loc[:,data_splits_label].values.copy()
    
    # create the target feature vector -- make sure this is a dataframe
    # output_label = ad_data_meta.obs.loc[:,[class_label,data_splits_label]].copy()
    output_label = ad_data_meta.obs.loc[:,class_label].copy().to_frame()
    
    if data_splits_label in ad_data_meta.obs.columns and split:
        X_train, X_validation, X_test = input_data.loc[ad_data.obs[data_splits_label] == 'train',:], \
                                        input_data.loc[ad_data.obs[data_splits_label] == 'validation',:], \
                                        input_data.loc[ad_data.obs[data_splits_label] == 'test',:] 
        y_train, y_validation, y_test = output_label.loc[ad_data.obs[data_splits_label] == 'train',:], \
                                        output_label.loc[ad_data.obs[data_splits_label] == 'validation',:], \
                                        output_label.loc[ad_data.obs[data_splits_label] == 'test',:]

        return (X_train, y_train), (X_validation, y_validation), (X_test, y_test), (input_data, output_label)
    else:
        return input_data, output_label   
    
def train_model(ad_data, class_label, use_layer, inverse_of_regularization=None, \
                solver='liblinear', regularization_type='l1', save_path=None, return_all=False):
    '''
        [Summary]
            Trains a binary classifier model (logistic regression) to classify specified binary class
            based on gene expression.
        [Parameters]
            ad_data                   : anndata object from which to construct the feature matrix and target vector
            class_label               : anndata.obs column name containing binary class labels
            use_layer                 : anndata.layers field denoting which values to use for feature matrix -- imputed data
                                        is reccomended
            inverse_of_regularization : a list of inverse of regularization params
            solver                    : solver used for training the binary classifier, see sklearn.linear_model.LogisticRegression
                                        documentation for details
            regularization_type       : 
            save_all                  : boolean value, if True, saves all the model 
        [Returns]
            Returns a logistic regression model (sklearn.linear_model.LogisticRegression): The best model based on mean 
            accuracy on the validation dataset.
            
            If `return_all` is True, returns a tuple containing:
            - best_model: The best model based on mean accuracy on the validation dataset.
            - all_models (list of sklearn.linear_model.LogisticRegression): List of all trained models.
            - all_scores (list of float): List of mean accuracy scores corresponding to each trained model.
    '''
    # deterimine regualrization params
    if inverse_of_regularization is None:
        inverse_of_regularization = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        
    # get data -- train, validation and test sets
    train_data, validation_data, test_data, _ = retrieve_IO_data(ad_data, class_label, use_layer)
    
    # prepare the input output
    X_train, y_train = train_data
    X_validation, y_validation = validation_data
    X_test, y_test = test_data
    
    # store model train MEAN_ACCURACY score here
    model_train_score = {}
    # store model validation MEAN_ACCURACY score here
    model_validation_score = {}
    # store model train MEAN_ACCURACY score here
    model_test_score = {}
    
    # feature scores -- use for determining which feature each model find most important
    model_coefficients = {}

    # store the models here -- model.fit()
    model_trained = {}
    
    # create and train the model here
    for params in tqdm.tqdm(inverse_of_regularization, \
                            total=len(inverse_of_regularization), desc='Training Model(s)'):
        # create the model
        model = LogisticRegression(
            penalty=regularization_type,
            solver=solver,
            C=params,
            random_state=42
        )

        # train the model
        model_trained[f'model_{params}'] = model.fit(X_train.values, y_train.values.ravel())

        # model training metrics, MEAN_ACCURACY -- add it
        model_train_score[f'model_{params}'] = model.score(X_train.values, y_train.values.ravel())

        # model validation metrics, MEAN_ACCURACY  -- add it
        model_validation_score[f'model_{params}'] = model.score(X_validation.values, y_validation.values.ravel())

        # model test metrics, MEAN_ACCURACY  -- add it
        model_test_score[f'model_{params}'] = model.score(X_test.values, y_test.values.ravel())
    
    # select the best perfoming baseline model -- consolidate the metrics
    df_scores = pd.DataFrame(0.0,
    index=model_train_score.keys(), 
    columns=['train_score','validation_score','test_score'])

    for elems, cols in zip([model_train_score, model_validation_score, model_test_score],df_scores.columns):
        df_scores.loc[elems.keys(), cols] = elems
    
    # store metric information in the .uns attribute
    ad_data.uns['model_selection_metrics'] = df_scores.copy()
    
    # select best model
    selected_model = deepcopy(model_trained[df_scores.loc[:,'validation_score'].idxmax()])
    
    # identify non zeroed features post training
    features = pd.DataFrame(selected_model.coef_.ravel().copy(), index=X_train.columns, \
                            columns=['features']).sort_values('features')
    features.loc[:, 'coef_index'] = range(features.shape[0])
    remaining_features = (features.loc[:,'features'] != 0.0).sum()
    print('best_model:', df_scores.loc[:,'validation_score'].idxmax())
    print(f'{remaining_features} out of {X_train.shape[1]} features remain in the selected model.')
    
    # save the result -- save only the selected model
    # if a path is provided -- check
    trained_models_path = os.path.join(save_path, 'selected_model')
    if save_path and not os.path.exists(trained_models_path):
        try:
            os.makedirs(trained_models_path)
            print(f"Directory '{save_path}' created successfully -- saving results to {save_path}.")
            dump(selected_model, f'{trained_models_path}/baseline_model.pkl')
            features.to_csv(f'{trained_models_path}/baseline_model_features.csv')
            
            # save other models
            for key, val in model_trained.items():
                dump(val, f'{trained_models_path}/{key}.pkl')
            
        except OSError as e:
            print(f"Error creating directory '{save_path}': {e}")
    # else os.path.exists(trained_models_path):
    else:
        print(f"Directory '{save_path}' already exists -- saving results to {save_path}.")
        dump(selected_model, f'{trained_models_path}/baseline_model.pkl')
        features.to_csv(f'{trained_models_path}/baseline_model_features.csv')
        
        # save other models
        for key, val in model_trained.items():
            dump(val, f'{trained_models_path}/{key}.pkl')

    # returns
    if not return_all:
        return deepcopy(selected_model)
    else:
        return deepcopy(selected_model), model_trained, df_scores

    
# function to compute performance a model given a combination of features
def determine_combination(*args):
    '''
        [Summary]
            Parallelizable function to compute probability estimates for a specified class 
            based on a specified combination of features.
        [Parameters]
            *args (tuple) : A tuple containing the following elements:
                - current_features (list) : List of features included in the current combination.
                - feature (str)           : The feature being evaluated in this function call in -- to be combined with the current feaute.
                - input_data (DataFrame)  : Input data containing features -- feature matrix Cell X Genes.
                - target_class (str)      : The target class to calculate the probability estimates for.
                - model (object)          : The machine learning model used for prediction -- logistic regression model.
        [Returns]
            DataFrame: A single-column DataFrame containing probability estimates for the specified target class 
            based on the given combination of features (combination of features = current_feature + feature ).
        [Description]
            This function is designed to be parallelized and is used to compute probability estimates 
            for a specified class based on a combination of features. Within each function call, 
            a specific combination of features is tested, and the probability estimates are returned.

            It creates a custom model by zeroing out the coefficients of features not included in the current 
            combination and then uses this model to predict probabilities. The resulting probability estimates 
            are returned as a DataFrame.
    '''
    # Unpacking the inputs
    current_features, feature, input_data, target_class, model = args
    
    # positive and negative class
    classes = dict(zip([0,1],model.classes_))
    positive_class = classes[1]
    negative_class = classes[0]
    
    # DataFrames to contain probability estimates -- to be returned by the end of this function call
    predictionScore_Positive = pd.DataFrame(0.0, index=input_data.index, columns=[feature])
    predictionScore_Negative = pd.DataFrame(0.0, index=input_data.index, columns=[feature])
    
    # These two lines are determining which features to zero out in the baseline
    baselineFeatures_Ref = pd.DataFrame(model.coef_.ravel().copy(), index=input_data.columns, columns=['baseline_coeff'])
    baselineFeatures_Ref.loc[~baselineFeatures_Ref.index.isin([feature] + current_features), :] = 0.0
    
    # Creation of the custom model
    meta_model = deepcopy(model)
    meta_model.coef_ = np.expand_dims(baselineFeatures_Ref['baseline_coeff'].values, axis=0)
    
    # Storing probability estimates in precreated DataFrames -- entire data is used 
    predictionScore_Positive.loc[:, feature] = meta_model.predict_proba(input_data)[:, 1] # Probability estimate for positive class
    predictionScore_Negative.loc[:, feature] = meta_model.predict_proba(input_data)[:, 0] # Probability estimate for negative class
    
    # return the scores depending on the target class
    if target_class == positive_class:
        return predictionScore_Positive
    if target_class == negative_class:
        return predictionScore_Negative
    
    
def initialize_selection(model, ad_data, target_class, \
                         class_labels, cluster_labels, use_layer, n_features=1, ignore_features=None, n_jobs=16):
    '''
        [Summary]
            Identifies a feature/gene that globally maximizes the probability estimate for a 
            specified target class.
        [Parameters]
            model           : The baseline scikit-learn logistic regression model.
            ad_data         : anndata object
            target_class    : The target class for which markers need to be identified.
            class_labels    : Name of the column in ad_data.obs for class labels.
            cluster_labels  : Name of the column in ad_data.obs for cluster labels.
            use_layer       : anndata.layers field denoting which values to use for feature matrix -- imputed data
            ignore_features : List of features to be ignored during ranking. Default is None.
            n_features      : Number of features to select. Deagult is 1.
            n_jobs          : Number of parallel jobs for computation. Default is 16.
        [Return]
            Name of gene/feature with highest probability estimate for the target class.
    '''
    # create the whole data
    train, validation, test, whole = retrieve_IO_data(ad_data, class_labels, use_layer) 
    input_data, output_label = whole
    
    feature_combo_AcrossAllCluster = {}
    numfeatures_AcrossAllCluster = n_features
    
    probEst = {}
    current_features = []

    # model features
    features = pd.DataFrame(model.coef_.ravel().copy(), index=input_data.columns, \
                            columns=['features']).sort_values('features')
    model_features = features.index[features.loc[:,'features'] != 0.0].tolist()
    
    # create an empty list if no features are provided
    if ignore_features is None:
        ignore_features = []

    for elem in range(numfeatures_AcrossAllCluster):
        loop_length = len(model_features)
        current_featuresLIST = loop_length * [current_features]
        feature_LIST = model_features
        input_dataLIST = loop_length * [input_data]
        class_LIST = loop_length * [target_class]
        model_LIST = loop_length * [deepcopy(model)]
        arguments = zip(current_featuresLIST, feature_LIST, input_dataLIST, class_LIST, model_LIST)
        
        # with tqdm(desc='Ranking Genes:', total=loop_length) as progress_bar:
        results = Parallel(n_jobs=n_jobs)(delayed(determine_combination)(*items) for items in arguments)
            
        predictionScore = pd.concat(results, axis=1)
        predictionScore.loc[:, ['cluster', 'class_labels']] = ad_data.obs.loc[predictionScore.index, [cluster_labels, class_labels]].values
        
        probEst[f'{elem + 1}_feature'] = predictionScore.loc[:, ~predictionScore.columns.isin(ignore_features + current_features)]

        select_feature = predictionScore.loc[predictionScore['class_labels'] == target_class, \
                                             ~predictionScore.columns.isin(ignore_features + ['cluster', 'class_labels'] + current_features)].mean().idxmax()
        feature_combo_AcrossAllCluster[f'{elem + 1}_feature'] = current_features + [select_feature]

        if select_feature not in current_features:
            current_features.append(select_feature)
    
    return feature_combo_AcrossAllCluster, probEst

# feature ranking function
def get_rankings(combinations, probEst, ad_data, target_class, model,
                 class_labels, cluster_labels, use_layer, n_features, ignore_features=None, n_jobs=16):
    '''
        [Summary]
            Ranks features by iteratively selecting features based on evaluation scores.
        [Parameters]
            combinations (dict)              : A dictionary containing feature combinations.
            probEst (dict)                   : A dictionary of probability estimates.
            ad_data (AnnData)                : An AnnData object containing observation metadata.
            target_class (str)               : Target class for feature ranking.
            n_features (int)                 : Number of features to be selected for the specified target class.
            model (object)                   : The trained scikit-learn logistic regression model.
            class_labels (str)               : Name of the column in ad_data.obs for class labels.
            cluster_labels (str)             : Name of the column in ad_data.obs for cluster labels.
            use_layer (str)                  : Layer or key-identifier used in the analysis.
            ignore_features (list, optional) : List of features to be ignored during ranking. Default is None.
            n_jobs (int, optional)           : Number of parallel jobs for computation. Default is 16.
        [Returns]
            tuple: A tuple containing:
                - all_model_featureCombination (dict)    : A dictionary containing selected feature combinations for each iteration.
                - predictionScore_Accum (dict)           : A dictionary of probability estimates for each iteration/addition of genes.
                - lowestPerformance_ClusterRecord (dict) : A dictionary mapping the combination features to clusters with the lowest 
                                                           performance throughout the ranking analysis, a record detailing why a specific gene
                                                           was selected.
                - target_class (str)                     : The target class for which the ranking was performed for.
                - n_features (int)                       : The number of features that was selected for the given target class.
    ''' 
    # starting feature -- selected from initialization step of the anlayses
    starting_feature = combinations['1_feature'][0]
    class_mask_target = probEst['1_feature']['class_labels'] == target_class
    target_class_mask = ad_data.obs[class_labels] == target_class
    improve_cluster = probEst['1_feature'].loc[class_mask_target & target_class_mask, :].groupby('cluster').mean()[starting_feature].idxmin()
    
    current_featureCombo = [starting_feature]
    lowestPerformance_Cluster = improve_cluster
    lowestPerformance_ClusterRecord = {starting_feature: improve_cluster}
    predictionScore_Accum = {'1_feature': probEst['1_feature']}
    all_model_featureCombination = {'1_feature': [starting_feature]}
    
    # create the whole data, retrieve_IO_data() returns test, validation, test, whole data
    # based on ad_data.obs['datasplits'] labels created from calling split_data()
    train, validation, test, whole = retrieve_IO_data(ad_data, class_labels, use_layer) 
    input_data, output_label = whole
    
    # data frame contaiing the model features
    features = pd.DataFrame(model.coef_.ravel().copy(), index=input_data.columns, \
                            columns=['features']).sort_values('features')
    model_features = features.index[features.loc[:,'features'] != 0.0].tolist()
    
    # create an empty list if not features 
    if ignore_features is None:
        ignore_features = []
        
    # before proceeding check if there are sufficient enough features that are non-zero
    # if current number of features in model < target_feature, raise an error
    if n_features >= abs(len(ignore_features) - len(model_features)):
        print(f'Number of features to select: {n_features}, Number of features available: {len(model_features)}')
        # print(f'Number of features that will be selected instead: {abs(len(ignore_features) - len(model_features))}')
        n_features = abs(len(ignore_features) - len(model_features))
        print(f'Number of features to be selected instead: {n_features} from the available {len(model_features)} features in the current model, ' 
              f'ignoring: {ignore_features}')
    else:
        print(f'Number of features to select: {n_features} from the available {len(model_features)} features in the current model. ' +
              (f'ignoring: {str(ignore_features)}' if len(ignore_features) > 0 else ''))

    for n_feature in range(1, n_features):
        loop_length = len(model_features)
        current_featuresLIST = loop_length * [current_featureCombo]
        feature_LIST = model_features
        input_dataLIST = loop_length * [input_data]
        class_LIST = loop_length * [target_class]
        model_LIST = loop_length * [deepcopy(model)] 
        arguments = zip(current_featuresLIST, feature_LIST, input_dataLIST, class_LIST, model_LIST)
        
        # with tqdm(total=loop_length, desc=f'[{target_class}] Current_number of features--{len(current_featureCombo) + 1}') as progress_bar:
        results_new = Parallel(n_jobs=n_jobs)(delayed(determine_combination)(*items) for items in arguments)
            
        predictionScore = pd.concat(results_new, axis=1)
        predictionScore.loc[:, ['cluster', 'class_labels']] = ad_data.obs.loc[predictionScore.index, [cluster_labels, class_labels]].values
        
        avgModelPerf_PerCluster = predictionScore.loc[
            predictionScore['class_labels'] == target_class,
            ~predictionScore.columns.isin(current_featureCombo + ignore_features + ['class_labels'])
        ].groupby('cluster').mean()
        
        feature_toadd = avgModelPerf_PerCluster.loc[lowestPerformance_Cluster, :].idxmax()
        all_model_featureCombination[f'{n_feature + 1}_feature'] = current_featureCombo + [feature_toadd]
        current_featureCombo.append(feature_toadd)
        
        lowestPerformance_Cluster = avgModelPerf_PerCluster.loc[:, feature_toadd].idxmin()
        lowestPerformance_ClusterRecord[feature_toadd] = lowestPerformance_Cluster
        
        predictionScore_Accum[f'{n_feature + 1}_feature'] = predictionScore.copy()
    
    return all_model_featureCombination, predictionScore_Accum, lowestPerformance_ClusterRecord, target_class, n_features
    
# function combining everything together
def rank_genes(ad_data, model, target_class, class_labels, cluster_labels, use_layer, n_features=10, ranking_method='method_1', **kwargs):
    '''
        [Summary]
            This function identifies markers for a specified target class based on the learned features of a provided logistic regression
            model. This function modifies the input Anndata object provided by adding the ranking results in `anndata.uns` as
            `anndata.uns['<target_class>_marker_results']`.
        [Parameters]
            ad_data (AnnData)    : The input AnnData object containing the subset of data.
            model (object)       : A machine learning model (scikit-learn classifier) used for feature ranking.
            target_class (str)   : The target class for which markers need to be identified.
            class_labels (str)   : Column name in `ad_data.obs` containing class labels.
            cluster_labels (str) : Column name in `ad_data.obs` containing cluster labels.
            use_layer (str)      : Layer name or identifier used in the analysis.
            n_features (int)     : The number of features to be selected for the specified target class. Default is 10.
            ranking_method (str) : The ranking method to use.
                                       - `method_1` initially selects a feature that maximizes probability estimates
                                          across clusters belonging to the `target_class` and subsequent selected features are selected to maximize
                                          probability estimates on a cluster-by-cluster basis.
                                       - `method_2` selects feature that maximizes probability estimates across clusters belonging to the 
                                         `target_class`.
            **kwargs (dict) : Additional keyword arguments for feature selection (passed into `intialized_selection()` and `get_rankings()`).
                - ignore_features (list or None, optional) : A list of feature names to be ignored during selection. Default is None.
                - n_jobs (int)                             : The number of CPU cores to be used for parallel processing for computation. 
                                                             Default is None (using all available cores).
        [Description]
            This function performs two main steps:
            1. Feature Initialization: Selects genes/features that globally maximize classification performance
               of the `target_class` from non-`target_class`.
            2. Feature Ranking: Selects genes/features in combination with current features that maximize classification
               performance on clusters that the current combination poorly classifies as `target_class` from non-`target_class`.
        [Usage]
            Example usage:
            >>> rank_genes(ad_data, model, 'marker_class', 'class_labels', 'cluster_labels', 'layer_name', n_features=10, ignore_features=['gene1', 'gene2'], n_jobs=4)
        [Note]
            The function modifies the input AnnData object `ad_data` by adding marker results to its `uns` attribute, denoted as:  
            `ad_data.uns[f'{target_class}_marker_results']`.

            The stored results have the following structure:
            {
                'target_class': target_class,
                'initial_feature': all_combos['1_feature'],
                f'top_{n_features}': all_combos[f'{n_features_used}_feature'],
                'all_combinations': all_combos,
                'all_combinations_probEst': pred_accum,
                'sequentially_added_features': records,
            }
    '''
    if ranking_method == 'method_1':
        # initialization step -- select genes/features that, globally, maximizes classification performance 
        # the of target_class from non target_class
        combinations, probEst = initialize_selection(
            ad_data=ad_data,
            model=deepcopy(model),
            target_class=target_class, 
            cluster_labels=cluster_labels,
            class_labels=class_labels, 
            use_layer=use_layer,
            **kwargs
        )

        # feature ranking step -- select genes/features in combination with current features that maximizes classification
        # performance on clusters that the current combination poorly classifies as the target_class from non target_class
        all_combos, pred_accum, records, target_class, n_features_used = get_rankings(
            combinations=combinations, 
            probEst=probEst, 
            ad_data=ad_data, 
            target_class=target_class, 
            n_features=n_features,
            use_layer=use_layer,
            model=deepcopy(model),
            class_labels=class_labels, 
            cluster_labels=cluster_labels,
            **kwargs
        )

        # store results
        ad_data.uns[f'{target_class}_marker_results'] = {
            'target_class': target_class,
            'initial_feature':all_combos['1_feature'],
            f'top_{n_features}': all_combos[f'{n_features_used}_feature'],
            'all_combinations': all_combos,
            'all_combinations_probEst': pred_accum,
            'sequentially_added_features': records,
            'ranking_method':ranking_method
        }
    
    # TO DO: implement a variation of the feature selection where we instead select features based on global performance
    # instead of cluster by cluster
    if ranking_method == 'method_2':
        # feature ranking step -- selects genes that globaly maximizes the probability estimates across clusters belonging
        # to target class
        combinations, probEst = initialize_selection(
            ad_data=ad_data,
            model=deepcopy(model),
            target_class=target_class, 
            cluster_labels=cluster_labels,
            class_labels=class_labels, 
            n_features=n_features,
            use_layer=use_layer,
            **kwargs
        )
        
        # store results
        ad_data.uns[f'{target_class}_marker_results'] = {
            'target_class': target_class,
            'initial_feature':combinations['1_feature'],
            f'top_{n_features}': combinations[f'{n_features}_feature'],
            'all_combinations': combinations,
            # 'all_combinations_probEst': pred_accum,  # these can be calculated by another function -- given the combinations
            # 'sequentially_added_features': records,  # these is not needed
            'ranking_method':ranking_method
        }

# function to compute the probability estimate for a class
# if features=None use all the genes in the model to compute the probability estiamte
# this function modifies the passed the anndata and adds probability estimate predictions in anndata.obsm attribute
def get_ProbabilityEstimates(ad_data, model, model_name, target_class, 
                       class_labels, cluster_labels, use_layer, target_features=None):
    '''
        [Summary]
            Complementary function for obtaining probability estimates of a trained model for a
            target class. If target_features are specified, this function will zero out every feature
            in the model except for the specified target_features.
        [Parameters]
            ad_data (AnnData)                : An AnnData object.
            model (object)                   : A trained machine learning model.
            model_name (str)                 : The name of the model.
            target_class (str)               : The target class for which to compute the probability estimates.
            class_labels (str)               : Name of the column in ad_data.obs for class labels.
            cluster_labels (str)             : Name of the column in ad_data.obs for cluster labels.
            use_layer (str)                  : Anndata.layers field denoting which values to use for the feature matrix. 
                                               Imputed data is preferred.
            target_features (list, optional) : A specified set of genes to compute the probability estimates for. 
                                               If provided, all other features in the model will be zeroed out. Default is None.
        [Returns]
            DataFrame: A DataFrame containing the probability estimates for a given target class, for each cluster
            belonging to that class.
        [Description]
            This function is a complementary utility for obtaining probability estimates of a trained model
            for a specific target class within an AnnData object. It computes probability estimates based on the
            input model and AnnData object. If `target_features` are specified, only those features will be used
            for the probability estimation, and all other features will be zeroed out in the model.

            The resulting DataFrame contains probability estimates for the target class, broken down by clusters
            belonging to the specified target_class. The probabilities are computed using the provided model and input data.
    '''
    # make a copy of the model so that modifications are not inplace
    test_model = deepcopy(model)

    # create the input and output label from the anndata
    data_input, data_output = retrieve_IO_data(ad_data, \
                                               class_label=class_labels, use_layer=use_layer, split=False)
    
    # create a dataframe to store predictions
    predictionScore = pd.DataFrame(0.0, index=data_input.index, columns=[model_name])
    
    # create a reference of the model's coefficient
    ref_baselineFeatures = pd.DataFrame(test_model.coef_.ravel().copy(), \
                                        index=data_input.columns, columns=['baseline_coeff'])
    
    # zero everything except for specified features if target features are specified
    if target_features is not None:
        ref_baselineFeatures.loc[~ref_baselineFeatures.index.isin(target_features),:] = 0.0
        test_model.coef_ = np.expand_dims(ref_baselineFeatures['baseline_coeff'].values, axis=0)
    
    # identify what to return based on the target class -- positive and negative class
    classes = dict(zip([0,1],test_model.classes_))
    positive_class = classes[1]
    negative_class = classes[0]
    
    # return the scores depending on the target class
    if target_class == positive_class:
        predictionScore.loc[:,model_name] = test_model.predict_proba(data_input)[:,1] # socket

    if target_class == negative_class:
        predictionScore.loc[:,model_name] = test_model.predict_proba(data_input)[:,0] # sheath

    # compute the mean scores for the target class for each cluster
    predictionScore.loc[:,['cluster','class_labels']] = ad_data.obs.loc[data_input.index,[cluster_labels, class_labels]].values.copy()
    
    # compute the mean probability estimates for the target_class for each clusters -- select the target_class
    predictionScore = predictionScore.loc[predictionScore['class_labels'] == target_class,:].drop(columns='class_labels')
    predictionScore = predictionScore.groupby('cluster').mean()
    
    return predictionScore

# package the following function above as a fdunction 
def calculate_ProbEst_Summary(ad_data, model, class_labels, cluster_labels, use_layer, 
                        target_class, feature_combos, return_results=False, model_name='baseline'):
    '''
        [Summary]
            Calculates the probability estimates of each cluster for a given target class 
            across different feature/gene combinations learned by a binary classifier (trained sci-kit learn
            logistic regression model).
        [Parameters]
            ad_data (AnnData)               : An AnnData object containing the data.
            model (object)                  : A trained machine learning model.
            class_labels (str)              : Name of the column in ad_data.obs for class labels.
            cluster_labels (str)            : Name of the column in ad_data.obs for cluster labels.
            use_layer (str)                 : Anndata.layers field denoting which values to use for the feature matrix.
            target_class (str)              : The target class for which to compute probability estimates.
            feature_combos (dict)           : A dictionary containing feature/gene combinations to evaluate, 
                                              where keys are names of combinations and values are lists of feature names.
            return_results (bool, optional) : If True, the function returns the accumulated probability estimates.
                                              Default is False.
            model_name (str, optional)      : The name to use for the model in the results. Default is 'baseline'.
        [Returns]
            if `return_results` is True, it returns the following:
                DataFrame: A dataframe containing probability estimates for each cluster
                for each different feature/gene combination present in feature_combos.
        [Description]
            This function calculates the probability estimates for a specified target class 
            across various feature/gene combinations. It iterates through each combination 
            in feature_combos and computes probability estimates for each cluster. The results 
            are stored in an AnnData object and optionally returned as a DataFrame.

            The accumulated probability estimates provide insights into the impact of different 
            feature combinations on cluster-specific predictions for the target class compared to baseline model.
        [Note]
            The function modifies the input AnnData object `ad_data` by updating the `anndata.uns` with the following
            new field ad_data.uns[f'{target_class}_probEst_Summary']
    '''  
    # compute the probability estimates per cluster for each of the combinations in 
    # feature combos
    accumulated_scores = []
    for name, features in feature_combos.items():
        score = get_ProbabilityEstimates(
            ad_data=ad_data, 
            model=model,
            model_name=name,
            target_class=target_class, 
            class_labels=class_labels,
            cluster_labels=cluster_labels, 
            use_layer=use_layer,
            target_features=features
        )
        score = score.stack().to_frame().reset_index().rename(columns={'level_1':'names',0:'scores'})
        accumulated_scores.append(score)
    
    # compute the probability estimates per cluster using all the features in the model -- this is for
    # the full set of features
    score_all_features = get_ProbabilityEstimates(
            ad_data=ad_data, 
            model=model,
            model_name=model_name,
            target_class=target_class, 
            class_labels=class_labels,
            cluster_labels=cluster_labels, 
            use_layer=use_layer,
            target_features=None
    )
    score_all_features = score_all_features.stack().to_frame().reset_index().rename(columns={'level_1':'names',0:'scores'})
    accumulated_scores.append(score_all_features)
        
    accumulated_scores = pd.concat(accumulated_scores)
    
    # add the scores to the anndata
    ad_data.uns[f'{target_class}_probEst_Summary'] = accumulated_scores.copy()
    
    if return_results:
        return accumulated_scores
    
# this should also modify the anndata inplace -- implement save plot 
def view_ProbEst_Summary(ad_data, model, feature_combos, target_class, save_plot=False, show_grid=False, **kwargs):
    '''
        [Summary]
            Complementary function for feature ranking analysis. Displays probability estimates per cluster 
            as features/genes are sequentially added during the feature ranking analysis.
        [Parameters]
            ad_data (AnnData)          : An AnnData object containing the data used to create the feature matrix and target vector
                                         for computing probability estimates.
            model (object)             : A trained machine learning model (e.g., scikit-learn classifier) used for feature ranking,
                                         typically the trained baseline logistic regression model.
            feature_combos (dict)      : Combinations of genes to be tested.
            target_class (str)         : The target class for which to compute the probability estimates summary.
            save_plot (bool, optional) : If True, saves the plot. Default is False.
            show_grid (bool, optional) : Determines whether to show gridlines on the plot. Default is False.
            
            **kwargs : Additional keyword arguments (passed in `calculate_ProbEst_Summary()`).
                - class_labels (str)    : Name of the column in ad_data.obs for class labels.
                - cluster_labels (str)  : Name of the column in ad_data.obs for cluster labels.
                - use_layer (str)       : `anndata.layers` field denoting which values to use for the feature matrix.
                - model_name (str)      : The name to use for the model in the results. Default is 'baseline'.
                - return_results (bool) : If True, the function returns the accumulated probability estimates.
                                          Default is False.
        [Description]
            This function is used in conjunction with feature ranking analysis. It calculates and displays
            probability estimates per cluster as features or genes are sequentially added during the ranking analysis.
            The probability estimates are based on the provided machine learning model and AnnData object.

            The function creates a stripplot that shows the average probability estimates for different feature combinations
            in feature_combos, with clusters colored for differentiation. In general, this visualization helps analyze how the
            combination of features affects probability estimates for the target class across clusters that belong to the target
            class.

            If `save_plot` is set to True, the plot will be saved. The `show_grid` parameter controls whether gridlines
            should be displayed on the plot.
        [Note]
            The function modifies the input AnnData object `ad_data` by updating the `anndata.uns` with the following
            new field `ad_data.uns[f'{target_class}_probEst_Summary']`.
    '''
    # calculate the probability estimates
    calculate_ProbEst_Summary(
        ad_data=ad_data,
        model=model,
        target_class=target_class, 
        feature_combos=feature_combos,
        **kwargs
    )
    
    probEst = ad_data.uns[f'{target_class}_probEst_Summary']
    
    # Get color palettes
    if 'Custom_Cluster_Colors' in ad_data.uns.keys():
        color_map = ad_data.uns['Custom_Cluster_Colors']
    else:
        color_map = None
    
    # Plot stripplot
    fig_strip = px.strip(probEst, x='names', y='scores', hover_name='cluster', \
                         color='cluster', stripmode='group', color_discrete_map=color_map)

    fig_strip.update_layout(

        height=700,
        width=1250,
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title='Average<br>Probability Estimates'),
        xaxis=dict(title='Sequentially Selected Features<br>(Probability Estimates using 1-10 genes)'),
        margin=dict(l=100, r=100, t=100, b=200), 
        title=dict(text=f'<b>{target_class}</b> Probability Estimates Per Cluster |'
        f' <b><i>{ad_data.obs.sex.unique().categories.values.item()}</i> Data</b>')
    )
    
    fig_strip.update_traces(
        marker=dict(opacity=0.6, size=10),
        marker_symbol='diamond'
    )
    fig_strip.update_xaxes(showgrid=show_grid, tickangle=90, tickfont=dict(family='Rockwell', color='gray', \
                                                       size=14), showline=show_grid, gridcolor='lightgray')
    fig_strip.update_yaxes(showgrid=show_grid, tickfont=dict(family='Rockwell', color='gray', \
                                         size=14), range=[0,1.1], showline=show_grid, gridcolor='lightgray')
    fig_strip.show()

    
def get_AUROCC(ad_data, model, target_class, class_labels,
              cluster_labels, use_layer, target_features=None):
    '''
        [Summary]
            Computes AUROCC for a given cluster belonging to a specified target class.
        [Parameters]
            ad_data (AnnData)                : An AnnData object containing the data.
            model (object)                   : The trained baseline scikit-learn logistic regression model.
            target_class (str)               : One of the target classes for which to compute the ROC scores.
            class_labels (str)               : Name of the column in ad_data.obs for class labels.
            cluster_labels (str)             : Name of the column in ad_data.obs for cluster labels.
            use_layer (str)                  : `anndata.layers` field denoting which values to use for feature matrix. Imputed data is preferred.
            target_features (list, optional) : A list of specified features to use for computing AUROCC. Default is None.
        [Return]
            Returns a dataframe containing the AUROCC scores for each cluster belonging to a target class.
        [Description]
            This function computes the Area Under the Receiver Operating Characteristic Curve (AUROCC) for each cluster
            belonging to a specified target class. It evaluates the model's prediction performance within individual clusters
            based on the provided machine learning model and AnnData object.
        [Note]
            The function modifies the input AnnData object `ad_data` by updating the `anndata.uns` with the following
            new field: ad_data.uns[f'{target_class}_AUROCC_Summary'] containing a dataframe of AUROCC scores for a specific
            class for each clusters.
    '''
    
    # create the input and output data -- remember, excluded cells will not be included
    data_input, data_output = retrieve_IO_data(ad_data, \
                                              class_label=class_labels, use_layer=use_layer, split=False)
    
    # prep the data  
    data_input.loc[:,'cluster'] = ad_data.obs.loc[data_input.index,cluster_labels].values.copy()
    data_output.loc[:,'cluster'] = ad_data.obs.loc[data_output.index,cluster_labels].values.copy()
    group_input = data_input.groupby('cluster')
    
    # make a copy of the model 
    test_model = deepcopy(model)
    
    # create a reference of the model's coefficients
    ref_baselineFeatures = pd.DataFrame(test_model.coef_.ravel().copy(), \
                                        index=data_input.columns[:-1], columns=['baseline_coeff'])
    
    # zero everything except for specified features if target features are specified
    if target_features is not None:
        if not isinstance(target_features, list):
            raise Exception('\'target_features\' argument must be a list')
        ref_baselineFeatures.loc[~ref_baselineFeatures.index.isin(target_features),:] = 0.0
        test_model.coef_ = np.expand_dims(ref_baselineFeatures['baseline_coeff'].values, axis=0)
    if target_features is None:
        print('Baseline model will be used -- all features will be used')
    
    # construct dataframe to store scores
    df_roc_auc = pd.DataFrame(0.0, index=group_input.groups.keys(), 
                              columns=['baseline',f'feature_subset'])
    
    # identify what to return based on the target class -- positive and negative class
    classes = dict(zip([0,1],test_model.classes_))
    positive_class = classes[1]
    negative_class = classes[0]
    
    # get groups for the positive class
    negative_clusters = ad_data.obs.loc[(ad_data.obs[class_labels] == negative_class) & 
                                        (ad_data.obs_names.isin(data_input.index)),
                                        cluster_labels].unique().tolist()
    
    # get groups for the negative class
    positive_clusters = ad_data.obs.loc[(ad_data.obs[class_labels] == positive_class) &  
                                        (ad_data.obs_names.isin(data_input.index)),
                                        cluster_labels].unique().tolist()    
    
    # presets -- parameters that determines which class to compute the roc auc scores for
    # and to return
    if target_class == positive_class:
        predict_proba_col = 1
        target_groups, other_group = positive_clusters, negative_clusters

    if target_class == negative_class:
        predict_proba_col = 0
        target_groups, other_group = negative_clusters, positive_clusters
    
    # actually compute scores
    for target in target_groups:
        # target_cluster + the rest of the non target cluster, input & output pairs
        # .iloc[:,:-1] is added to exclude the ' cluster' column
        meta_input = data_input.loc[(data_input['cluster'].isin(other_group)) | 
                                    (data_input['cluster'] == target),:].iloc[:,:-1]
        meta_output = data_output.loc[(data_output['cluster'].isin(other_group)) | 
                                      (data_output['cluster'] == target),:].iloc[:,:-1]
        
        # compute the score -- uses baseline/complete features
        auroc_baseline = roc_auc_score(
            y_true=meta_output,
            y_score=model.predict_proba(meta_input)[:,predict_proba_col]
        )
        df_roc_auc.loc[target, df_roc_auc.columns[0]] = auroc_baseline
        
        # compute the score for the subset of features
        auroc_subset = roc_auc_score(
            y_true=meta_output,
            y_score=test_model.predict_proba(meta_input)[:,predict_proba_col]
        )
        df_roc_auc.loc[target, df_roc_auc.columns[1]] = auroc_subset
    
    # return the corresponding results depending on the target class
    # consider adding this onto the anndata object
    if target_class == positive_class:
        df_roc_auc = df_roc_auc.loc[df_roc_auc.index.isin(positive_clusters),:]
        df_roc_auc = df_roc_auc.unstack().reset_index() \
            .rename(columns={'level_0':'model_label','level_1':'cluster', 0:'AUROCC_Scores'})
        ad_data.uns[f'{target_class}_AUROCC_Summary'] = df_roc_auc
        return df_roc_auc
    if target_class == negative_class:
        df_roc_auc = 1 - df_roc_auc.loc[df_roc_auc.index.isin(negative_clusters),:]
        df_roc_auc = df_roc_auc.unstack().reset_index() \
            .rename(columns={'level_0':'model_label','level_1':'cluster', 0:'AUROCC_Scores'})
        ad_data.uns[f'{target_class}_AUROCC_Summary'] = df_roc_auc
        return df_roc_auc 
    
# define the following function 
def view_AUROCC_Summary(ad_data, model, target_class, class_labels, 
                       cluster_labels, use_layer, target_features=None, show_grid=False):
    '''
        [Summary]
            Plots the AUROCC per cluster belonging to the specified target_class.
        [Parameters]
            ad_data         : anndata object 
            model           : trained baseline scikit-learn logistic regression model.
            target_class    : one of the target class for which to compute the AUROCC for
            class_labels    : Name of the column in ad_data.obs for class labels.
            cluster_labels  : Name of the column in ad_data.obs for cluster labels.
            use_layer       : anndata.layers field denoting which values to use for feature matrix 
            target_features : set of specified genes that is also present in the model 
            show_grid       : 
        [Returns]
    '''
    # compute the scores for the individual clusters
    rocauc_scores = get_AUROCC(
        ad_data=ad_data,
        model=model, 
        target_class=target_class, 
        class_labels=class_labels, 
        cluster_labels=cluster_labels, 
        use_layer=use_layer,
        target_features=target_features 
    )
    
    # get/apply color palettes/colormap
    if 'Custom_Cluster_Colors' in ad_data.uns.keys():
        color_map = ad_data.uns['Custom_Cluster_Colors']
    else:
        color_map = None
    
    fig_strip = px.strip(rocauc_scores, x='model_label',y='AUROCC_Scores', hover_name='cluster', \
                         color='cluster', stripmode='group', color_discrete_map=color_map)

    fig_strip.update_layout(
        height=800,
        width=600,
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(title='AUROCC<br>Scores'),
        xaxis=dict(title=None),
        margin=dict(l=100, r=100, t=100, b=200),
        title=dict(text=f'<b>{target_class}</b> AUROCC Scores Per Cluster |'
        f' <b><i>{ad_data.obs.sex.unique().categories.values.item()}</i> Data</b>')
    )
    fig_strip.update_traces(
        marker=dict(opacity=0.6, size=10),
        marker_symbol='diamond'
    )
    fig_strip.update_xaxes(showgrid=show_grid, tickangle=90, tickfont=dict(family='Rockwell', color='gray', \
                                                       size=14), showline=show_grid, gridcolor='lightgray')
    fig_strip.update_yaxes(showgrid=show_grid, tickfont=dict(family='Rockwell', color='gray', \
                                         size=14), range=[0,1.1], showline=show_grid, gridcolor='lightgray')
    fig_strip.show()
    

    
    
    
    
    
    
    