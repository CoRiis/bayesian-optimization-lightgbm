import os
import argparse
import time
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.utils import split_dataset


def load_data_set():
    """ Loading the Instagram data set. Returns a Panda DataFrame"""
    return pd.read_parquet("dataset.parquet")


def train_gbm(df, params, feature_groups="[author,content,temporal]", verbose=False):
    """
    Training a GBM and returning the Spearman Ranking Correlation (SRC)
    :param df: pandas data frame
    :param params: parameters in a dict for LightGBM
    :param feature_groups: ['list', 'of', 'features', 'to', 'use'] w/o engagement signal
    :param verbose: selfexplainatory
    :return: SRC
    """
    
    features = []
    cat_features = []
    possible_features = ['author', 'content', 'temporal', 'yolo', 'efficientnet', 'places']
    if 'author' in feature_groups:
        features += ['logFollowers', 'logFollowing', 'logPosts', 'followersStatusRatio', 'followersFriendsRatio']
    if 'content' in feature_groups:
        features += ['type', 'filter', 'lang', 'tagCount', 'usersTagged', 'carouselSize', 'isEnglish', 'userHasLiked']
        cat_features += ['type', 'filter', 'lang']
    if 'temporal' in feature_groups:
        features += ['postedHour', 'postedDate', 'postedWeekDay']
        cat_features += ['postedDate', 'postedWeekDay']
    if 'yolo' in feature_groups:
        features += df.columns[df.columns.str.startswith('yolo')].tolist()
    if 'efficientnet' in feature_groups:
        features += df.columns[df.columns.str.startswith('eff')].tolist()
    if 'places' in feature_groups:
        features += df.columns[df.columns.str.startswith('place')].tolist()
    if 'iipa' in feature_groups:
        features += ['pred_iipa']
        
        
    # split dataset into training and test
    X_train, X_test, y_train, y_test = split_dataset(df, features,
                                                     predict=['logLikes'],
                                                     training_samples=None,
                                                     seed=None)

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train, feature_name=features, categorical_feature=cat_features)
    lgb_eval = lgb.Dataset(X_test, y_test, feature_name=features, categorical_feature=cat_features,
                           reference=lgb_train)

    if verbose:
        print("\nData:")
        print("Input features:", X_train.shape[1])
        print("Posts in train data\t %7.d" % X_train.shape[0])
        print("Posts in test  data\t %7.d" % X_test.shape[0])
        print('\nStarting training...')
        
    gbm = lgb.train(params, lgb_train,
                    num_boost_round=1000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=5,
                    verbose_eval=verbose,
                    feature_name=features,
                    categorical_feature=cat_features)

    if verbose:
        print('\nStarting predicting...')
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    # evaluation
    rho, p_value = spearmanr(y_test, y_pred)
    rmse = (mean_squared_error(y_test, y_pred) ** 0.5)
    mae = mean_absolute_error(y_test, y_pred)
    if verbose:
        print('RMSE of prediction is: %.3f' % rmse)
        print('MAE of prediction is: %.3f' % mae)
        print("Spearman Rank Correlation: %.3f (p-value: %.3f)" % (rho, p_value))
        
    return rho