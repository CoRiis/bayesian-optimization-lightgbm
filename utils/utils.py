import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from glob import glob


ENV_TORCH_HOME = 'TORCH_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'


def _get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(ENV_TORCH_HOME, os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch')))
    return torch_home


def load_parquet_files_into_df(path):
    """
    Load parquet files into pandas data frame
    :param path: path/to/folder/with/parquet_files
    :return: data frame
    """
    data_dir = Path(path)
    print('Loading files')
    df = pd.concat(
        [pd.read_parquet(parquet_file) for parquet_file in tqdm(data_dir.glob('*.parquet'))],
        ignore_index=True
    )
    return df


def read_parquet(path, to_read=None, read_frac=1):
    partitions = pd.concat(
        [pd.read_parquet(f, columns=to_read).sample(frac=read_frac) for f in glob(path + '/*.snappy.parquet')],
        ignore_index=True
    )
    return pd.DataFrame(partitions)


def power_transformation(df, feature):
    log_var = np.log1p(df[feature])
    return log_var - np.mean(log_var)


def transform(df):
    """
    Transformation of variables
    :param df:  Pandas data frame
    :return: df w/ transformed variables
    """

    # Impute variables
    df['carouselSize'] = df['carouselSize'].fillna(0).astype('uint8')
    df['usersTagged'] = df['usersTagged'].fillna(0).astype('uint8')
    df['comments'] = df['comments'].fillna(0).astype('uint16')

    # Log transformation and then subtract mean
    df['logLikes'] = power_transformation(df, 'likes')
    df['logComments'] = power_transformation(df, 'comments')
    df['logFollowers'] = power_transformation(df, 'followers')
    df['logFollowing'] = power_transformation(df, 'following')
    df['logPosts'] = power_transformation(df, 'posts')

    # Extract IIPA
    df["pred_iipa"] = [iipa[0] for iipa in df["pred_iipa"].values]

    # Extract hour, date, and week day
    created_time = df["createdTime"].tolist()
    df["postedDate"] = [date.date() for date in created_time]
    df["postedHour"] = [date.hour for date in created_time]
    df["postedWeekDay"] = [date.weekday() for date in created_time]

    # Compute new features
    df['tagCount'] = [len(tags) for tags in df['tags'].fillna(0).values]
    df['isEnglish'] = df['lang'].str.contains('en').astype('uint8')
    df['followersStatusRatio'] = df.followers / (df.posts + 1)
    df['followersFriendsRatio'] = df.followers / (df.following + 1)

    # Convert to intX
    df['userHasLiked'] = df['user_has_liked'].astype('uint8')
    df['tagCount'] = df['tagCount'].astype('uint8')
    df['postedHour'] = df['postedHour'].astype('uint8')
    df['likes'] = df['likes'].astype('uint16')

    # Convert categorical features to index (integers)
    cat = ['type', 'filter', 'lang', 'postedDate', 'postedWeekDay']
    for var in cat:
        df[str(var)] = df[str(var)].astype("category")
        df[str(var)] = df[str(var)].cat.codes.astype('uint8')

    # Drop variables
    drop_var = ['tags', 'source', 'createdTime']
    df = df.drop(columns=drop_var)

    return df


def criteria_df(df, criteria_type=None, verbose=False):
    """
    Remove posts that do not fulfill the criteria
    :param df: data frame
    :return: data frame with remove posts
    """
    cut_off = pd.to_datetime('2018-12-20 00:00:00', format='%Y-%m-%d %H:%M:%S')  # 2018-12-09

    df_crit = None
    if criteria_type is None:
        df_crit = df[(df["createdTime"] < cut_off) &
                     (df["likes"].notnull())]

    if criteria_type == 'gayberi2019':
        df_crit = df[(df["followers"] > 100) &
                     (df["following"] > 100) &
                     (df["posts"] > 50) &
                     (df["createdTime"] < cut_off) &
                     (df["likes"].notnull())]

    if verbose:
        print('\t\tObservations before removing posts \t%7.d' % df.shape[0])
        print('\t\tObservations after removing posts \t%7.d' % df_crit.shape[0])

    return df_crit.reset_index(drop=True)


def split_dataset(df, features, predict, training_samples=None, seed=np.random.randint(0,1e5)):
    """
    Split dataset into training and validation
    :param df: pandas data frame
    :param features: ['list', 'of', 'features'] w/o engagement signal
    :param predict: ['feature'] to predict (engagement signal)
    :param training_samples: number of training samples, if None: 90 %
    :param seed: a random seed, e.g. 42
    :return:
    """
    # Shuffle rows in data frame
    df_shuffled = df.sample(frac=1, random_state=seed)

    # Split into training and test
    offset = int(df_shuffled.shape[0] * 0.9)
    if training_samples is None:
        train = df_shuffled.iloc[:offset, ]
    else:
        train = df_shuffled.iloc[:training_samples, ]
    test = df_shuffled.iloc[offset:, ]
    X_train = train[features]
    X_test = test[features]
    y_train = train[predict]
    y_test = test[predict]

    return X_train, X_test, y_train, y_test
