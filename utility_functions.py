import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')

def preprocess_data_mnist(file_name):

    '''
    Preprocesses mnist_train.csv.

    Args:
        file_name: Filename (string)
    Returns:
        Three pandas.dataframe objects:
        df: Overall DataFrame containing the whole MNIST Train Dataset.
        X : Input Features in MNIST Dataset.
        Y : Target Feature in MNIST Dataset.
    '''
    
    df = pd.read_csv(file_name)
    X = df.drop(columns='label', axis=1)
    X = X.fillna(0)
    Y = pd.get_dummies(df['label'])
    return df, X, Y


def normalize_train_data(df):

    '''
    Performs mean centering and variance scaling on a dataframe used for training.

    Args:
        pandas.DataFrame object
    Returns:
        df   : pandas.DataFrame object with columns mean centered and variance scaled.
        means: List of mean values of samples used in training.
        std  :  List of std. deviations of samples used in training.
    
    '''
    
    features = df.columns
    epsilon = 1e-8
    means, stds = [], []
    for feature in features:
        mean, std = df[feature].mean(), df[feature].std()
        df[feature] = (df[feature] - mean) / (std + epsilon)
        means.append(mean); stds.append(std)
    return df, means, stds

def normalize_test_data(df, means, stds):

    '''
    Performs mean centering and variance scaling on a dataframe used for testing.

    Args:
        df   : pandas.DataFrame object
        means: List of mean values of samples used in training.
        std  : List of std. deviations of samples used in training.
    Returns:
        df: pandas.DataFrame object with columns mean centered and variance scaled.
    '''
    
    features = df.columns
    epsilon = 1e-8
    for i, feature in enumerate(features):
        mean, std = means[i], stds[i]
        df[feature] = (df[feature] - mean) / (std + epsilon)
    return df
    

def accuracy(target, output):

    '''
    Computes accuracy

    Args:
        target: numpy.array containing the target values.
        output: numpy.array of predicted values.
    Returns:
        Accuracy score
    '''
    misclassified = np.array([0 if (i==j).all() else 1 for i,j in zip(target,output)])
    return 1 - np.sum(misclassified) / target.shape[0]

