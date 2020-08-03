import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

import warnings
warnings.simplefilter('ignore')

def preprocess_train_data(file_name):

    '''
    Preprocesses titanic_train.csv.

    Args:
        file_name: Filename (string)
    Returns:
        Three pandas.dataframe objects:
        df: Overall DataFrame containing the whole Titanic Dataset.
        X : Input Features in Titanic Dataset.
        Y : Target Feature in Titanic Dataset.
    '''
    
    df = pd.read_csv(file_name)[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]
    df[['Male', 'Q', 'S']] = pd.get_dummies(df[['Sex', 'Embarked']], drop_first=True)
    df = df.drop(columns=['Sex', 'Embarked'])
    df = df.dropna()
    X = df.drop(columns='Survived', axis=1)
    Y = pd.get_dummies(df['Survived'])
    return df, X, Y

def preprocess_test_data(file_name):

    '''
    Preprocesses titanic_test.csv.

    Args:
        file_name: Filename (string)
    Returns:
        pandas.dataframe object df:
        df: Overall DataFrame containing the whole Titanic Dataset.
    '''
    
    df = pd.read_csv(file_name)[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    df[['Male', 'Q', 'S']] = pd.get_dummies(df[['Sex', 'Embarked']], drop_first=True)
    df = df.drop(columns=['Sex', 'Embarked'])
    df = df.dropna()
    return df

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
    means = []
    stds = []
    for feature in features:
        mean = df[feature].mean()
        std = df[feature].std()
        df[feature] = (df[feature] - mean) / std
        means.append(mean)
        stds.append(std)
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
    for i, feature in enumerate(features):
        mean = means[i]
        std = stds[i]
        df[feature] = (df[feature] - mean) / std
    return df
    
def plot_confusion_matrix(y_true, y_pred, title):

    '''
    Plots a heatmap from Confusion Matrix

    Args:
        y_true: numpy.array containing the target values.
        y_pred: numpy.array of predicted values.
    Returns:
        seaborn.heatmap object.
    '''
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, lw=3, linecolor='black', cmap='Blues', fmt='g')
    plt.title(title + ' Data')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
def plot_ROC_curve(X_tr, Y_tr, X_te, Y_te, model):

    '''
    Plots the ROC curve and AUC score for a given set of inputs

    Args:
        X_tr : pandas.DataFrame object containing Train Features.
        Y_tr : pandas.DataFrame object containing Train Labels.
        X_te : pandas.DataFrame object containing Test Features.
        Y_te : pandas.DataFrame object containing Test Labels.
        model: model object used as decision_function.
    Returns:
        matplotlib.pyplot.plot object
    '''
    
    fpr_tr, tpr_tr, thresholds_tr = roc_curve(Y_tr.values[:, 1], model.predict_proba(X_tr.values)[:, 1])
    fpr_te, tpr_te, thresholds_te = roc_curve(Y_te.values[:, 1], model.predict_proba(X_te.values)[:, 1])
    auc_tr = roc_auc_score(Y_tr.values[:, 1], model.predict_proba(X_tr.values)[:, 1])
    auc_te = roc_auc_score(Y_te.values[:, 1], model.predict_proba(X_te.values)[:, 1])
    plt.plot(fpr_tr, tpr_tr, 'b', label='Train AUC = {}'.format(round(auc_tr, 4)))
    plt.plot(fpr_te, tpr_te, 'r', label='Test AUC = {}'.format(round(auc_te, 4)))
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid()
    plt.show()

def print_classification_report(y_tr, y_te, y_pred_tr, y_pred_te):
    
    '''
    Plots a heatmap from Confusion Matrix

    Args:
        y_tr      : numpy.array containing the target values used for training.
        y_te      : numpy.array containing the target values used for testing.
        y_pred_tr : numpy.array of predicted values from train data.
        y_pred_te : numpy.array of predicted values from test data.
    Returns:
        Prints Classification Report from sklearn.metrics.classification_report
    '''
    
    print('\n' + 22*'*'+ ' Training ' + 22*'*'+'\n')
    print(classification_report(y_tr, y_pred_tr))
    print('\n' + 22*'*'+ ' Testing ' + 22*'*'+'\n')
    print(classification_report(y_te, y_pred_te))
