import pandas as pd 
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

from preprocessing import split_data
from acquire import acquire_data
from prepare import prepare_data

import warnings
warnings.filterwarnings('ignore')

def build_model(train, feature_clustered, k):   
    # create cluster object, train and fit
    X = train[feature_clustered]
    kmeans = KMeans(k)
    kmeans.fit(X)

    #encode cluster into different colomns
    train['cluster'] = kmeans.predict(train[feature_clustered])
    cluster_df = pd.get_dummies(train.cluster)
    # concatenate the dataframe with the  one hot encoded cluster  to the original dataframe
    train= pd.concat([train, cluster_df], axis =1)
    # drop cluster columns
    train = train.drop(columns = ['cluster'])

    #
    X =train.drop(columns = ['regionidcity', 'regionidzip', 'actual_value', 'logerror','estimated'])\
    .columns
    Y =['estimated']

    # Initialize the Linear Regression Object 
    lm = LinearRegression()

    rfe = RFE(lm, 12)

    # Transforming data using RFE
    X_rfe = rfe.fit_transform(train[X],train[Y])  
    mask = rfe.support_

    # select the column names of the features that were selected and convert them to a list for future use. 
    rfe_features = train[X].columns[mask]

    lm = LinearRegression()
    lm.fit(X_rfe,train[Y])
    train['prediction'] = lm.predict(X_rfe)

    # select the column names of the features that were selected and convert them to a list for future use. 
    mask = rfe.support_
    rfe_features = train[X].columns[mask]
    rfe_features

    RMSE_train = np.sqrt(mean_squared_error(train.estimated, train.prediction))
    return RMSE_train, kmeans, lm, rfe_features


def validate_model(validate,feature_clustered,kmeans,lm,rfe_features):   
    validate['cluster'] = kmeans.predict(validate[feature_clustered])
    cluster_df = pd.get_dummies(validate.cluster)
    # concatenate the dataframe with the  one hot encoded cluster  to the original dataframe
    validate= pd.concat([validate, cluster_df], axis =1)
    # drop cluster column
    validate = validate.drop(columns = ['cluster'])
    validate['prediction'] = lm.predict(validate[rfe_features])

    RMSE_vali = np.sqrt(mean_squared_error(validate.estimated, validate.prediction))
    return RMSE_vali


def test_model(test,feature_clustered,kmeans,lm,rfe_features):   
    test['cluster'] = kmeans.predict(test[feature_clustered])
    cluster_df = pd.get_dummies(test.cluster)
    # concatenate the dataframe with the  one hot encoded cluster  to the original dataframe
    test= pd.concat([test, cluster_df], axis =1)
    # drop cluster column
    test = test.drop(columns = ['cluster'])
    test['prediction'] = lm.predict(test[rfe_features])

    RMSE_test = np.sqrt(mean_squared_error(test.estimated, test.prediction))
    return RMSE_test
