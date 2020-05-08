import pandas as pd
import numpy as np
from prepare import prepare_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def split_data(df, size = .8):
    train, test = train_test_split(df, train_size = size, random_state=123)
    train, validate = train_test_split(train, train_size = size, random_state = 123)
    return train, validate, test

def scale_data(train, validate, test, X):
    scaler = MinMaxScaler()
    scaler.fit(train[X])
    train[X] = scaler.transform(train[X])
    validate[X] = scaler.transform(validate[X])
    test[X] = scaler.transform(test[X])
    return scaler, train, validate, test

def split_scale_data():
    df = prepare_data()
    train, validate, test = split_data(df)
    #Select features to be scaled
    X = train.select_dtypes(include = ['float']).columns
    return scale_data(train, validate, test, X)
    
    
    
    
    
    
    