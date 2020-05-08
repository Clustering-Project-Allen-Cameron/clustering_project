import pandas as pd
import numpy as np
from acquire import acquire_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore

def missing_value_percentage(series):
    '''Returns percentage of missing values in a series'''
    return series.isna().sum() / series.size

def drop_useless_columns(df, percentage):
    '''
    Checks the percentage of missing value of each column,
    then drops columns whose missing values are more than that percentage
    '''
    s = df.apply(missing_value_percentage, axis = 0)
    df = df.drop(columns = s[s > percentage].index.tolist()) 
    return df

def drop_duplicated_observation(df):
    '''Removes all but the most recent transaction dates
    for each house'''
    df = df.sort_values('transactiondate', ascending=False)
    return df.drop_duplicates(subset='parcelid', keep='last')

def drop_ineffecitve_columns(df):
    '''Drops columns that have been determined to not be useful'''
    ineffecitve_columns =['id',
                          'id.1',
                          'calculatedbathnbr',
                          'finishedsquarefeet12',
                          'fullbathcnt',
                          'roomcnt',
                          'assessmentyear',
                          'censustractandblock',
                          'propertylandusetypeid',
                          'rawcensustractandblock',
                          'propertycountylandusecode',
                          'transactiondate',
                          'parcelid',
                          'regionidcounty']
    df = df.drop(columns=ineffecitve_columns)
    return df

def drop_null(df):
    '''Drops null values'''
    return df.dropna()

def create_new_features(df):
    '''
    Creates several new features predicted to be useful in modeling
    '''
    df['age']=(2017 - df.yearbuilt)
    df['tax_rate']= df.taxamount/df.taxvaluedollarcnt*100
    df['estimated'] = df.taxvaluedollarcnt*10**df.logerror
    df = df.drop(columns=['yearbuilt',
                      'structuretaxvaluedollarcnt', 
                      #'taxvaluedollarcnt', 
                      'landtaxvaluedollarcnt',
                      'taxamount'])
    df = df.rename(columns={'calculatedfinishedsquarefeet': 'house_size', 
                            'lotsizesquarefeet': 'lotsize',
                            'taxvaluedollarcnt': 'actual_value'})
    return df

def creat_dummy_var(df):
    '''Encodes fips data into columns representing counties'''
    county_df = pd.get_dummies(df.fips)
    county_df.columns = ['LA', 'Orange', 'Ventura']
    # concatenate the dataframe with the 3 county columns to the original dataframe
    df= pd.concat([df, county_df], axis =1)
    # drop regionidcounty and fips columns
    df = df.drop(columns = ['fips'])
    return df

def regassign_dtypes(df):
    '''Changes data types of columns which should not be 
    represented as integers although they are numeric'''
    df[['regionidcity','regionidzip']] = (df[['regionidcity','regionidzip']]
                                          .astype('object'))
    return df

def identify_outliers(s):
    '''
    Given a series and a cutoff value, k, returns the upper outliers for the
    series.

    The values returned will be either 0 (if the point is not an outlier), or 
    a null value which will be used to drop later.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + 3*iqr
    lower_bound = q1 - 3*iqr
    return s.apply(lambda x: x if (x<upper_bound)&(x>lower_bound) else np.NaN)

def count_outliers(s, method):
    '''
    Given a series and requested testing method, returns the count of all values outside 
    the bounds of upper and lower outliers.
    '''
    if method in ['iqr','IQR']:
        k = 5
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        upper_bound = q3 + k * iqr
        lower_bound = q1 - k * iqr
        outliers_high = [num for num in s if num > upper_bound]
        outliers_low = [num for num in s if num < lower_bound]
        return len(outliers_high) + len(outliers_low)
    elif method in ['stdev','std','zscore','z-score']:
        return (zscore(s) > 5).sum() + (zscore(s) < -5).sum()
    else:
        return s[s > s.quantile(.9995)].shape[0]

def drop_outliers(df):
    '''Drops all values in a dataframe determined to be outliers'''
    for col in df.select_dtypes(include=['float64'])\
        .drop(columns = ['latitude', 'longitude']).columns:
        df[col] = identify_outliers(df[col])
    return df.dropna()

def prepare_data():
    '''Main function used to quickly run all above functions'''
    df = acquire_data()
    df = drop_useless_columns(df, .3)
    df = drop_duplicated_observation(df)
    df = drop_ineffecitve_columns(df)
    df = drop_null(df)
    df = create_new_features(df)
    df = creat_dummy_var(df)
    df = regassign_dtypes(df)
    df = drop_outliers(df)
    return df












