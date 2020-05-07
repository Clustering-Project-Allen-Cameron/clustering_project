import env
import pandas as pd
import numpy as np
from scipy.stats import zscore

def get_zillow_data():
    url = env.get_db_url('zillow')
    query = '''
    SELECT * FROM predictions_2017
    LEFT JOIN properties_2017 USING (parcelid)
    LEFT JOIN airconditioningtype USING (airconditioningtypeid)
    LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
    LEFT JOIN buildingclasstype USING (buildingclasstypeid)
    LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
    LEFT JOIN propertylandusetype USING (propertylandusetypeid)
    LEFT JOIN storytype USING (storytypeid)
    LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
    WHERE (latitude IS NOT NULL AND 
            longitude IS NOT NULL)'''
    df = pd.read_sql(query, url)

    new_dates = df.groupby(by='parcelid').transactiondate.max().reset_index()
    df.drop(columns=['parcelid','transactiondate'], inplace=True)
    df = new_dates.join(df, how='left')
    df.drop(columns='id', inplace=True)
    return df

def null_rows_info(df):
    summary = pd.DataFrame(df.columns)
    summary['num_rows_missing'] = df.isna().sum().values
    summary['pct_rows_missing'] = df.isna().sum().values / len(df)
    summary.set_index(0, inplace=True)
    summary.index.name=''
    return summary

def null_cols_info(df):
    summary = pd.DataFrame(df.isna().sum(axis=1).values)
    summary.reset_index(inplace=True)
    summary.rename(columns={0:'num_cols_missing'}, inplace=True)
    df2 = summary.groupby('num_cols_missing')\
                                .count().reset_index()
    df2['pct_cols_missing'] = df2.num_cols_missing / df.shape[1]
    df2.rename(columns={'index':'num_rows'}, inplace=True)
    return df2

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


def drop_outliers_method(s, k):
    '''
    Given a series and a cutoff value k, drops all values outside the bounds of 
    upper outliers and returns the series.
    '''
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    upper_bound = q3 + k * iqr
    # lower_bound = q1 - k * iqr
    outliers_high = pd.Series([num for num in s if num > upper_bound])
    # outliers_low = pd.Series([num for num in s if num < lower_bound])
    if count_outliers(s,'iqr') > 0:
        s = s[s < outliers_high.min()]
    # s = s[s > outliers_low.max()]

    return s

def drop_all_outliers(df, cols):
    for s in df[cols]:
        df[s] = drop_outliers_method(df[s],5)
    for s in df[cols]:
        df[s] = drop_outliers_method(df[s],5)
    df = df.dropna()
    return df

def extra_clean(df):
    #Impute nulls
    df.calculatedfinishedsquarefeet.fillna(
        df.calculatedfinishedsquarefeet.median(), inplace=True)
    df.lotsizesquarefeet.fillna(df.lotsizesquarefeet.median(),
                            inplace=True)
    df.regionidcity.fillna(df.regionidcity.mode()[0], inplace=True)
    df.regionidzip.fillna(df.regionidzip.mode()[0], inplace=True)
    df.yearbuilt.fillna(df.yearbuilt.median(), inplace=True)
    df.structuretaxvaluedollarcnt.fillna(
        df.structuretaxvaluedollarcnt.median(), inplace=True)
    df.set_index('parcelid', inplace=True)
    df.dropna(inplace=True)
    return df

def handle_missing_values(df, req_column = .7, req_row = .6):
    single_unit = [261]
    df = df[df.propertylandusetypeid.isin(single_unit)] 

    threshold = int(round(req_column*len(df),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    
    threshold = int(round(req_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)

    df=df.drop(columns=['calculatedbathnbr','finishedsquarefeet12',
                   'fullbathcnt','propertylandusetypeid',
                    'propertylandusedesc','assessmentyear',
                    'transactiondate', 'roomcnt',
                    'censustractandblock','regionidcounty'])

    df = extra_clean(df)

    return df

def cols_as_strings(df, cols):
    for col in cols:
        df[col] = df[col].astype(str)

def cols_as_floats(df, cols):
    for col in cols:
        df[col] = df[col].astype(float)