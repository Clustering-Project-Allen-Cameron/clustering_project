import env
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

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
                    'transactiondate',
                    'censustractandblock','regionidcounty'])
    return df

def extra_clean(df):
    #Drop outliers
    df.drop(df.bathroomcnt.nlargest(1).index.tolist(), 
    inplace=True)
    df.drop(df.calculatedfinishedsquarefeet.nlargest(3).index.tolist(), 
    inplace=True)
    df.drop(df.structuretaxvaluedollarcnt.nlargest(1).index.tolist(), 
    inplace=True)
    df.drop(df.landtaxvaluedollarcnt.nlargest(4).index.tolist(), 
    inplace=True)
    df.drop(df.regionidzip.nlargest(12).index.tolist(), 
    inplace=True)
    df.drop(df.lotsizesquarefeet.nlargest(13).index.tolist(), 
    inplace=True)

    #Impute nulls with median
    df.calculatedfinishedsquarefeet.fillna(
        df.calculatedfinishedsquarefeet.median(), inplace=True)
    df.lotsizesquarefeet.fillna(df.lotsizesquarefeet.median(),
                            inplace=True)
    df.regionidcity.fillna(df.regionidcity.median(), inplace=True)
    df.regionidzip.fillna(df.regionidzip.median(), inplace=True)
    df.yearbuilt.fillna(df.yearbuilt.median(), inplace=True)
    df.structuretaxvaluedollarcnt.fillna(
        df.structuretaxvaluedollarcnt.median(), inplace=True)
    df.roomcnt.replace(0,df.roomcnt.median(), inplace=True)
    df.set_index('parcelid', inplace=True)
    df.dropna(inplace=True)
    return df
