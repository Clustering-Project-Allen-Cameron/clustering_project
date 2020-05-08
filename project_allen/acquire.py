import pandas as pd
from env import url
import os.path

def zillow_data():
    query = '''
        Select *
        From properties_2017
        Join predictions_2017 Using(parcelid)
        Where propertylandusetypeid = 261
    '''
    df = pd.read_sql(query, url('zillow'))
    df.to_csv('zillow.csv')
    return df

#check if there is a csv file, if not run squl query
def acquire_data():
    if os.path.exists('zillow.csv'):
        df = pd.read_csv('zillow.csv',  index_col=0)
    else:
        df = zillow_data()
    return df