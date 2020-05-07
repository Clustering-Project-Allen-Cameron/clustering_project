from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def split_data(X, y):
    X_train, X_test = train_test_split(X, train_size=.8, random_state=42)
    X_train, X_validate = train_test_split(X_train, train_size=.8, random_state=42)
    y_train, y_test = train_test_split(y, train_size=.8, random_state=42)
    y_train, y_validate = train_test_split(y_train, train_size=.8, random_state=42)

    return X_train, X_validate, X_test, y_train, y_validate, y_test

def feature_engineering(df):
    df.fips.replace({6037:'LA_County', 6059:'Orange_County',
                6111:'Ventura_County'}, inplace=True)
    df = df.rename(columns={'fips':'county'})

    df['age'] = 2017 - df.yearbuilt
    df['taxrate'] = df.taxamount/df.taxvaluedollarcnt
    df['land_sqft_price'] = df.landtaxvaluedollarcnt/\
        df.lotsizesquarefeet
    df['house_sqft_price'] = df.structuretaxvaluedollarcnt/\
        df.calculatedfinishedsquarefeet
    df = df.drop(columns = ['yearbuilt','taxvaluedollarcnt',
                    'landtaxvaluedollarcnt', 'taxamount',
                    'structuretaxvaluedollarcnt'])
    return df

def scale_data(train, validate, test):
    scaler = MinMaxScaler().fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), 
                            columns=train.columns,
                            index=train.index)
    validate_scaled = pd.DataFrame(scaler.transform(validate), 
                            columns=validate.columns,
                            index=validate.index)
    test_scaled = pd.DataFrame(scaler.transform(test), 
                            columns=test.columns,
                            index=test.index)                      
    return train_scaled, validate_scaled, test_scaled