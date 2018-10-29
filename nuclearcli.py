from numba import (cuda, jit)

import pandas as pd
import numpy as np

def real_estate_df():
    """30 Years of Housing Prices"""

    df = pd.read_csv("https://raw.githubusercontent.com/noahgift/real_estate_ml/master/data/Zip_Zhvi_SingleFamilyResidence.csv")
    df.rename(columns={"RegionName":"ZipCode"}, inplace=True)
    df["ZipCode"]=df["ZipCode"].map(lambda x: "{:.0f}".format(x))
    df["RegionID"]=df["RegionID"].map(lambda x: "{:.0f}".format(x))
    return df

def numerical_real_estate_array(df):
    """Converts df to numpy numerical array"""

    columns_to_drop = ['RegionID', 'ZipCode', 'City', 'State', 'Metro', 'CountyName']
    df_numerical = df.dropna()
    df_numerical = df_numerical.drop(columns_to_drop, axis=1)
    return df_numerical.values

def real_estate_array():
    """Returns Real Estate Array"""

    df = real_estate_df()
    rea = numerical_real_estate_array(df)
    return rea

@jit
def mean_values(rea):
    """Calculate Mean"""

    return rea.mean()        
