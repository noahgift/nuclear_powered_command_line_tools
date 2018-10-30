import click

from numba import (cuda, jit, vectorize)
import numba
import math
import pandas as pd
import numpy as np

from functools import wraps
from time import time

@click.group()
def cli():
    pass

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap


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
    return np.float32(rea)

@timing
def expmean(rea):
    """Regular Function"""

    val = rea.mean() ** 2
    return val

@timing
@jit(nopython=True)
def expmean_jit(rea, num=1000):
    """Perform multiple mean calculations"""

    val = rea.mean() ** 2
    return val

@vectorize(['float32(float32, float32)'], target='cuda')
def add_ufunc(x, y):
    return x + y


@cli.command()
def cuda_operation():
    """Performs Vectorized Operations on GPU"""

    x = real_estate_array()
    y = real_estate_array()

    print("Moving calculations to GPU memory")
    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)
    out_device = cuda.device_array(shape=(x.shape[0],x.shape[1]), dtype=np.float32)
    print(x_device)
    print(x_device.shape)
    print(x_device.dtype)

    print("Calculating on GPU")
    add_ufunc(x_device,y_device, out=out_device)

    out_host = out_device.copy_to_host()
    print(f"Calculcations from GPU {out_host}")

@cli.command()
@click.option('--jit/--no-jit', default=False)
def jit_test(jit):
    rea = real_estate_array()
    if jit:
        click.echo(click.style('Running with JIT', fg='green'))
        expmean_jit(rea)
    else:
        click.echo(click.style('Running NO JIT', fg='red'))
        expmean(rea)

if __name__ == "__main__":
    cli()
