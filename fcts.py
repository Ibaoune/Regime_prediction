#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:44:44 2023

@author: elabar
"""

import xarray as xr
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,LinearRegression



def offset_time_dim(da,offset,offset_unit='days',offset_dim='time',deep=False):
    time_offset=dt.timedelta(**{offset_unit:offset})
    new_dim=pd.to_datetime(da[offset_dim])+time_offset
    new_da=da.copy(deep=deep)
    new_da[offset_dim]=new_dim
    return new_da

def get_occurrence(states,state_combinations=None):
    
    if np.ndim(states[0])==0:
        states=np.atleast_2d(states)

    states=np.array([a for b in states for a in b])
    
    if state_combinations is None:
        state_combinations=np.unique(states)
    K=len(state_combinations)
    occ=np.zeros(K)

    #We loop over transition matrix elements, using list comprehensions
    #to unravel our list of sequences
    for i,state in enumerate(state_combinations):
        
        occ[i]=sum(states==state)/len(states)
    
    assert np.abs(np.sum(occ)-1)<1e-5
    return occ

def csv_to_xarray(series):
    series.columns=['Year','Month','Day','data']
    dat=[(dt.datetime(year=r['Year'],month=r['Month'],day=r['Day']),float(r['data'].replace(",","."))) for i,r in series.iterrows()]
    dat=np.array(dat)
    dates=dat[:,0]
    values=dat[:,1]
    da=xr.DataArray(data=values.astype(float),coords={'time':dates})
    return offset_time_dim(da,9,'hours')


def seasonal_from_daily(x):
    years=np.unique(x['time.year'])
    
    #This makes sure December is counted in the next year: i.e. Dec 2000 is counted in DJF 2001
    seasonal_arr=[np.mean(x[x['time.year']==y-(x['time.month']==12)].values) for y in years]
    
    seasonal_x=xr.DataArray(data=seasonal_arr,coords={'year':years}).dropna('year')
    
    return seasonal_x

def DJF_regime_occ(x):
    
    djf_years=np.unique((x['time.year'] + (x['time.month']==12).astype(int)))
    x=x[x['time.season']=='DJF'].dropna('time')
    seasonal_occs=xr.DataArray(
        data=np.array(\
            [get_occurrence(\
                x[x['time.year']==y-(x['time.month']==12)],\
                state_combinations=np.unique(x)\
            ) for y in djf_years])\
        ,coords={'year':djf_years,\
                 'regime':np.unique(x)})
    return seasonal_occs

def reconstruction_plot(X,y,title=''):
    
    fig,ax=plt.subplots(1)
    X.plot(marker='o',color='k',lw=3,ax=ax,label='Truth')

    x,y=xr.align(X,y)
    corr=np.corrcoef(x,y)[1,0]

    y.plot(marker='o',color='r',ax=ax,label='Reconstruction')
    ax.legend()
    ax.set_title(f'{title}, ACC:{corr:.2f}')
#    ax.set_title(f'Oujda (Region II), Cor={corr:.2f}')
    return fig,ax