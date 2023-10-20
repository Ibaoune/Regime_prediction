#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This code runs for a single station and a single regime framework.
    It would not be too hard to loop over different input files.
    
    Prepared by Josh Dorrington for Rachida El Ouaraini on 08/11/2022.
    
The steps are:
    1. Load and process regime state sequences and rainfall time series
    2. Compute seasonal mean rainfall
    3. Compute seasonal regime occurrence
    4. Compute average rainfall in regime
    5. Compute seasonal estimate of rainfall based on regime, by combining 3. and 4.
    6. Save the result
    7. Optionally, plot the result.
    
    You may want to change parts of 1. and 6. to deal with the data the way you
    have it stored on your end. 
"""

import os
import xarray as xr
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import fcts as fc


def main(path_to_station_rainfall,path_to_regime_state,plot_result,namesation,outdir):
    """1. Load and process data"""
    
    station_rainfall=pd.read_csv(path_to_station_rainfall,delimiter='\t')
    
    #THIS RELIES ON THE FILE BEING THE SAME FORMAT AS YOU SHARED WITH ME
    station_rainfall=fc.csv_to_xarray(station_rainfall) 
    
    #Here, I compute wet days and extreme days from rainfall now, but you might want to do it in advance.
    #Then you just load and pass into csv_to_xarray like for the rainfall data
    station_is_wet=(station_rainfall>0.2).astype(int)
    p95=station_rainfall.quantile(0.95)
    station_is_extreme=(station_rainfall>p95).astype(int)
    
    regime_series=xr.open_dataarray(path_to_regime_state)
    
    """2. Compute seasonal mean rainfall"""
    
    rainfall_seasonal=fc.seasonal_from_daily(station_rainfall)
    is_wet_seasonal=fc.seasonal_from_daily(station_is_wet)
    is_extreme_seasonal=fc.seasonal_from_daily(station_is_extreme)
    
    """3. Compute seasonal regime occurrence"""
    
    regime_seasonal_occs=fc.DJF_regime_occ(regime_series)
    
    regime_seasonal_occs=regime_seasonal_occs[:,1:]
    
    """4. Compute average rainfall in regime"""
    
    def average_value(x,s):
        x,s=xr.align(x,s)
        if len(x)==0:
            raise(ValueError('The rainfall and regime data are mismatched. Hours are probably wrong'))
        vals=np.array([x[s==k].mean().values for k in np.unique(s)])
        return vals
    
    rainfall_composite=average_value(station_rainfall,regime_series)
    is_wet_composite=average_value(station_is_wet,regime_series)
    is_extreme_composite=average_value(station_is_extreme,regime_series)
    
    rainfall_composite=rainfall_composite[1:]
    is_wet_composite=is_wet_composite[1:]
    is_extreme_composite=is_extreme_composite[1:]
    
    """5. Compute seasonal regime estimate of rainfall"""
    
    def regime_reconstruction(seasonal_occs,rainfall_composites):
        return (seasonal_occs*rainfall_composites).sum('regime')
    rainfall_reconstruction=regime_reconstruction(regime_seasonal_occs,rainfall_composite)
    is_wet_reconstruction=regime_reconstruction(regime_seasonal_occs,is_wet_composite)
    is_extreme_reconstruction=regime_reconstruction(regime_seasonal_occs,is_extreme_composite)
    
    """6. Save the output """
    outdir_nc = outdir+'/outNetcdf/'
    if not os.path.exists(outdir_nc) : os.mkdir(outdir_nc) 
    rainfall_reconstruction.to_netcdf(outdir_nc+"reconstructed_rainfall_"+namesation+".nc")
    is_wet_reconstruction.to_netcdf(outdir_nc+"reconstructed_is_wet_"+namesation+".nc")
    is_extreme_reconstruction.to_netcdf(outdir_nc+"reconstructed_is_extreme_"+namesation+".nc")
    
    rainfall_seasonal.to_netcdf(outdir_nc+"seasonal_rainfall_"+namesation+".nc")
    is_extreme_seasonal.to_netcdf(outdir_nc+"seasonal_is_extreme_"+namesation+".nc")
    is_wet_seasonal.to_netcdf(outdir_nc+"seasonal_is_wet_"+namesation+".nc")
    
    """7. Plot the output"""
    
    if plot_result:
        
        outdir_figs = outdir+'/outFigs/'
        if not os.path.exists(outdir_figs) : os.mkdir(outdir_figs) 
        f,a=fc.reconstruction_plot(rainfall_seasonal,rainfall_reconstruction,'Rainfall, '+namesation)
        f.savefig(outdir_figs+'Rainfall_K4_'+namesation+'.png')
            
        f,a=fc.reconstruction_plot(is_wet_seasonal,is_wet_reconstruction,'Wet Day Fraction, '+namesation)
        f.savefig(outdir_figs+'Wet_K4_'+namesation+'.png')
    
        f,a=fc.reconstruction_plot(is_extreme_seasonal,is_extreme_reconstruction,'Extreme day fraction, '+namesation)
        f.savefig(outdir_figs+'Extreme_K4_'+namesation+'.png')


if __name__ == "__main__":
    
    stations = ['Oujda','Midelt','Meknes']
    path_to_RR = 'Data/RR_data/'
    path_to_regime_state='GJR3_OK.nc'
    
    path_out = 'outdir/'
    if not os.path.exists(path_out) : os.mkdir(path_out)
    
    #Plot the data?
    plot_result=True
    
    #Where to load data from
    for ist, st in enumerate(stations):
        name_file_st = 'RR_'+st+'.txt'
        path_to_station_rainfall= path_to_RR + name_file_st
        outdir= path_out+st
        if not os.path.exists(outdir) : os.mkdir(outdir)
        main(path_to_station_rainfall,path_to_regime_state,plot_result, namesation=st, outdir=outdir)

    


    

    