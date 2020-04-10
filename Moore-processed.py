#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 16:13:00 2018
The intention of this piece of code is to analyze the data already processed in code Moore-Data.py
@author: eprietop
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from DJSFunctions import normalization
from curvature import smoothness
import os

def plot_line(fx, fy, cycles, title='Vertical GRF'):
    f, axes = plt.subplots(1,len(fx), sharey=True, figsize = (15,10))
    f.text(0.5, 0.04, 'Gait Cycle', ha='center', va='center')
    f.text(0.06, 0.5, 'GRF Vertical [N/kg %BH]', ha='center', va='center', rotation='vertical')
    plt.suptitle('subject {i} at vel. {k} in event {j}'.format(i=title[0], j=title[1], k=title[2]))
    for n in range(2):    
        axes[n].plot(fx[n], fy[n], 'b-');
        for xc in fx[n][cycles[n]]:
            axes[n].axvline(x=xc, color='k', linestyle='--')
    return

def plot_grf(df, ylabel="GRF N [% BW]", xlabel="Time [s]", title='Ground Reaction Forces'):
    size = df.shape
    x = df.columns
    for i in range(size[0]):
        plt.plot(x,df.iloc[i,:])
    plt.ylabel(ylabel, fontsize=16) 
    plt.xlabel(xlabel, fontsize=16) 
    plt.title(title, fontsize=16) 
    return 
# Python code to remove duplicate elements 
def Remove(duplicate): 
    final_list = [] 
    for num in duplicate: 
        if num not in final_list: 
            final_list.append(num) 
    return final_list 

def cycle_detection_from_GRF(data_x, data_y, localmin, localmax, threshold, x='NA/NS', k='NA/NS', event='NA/NS' ):
    # We will work only with max above 0.5 and min below 0.15 to detect IC
    new_localmax = localmax[data_y[localmax] >= 0.5]
    new_localmin = localmin[data_y[localmin] <= 0.15]
    cycles = []
    for i in new_localmax:
        # We are intersecting values less than local max and greater than
        # localmax + 55. Change this range if needed.
        # the threshold let us to dismiss the near local minima in the same cycle
        A = new_localmin[new_localmin <= i]
        B = new_localmin[new_localmin >= i-threshold]
        try:
            cycles.append(np.intersect1d(A,B)[0]) 
        except IndexError:
            pass
    # Removing repeated points
    cycles_real = Remove(cycles)
    # detecting cycles greater than 160 frames
    q = np.array([cycles_real[i+1] - cycles_real[i] for i in range(len(cycles_real)-1)])
    error_i = q[q <= 40]
    error_p = q[q >= 160]
    err= len(error_p)*1.0/len(cycles_real) + len(error_i)*1.0/len(cycles_real)
    print ('The error detecting cycles for subject {x} at {k} at event {j} is {e}'.format(x=x, k=k, j=event, e=err))
    return cycles_real, err
            

# Anthropometric table
anthropo_table =  pd.read_excel('MooreSubjectsTable.xlsx', index_col = 'Id')
trials = anthropo_table.loc[3:,'0.8 m/s':] # taking all the trails to be analysed


# Checking if the data exist 
cwd = os.getcwd()
directory = cwd+'/Moore-Data-processed'
if not os.path.exists(directory):
    print('There is no folder with the data processed, please run Moore-data.py first')
   
os.chdir(directory)
# =============================================================================
# Separating the data needed and already processed
# =============================================================================
# doing loops for every event
events = ['First Normal Walking','Longitudinal Perturbation','Second Normal Walking']
filter_ = smoothness()
cycles_l, cycles_r = [],[]
for j in events[0:1]:  
    os.chdir(directory+'/'+j)
    for i in trials.index.values[0:1]: # index anthorpo table
        norm_ = normalization(anthropo_table['Height [m]'][i], anthropo_table['Mass [kg]'][i])
        for k in ['0.8 m/s','1.2 m/s','1.6 m/s'][0:1]:
            force_data = pd.read_csv('GRF '+str(trials[k][i]), index_col= 0)
            Forces_norm = norm_.force_n(force_data)
            # we selected 55 for 1.2 and 1.6 and 40 for 0.8, just trial and error
            if k == '0.8 m/s': thres = 55
            else: thres = 40
#            New_fx, New_fy, maxr, minr, fig= filter_.max_min_local(Forces_norm.index, Forces_norm, labels=Forces_norm.columns, win_len=30, plot = True)
            New_fx, New_fy, maxr, minr = filter_.max_min_local(Forces_norm.index, Forces_norm, labels=Forces_norm.columns, win_len=20, plot = False)
            cycle_r, cycle_l = [cycle_detection_from_GRF(New_fx[x], New_fy[x], minr[x], maxr[x],thres, i, k, j) for x in range(len(New_fx))]
            cycles_r.append(cycle_r)
            cycles_l.append(cycle_l)   
#            Deactivate the below line if you want to plot the cycle division lines
#            plot_line(New_fx, New_fy, cycles, [i,j,k])
            split_data = [New_fy[0][cycle_r[0][i]:cycle_r[0][i+1]] for i in range(len(cycle_r[0])-1)]
            split_data = [i for i in split_data if len(i) > 40] # Taking out bad cycles
#            Resizing to 100 samples, representing the data cycle
            split_data_reg = [np.resize(i, (100,)) for i in split_data]
            split_concat = pd.DataFrame(split_data)
#            The Data shuold be processed in order to plot the subjects info
            plot_grf(split_concat)
                
    os.chdir("../")
 
# =============================================================================
# Splitting and plotting the data by cycles
# =============================================================================

