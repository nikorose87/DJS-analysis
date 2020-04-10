#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 12:57:56 2018
Here we share a rich gait data set collected from fifteen subjects walking at 
three speeds on an instrumented treadmill. Each trial consists of 120 seconds 
of normal walking and 480 seconds of walking while being longitudinally 
perturbed during each stance phase with pseudo-random fluctuations in the 
speed of the treadmill belt. A total of approximately 1.5 hours of normal 
walking (> 5000 gait cycles) and 6 hours of perturbed walking (> 20,000 gait 
cycles) is included in the data set. We provide full body marker trajectories 
and ground reaction loads in addition to a presentation of processed data that
 includes gait events, 2D joint angles, angular rates, and joint torques along 
 with the open source software used for the computations.
@author: eprietop
"""
from multiprocessing import Pool
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
#import src.gait as gait
#import src.gaitpure as gait
import gaitanalysis.gait as gait
from gaitanalysis.motek import DFlowData


# =============================================================================
# Loop for every trial
# =============================================================================

# According to the paper there 15 subjects that made the trails, plus other 2
# who probe the protocol first. Also the trials 3 and 4 does not have the record 
# file, so they cannot be analysed.
# doing loops for every event
def DataMoore(trials):
    antropometric = np.array([0,0,0])
    events = ['First Normal Walking','Longitudinal Perturbation','Second Normal Walking']
    for j in events:  
        if not os.path.exists(directory+'/'+j):
            os.makedirs(j)
        os.chdir(directory+'/'+j)
        for i in trials:
            trial = i
            Datadir = '/home/eprietop/Documents/perturbed-data-paper/raw-data/T0%s/'% trial 
            data = DFlowData(Datadir+'mocap-0%s.txt'% trial,Datadir+'record-0%s.txt'% \
                                 trial,Datadir+'meta-0%s.yml'% trial)
            # =============================================================================
            # Setting the variables according to the marker and forces, referenced in the paper
            # =============================================================================
            try:
                mass = data.meta['subject']['mass']
                height = data.meta['subject']['height']
            except KeyError:
                mass = np.NaN
                height = np.NaN
            if [trial, mass, height] in antropometric:
                pass
            else:
                antropometric = np.vstack((antropometric, [trial, mass, height]))
            #events = data.meta['events']
            # List of markers for lower inverse dynamics
            body_parts = ['SHO','GTRO','LEK','LM','HEE','MT5']
            body_parts_l = ['L'+c+e for c in body_parts for e in ['.PosX','.PosY']]
            body_parts_r = ['R'+c+e for c in body_parts for e in ['.PosX','.PosY']]
            #List of forces cells
            forces_l = ['FP'+str(1)+c for c in ['.ForX','.ForY','.MomZ']]
            forces_r = ['FP'+str(2)+c for c in ['.ForX','.ForY','.MomZ']]
            
            data.clean_data()
            event_df = data.extract_processed_data(event=j)
            Gaitdata = gait.GaitData(event_df)
            Gaitdata.inverse_dynamics_2d(body_parts_l, body_parts_r,
                                      forces_l, forces_r, mass, 6.0)
            # We had to put a threshold of 20
            cycles = Gaitdata.grf_landmarks(forces_r[1], forces_l[1], threshhold=20.0, do_plot=False)
            cycles_df = pd.DataFrame(list(cycles), index = ['R_strike','L_strike','R_off','L_off'])
            cycles_df.to_csv('Cycle_detection %s'% trial)
            #Taken the data processed
            Process_data = event_df.iloc[:,-31:].round(4)
            Process_data.to_csv('ID'+'_0%s'%trial)
            GRF_Y = pd.concat((event_df[forces_r[1]],event_df[forces_l[1]]), axis = 1)
            GRF_Y.to_csv('GRF %s'% trial)
        os.chdir("../")
    antro = pd.DataFrame(antropometric, columns = ['trial', 'mass [kg]','height [m]'])
    antro.to_csv('trial-anthropometries')
    os.chdir("../") 
    return


if __name__ == '__main__':
    # In order to test the succesful test we have to create trials
    anthropo_table =  pd.read_excel('MooreSubjectsTable.xlsx', index_col = 0)
    trials = anthropo_table.loc[7:,'0.8 m/s':] # taking all the trails to be analysed
    # Verifiying if the folder already exist and setting pwd there
    cwd = os.getcwd()
    directory = cwd+'/Moore-Data-processed'
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.chdir(directory)
    DataMoore(trials.values.reshape((1,-1))[0])