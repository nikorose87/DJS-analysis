#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:53:42 2018

@author: nikorose
"""
import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from DJSFunctions import EnergyFunctions as EF
from DJSFunctions import Regressions as Reg
warnings.filterwarnings('ignore')
#output_notebook()
##Cleaning and selecting the data

localdir =  "/home/nikorose/Dropbox/Gait_Analysis_data/Downloaded/Stanfield et al"
StandfieldData =  pd.ExcelFile(localdir+'/SpeedAllSummary.xls')
labels =  ['0.5 m/s', '1.0 m/s', '1.5 m/s']
ranges = {labels[0]: [95,95+15], labels[1]: [112,112+15], labels[2]: [129,129+15]}

## Building the subtables

def tablex(sheet, minN, maxN, cyclelabel):
    """Function to do the Dataframe of each variable"""
    var_name = StandfieldData.parse(sheet)
    var = var_name.iloc[minN:maxN, 1:53]
    var.index = var_name.iloc[minN:maxN, 0]
    var.column = cyclelabel
    return var 

# Setting Angles, moments, power
cycle = np.array(range(2, 101, 2))
Tables = [tablex("Angles", i[0], i[1], cycle) for i in ranges.values()]
TablesT = [Tables[i].T for i in range(len(ranges))]
Angles = [TablesT[i].ix[:,0:3] for i in range(len(ranges))] # ix is for calling rows by position, not by the name
Moments = [TablesT[i].ix[:,3:6] for i in range(len(ranges))]
Powers = [TablesT[i].ix[:,6:9] for i in range(len(ranges))]
Angles = pd.concat(Angles, axis = 1)
Moments = pd.concat(Moments, axis = 1)
Powers = pd.concat(Powers, axis = 1)

## Extracting data only for the ankle joint 

AnkleAngle = Angles.ix[:,0::3]
AnkleMoment = Moments.ix[:,0::3]
AnklePower = Powers.ix[:,0::3]
Ankles = [AnkleAngle, AnkleMoment, AnklePower]
for i in Ankles: i.columns = labels[::-1]

Energy= EF()

### Dollowing procedure of Crenna to obtain DJS
Work_abs = pd.DataFrame([Energy.integration(AnkleAngle, AnkleMoment, i, dx=0.4, Max = np.argmax((AnkleAngle[i].values))) for i in labels], index=labels, columns = ['Work Abs (J/s)'])
Work_prod = pd.DataFrame([Energy.integration(AnkleAngle, AnkleMoment, i, dx=0.4, Min = np.argmax((AnkleAngle[i].values))) for i in labels], index=labels, columns = ['Work Prod (J/s)'])
Work_gen = pd.DataFrame([Energy.integration(AnkleAngle, AnkleMoment, i, dx=0.4, Max = 51) for i in labels], index=labels, columns = ['Work Generated (W)']) # DJS integration
TotalWork = pd.DataFrame(np.concatenate([np.abs(Work_prod.values), np.abs(Work_abs.values), -Work_prod.values - Work_abs.values], axis = 1), index=labels, columns = ['Work Prod (J/s)', 'Work Abs (J/s)','Net work (J/s)'])

### Following the integration of Power to obtain the DJS
Work_gen2 = pd.DataFrame([Energy.integrationPower(AnklePower, i, dx=0.4, Max = 51) for i in labels], index=labels[::-1], columns = ['Work Generated (J/s)']) # Power integration
zeros = Energy.zeros(AnklePower)
SubPhaseWork = Energy.work(zeros, AnklePower, labels[::-1])

## Detecting subphase points in DJS loop

Regress = Reg()
points = Regress.CrennaPoints(AnkleMoment, AnkleAngle, labels, percent=0.07)
points = points*2 # Due to we have paired indexes
Thresholds = points.drop(['ERP','LRP','DP'])
Points = points.drop(['Thres1','Thres2'])

Regressions, MeanErrorSquares, R2, PredictedValues = Regress.LinearRegression2(AnkleAngle, AnkleMoment, labels, points/2)
# Three subplots sharing both x/y axes
f, axs = plt.subplots(1,3, figsize=(15, 6))
conf_plot = {'color': ['m', 'g', 'b'], 'linestyle': ['--', ':', '-.']}
f.suptitle('Dynamic stiffness in ankle joint for children between 7 through 12')
axs = axs.ravel()
mem = []
for i in labels:
    x0 = AnkleAngle[:][i].values.astype('Float64')
    y0 = AnkleMoment[:][i].values.astype('Float64')
    count = 0
    for j in range(len(points.index)-1):
                if AnkleAngle[i][points[i][j]] > AnkleAngle[i][points[i][j+1]]:
                    space = -0.5
                    plus = -5
                else:
                    space = 0.5
                    plus = 1.5
                x1 = np.arange(AnkleAngle[i][points[i][j]], AnkleAngle[i][points[i][j+1]], space) ## Obtaining array from X space for regression
                y1 = Regress.func(x1, Regressions[count][0], Regressions[count][1])
                mem.append([x1,y1])
                axs[labels.index(i)].plot(x1, y1, linewidth=1, label = labels[labels.index(i)], color = conf_plot['color'][labels.index(i)])
    axs[labels.index(i)].plot(x0, y0, linewidth=2, label = labels[labels.index(i)], color = conf_plot['color'][labels.index(i)], linestyle = conf_plot['linestyle'][labels.index(i)])
    axs[labels.index(i)].plot(AnkleAngle[i][Points[i].values.astype('Float64')], AnkleMoment[i][Points[i]].values.astype('Float64'), 'r+')
    axs[labels.index(i)].grid(True)
    axs[labels.index(i)].set_xlabel(i)
    axs[labels.index(i)].set_ylabel('Moment [Nm]')
    for y in Thresholds[i]: axs[labels.index(i)].axhline(y=y0[int(y/2)], linewidth=1, linestyle = '--', color = 'k')

#### Plotting Dynamic Joint Stiffness at the ankle at once graph
#fig, ax = plt.subplots(figsize=(8, 8))
#plt.rc('axes', prop_cycle=(cycler('color', ['m', 'g', 'b']) + cycler('linestyle', ['--', ':', '-.'])))
#for i in labels:
#    x0 = AnkleAngle[:][i].values.astype('Float64')
#    y0 = AnkleMoment[:][i].values.astype('Float64')
#    plt.plot(x0, y0, linewidth=2, label = labels[labels.index(i)])
#    plt.plot(AnkleAngle[i][Points[i].values.astype('Float64')], AnkleMoment[i][Points[i]].values.astype('Float64'), 'r+')
#    for y in Thresholds[i]: plt.axhline(y=y, linewidth=1, linestyle = '--', color = 'k')
#ax.set_title('Dynamic stiffness in ankle joint for children between 7 through 12')
#plt.legend()
#plt.grid(True)
#plt.show()

### Plotting power cycle and integration 
fig, ax1 = plt.subplots(figsize=(8, 8))
plt.rc('axes', prop_cycle=(cycler('color', ['m', 'g', 'b']) + cycler('linestyle', ['--', ':', '-.'])))
for i in labels:
    P0= AnklePower[:][i].values.astype('Float64')
    plt.plot(cycle, P0, linewidth=2, label = labels[labels.index(i)])
    xcoords = zeros[labels.index(i)] * 2
    for xc in xcoords: plt.axvline(x=xc, linewidth=1, linestyle = '--', color = 'k')
plt.legend()
plt.grid(True)
ax1.set_title('Power cycle in ankle joint for children between 7 through 12')
plt.show()


f1, ax = plt.subplots(figsize=(10, 6))
for i in range(4): 
    try:
        plt.plot(PredictedValues[i][:,0], PredictedValues[i][:,1])
    except TypeError:
        pass









