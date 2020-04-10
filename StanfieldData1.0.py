#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:53:42 2018
Gait analysis was performed on 16 children between the ages of 7 and 12 years (eight boys and eight girls) 
each year for 5 consecutive years. Children walked barefoot at self-selected normal velocities. 
Ground reaction forces and motion analysis data were collected at 50 Hz for approximately three sets of data 
for each leg for each child for each of the 5 consecutive years of the study (a total of 457 trials). 
Joint angles, moments and powers were calculated using Vicon Clinical Manager and have been described previously. 
Weight, height and leg length (greater trochanter to lateral malleolus) data were recorded.
@author: nikorose
"""
import pandas as pd
import os 
import warnings
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import show, gridplot
from cycler import cycler
from DJSFunctions import EnergyFunctions as EF
from DJSFunctions import Regressions as Reg
from DJSFunctions import Plotting 
warnings.filterwarnings('ignore')
#output_notebook()
##Cleaning and selecting the data

localdir =  "../../../Gait_Analysis_data/Downloaded/Stanfield et al"
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
Mass_Avg = 32.64 ## Mass of the average children
AnkleAngle = Angles.ix[:,0::3]
AnkleMoment = Moments.ix[:,0::3] / Mass_Avg
AnklePower = Powers.ix[:,0::3] / Mass_Avg
Ankles = [AnkleAngle, AnkleMoment, AnklePower]
for i in Ankles: i.columns = labels[::-1]

Energy= EF()

### Following procedure of Crenna to obtain DJS
Work_abs = pd.DataFrame([Energy.integration(AnkleAngle, AnkleMoment, i, dx=0.4, Max = np.argmax((AnkleAngle[i].values))) for i in labels], index=labels, columns = ['Work Abs (J/s)'])
Work_prod = pd.DataFrame([Energy.integration(AnkleAngle, AnkleMoment, i, dx=0.4, Min = np.argmax((AnkleAngle[i].values))) for i in labels], index=labels, columns = ['Work Prod (J/s)'])
Work_gen = pd.DataFrame([Energy.integration(AnkleAngle, AnkleMoment, i, dx=0.4, Max = 51) for i in labels], index=labels, columns = ['Work Generated (W)']) # DJS integration
TotalWork = pd.DataFrame(np.concatenate([np.abs(Work_prod.values), np.abs(Work_abs.values), -Work_prod.values - Work_abs.values], axis = 1), index=labels, columns = ['Work Prod (J/s)', 'Work Abs (J/s)','Net work (J/s)'])

### Following the integration of Power to obtain the DJS
Work_gen2 = pd.DataFrame([Energy.integrationPower(AnklePower, i, dx=0.4, Max = 51) for i in labels], index=labels[::-1], columns = ['Work Generated (J)']) # Power integration
zeros = Energy.zeros(AnklePower)
SubPhaseWork = Energy.work(zeros, AnklePower, labels[::-1])

## Detecting subphase points in DJS loop

Regress = Reg()
points = Regress.CrennaPoints(AnkleMoment, AnkleAngle, labels, percent=0.07)
points = points*2 # Due to we have paired indexes
Thresholds = points.drop(['ERP','LRP','DP'])
Points = points.drop(['Thres1','Thres2'])

### Making a loop to see how alpha coefficient affects the regressions. Now we know that for a better R2, the alpha should be low, e.g 1e-8
#R2s =[]
#alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
## Make a while loop until R2 of 0.7 will be reached
#for alph in alpha_ridge:
Regressions, MeanErrorSquares, R2, PredictedValues = Regress.LinearRegression2(AnkleAngle, AnkleMoment, labels, points, alpha= 1e-8)
#R2s.append(np.average(R2))
# Three subplots sharing both x/y axes
f, axs = plt.subplots(1,3, figsize=(15, 6))
conf_plot = {'color': ['m', 'g', 'b'], 'linestyle': ['--', ':', '-.']}
f.suptitle('Dynamic stiffness in ankle joint for children between 7 through 12')
axs = axs.ravel()
count = 0
for i in labels:
    x0 = AnkleAngle[:][i].values.astype('Float64')
    y0 = AnkleMoment[:][i].values.astype('Float64')
    for j in range(len(points.index)-1):
        try:
            axs[labels.index(i)].plot(PredictedValues[j+count][:,0], PredictedValues[j+count][:,1], linewidth=1, color = 'k', linestyle = '-')
        except TypeError:
            pass
    count +=4
    axs[labels.index(i)].plot(x0, y0, linewidth=2, label = labels[labels.index(i)], color = conf_plot['color'][labels.index(i)], linestyle = conf_plot['linestyle'][labels.index(i)])
    axs[labels.index(i)].plot(AnkleAngle[i][Points[i].values.astype('Float64')], AnkleMoment[i][Points[i]].values.astype('Float64'), 'r+')
    axs[labels.index(i)].grid(True)
    axs[labels.index(i)].set_xlabel(i)
    axs[labels.index(i)].set_ylabel('Moment [Nm/kg]')
    for y in Thresholds[i]: axs[labels.index(i)].axhline(y=y0[int(y/2)], linewidth=1, linestyle = '--', color = 'k')

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
plt.xlabel('Gait percent')
plt.ylabel('Power[W/kg]')
ax1.set_title('Power cycle in ankle joint for children between 7 through 12')
plt.show()

### Trying to plot the direction of the DJS slope
plot = Plotting()
Figures = plot.QuasiLine(AnkleAngle,AnkleMoment, grid_space= 600, size= cycle*0.2)
#show(Figures)

### Trying to plot the linear regressions like a class

Regress_plot = plot.QuasiRectiLine(AnkleAngle, AnkleMoment, points, PredictedValues, labels)
grid = gridplot([i for i in Regress_plot], ncols=3, plot_width=400, plot_height=400)
#show(grid)

# Plotting Power cycle with bokeh
power_plot = plot.PowerLine(AnklePower,cycle,zeros*2, vertical=True)
show(power_plot)









