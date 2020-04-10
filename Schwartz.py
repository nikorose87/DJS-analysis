#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 14:33:06 2018

@author: eprietop
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

# =============================================================================
# Some functions
# =============================================================================
def errorfill(Data, x, color=None, alpha_fill=0.3, ax=None):
    """
    Plotting error bars.
    """   
    y = Data['mean'].values
    errorPower = [y - Data['-1sd'].values, Data['+1sd'].values -y] 
    fig, ax1 = plt.subplots()
    ax1.plot(x, y, 'r', linewidth=2, label='Ankle Joint Power')
    ax1.errorbar(x, y, yerr=errorPower, fmt='o', marker='o', markersize=8,
            linestyle='dotted',label='SD')
    #plt.ylabel('Ankle Power (W/kg)')
    #plt.xlabel('Gait Cycle [%]')
    return ax1 
#    ax.fill_between(x, y-0.02, y+0.02, color=color, alpha=alpha_fill)
# =============================================================================
# #Cleaning and selecting the data
# =============================================================================
localdir =  "../../../Gait_Analysis_data/Downloaded/Schwartz et al"

# Selecting angles, GRF, Moment and Power
DataSheetIndex = ['Joint Rotations', 'Ground Reaction Forces','Joint Moments','Joint Power']
Schwartz = [pd.read_excel(localdir+"/Schwartz.xls", sheet_name= i) for i in DataSheetIndex]

Iterations = [['XS','S','M','L','XL'],['-1sd','mean','+1sd']]
labelsx =  pd.MultiIndex.from_product(Iterations, names =['Velocity','Stats'])
level1 = ['L', 'M', 'S', 'XL', 'XS']
indexes = range(51)
cycle_gait = np.linspace(0,1,51)
## Extracting the data for the Ankle Angles DJS
AnkleAngle = Schwartz[0].loc[Schwartz[0]['Angle'] == 'Ankle Dorsi/Plantarflexion'].drop(['Angle','% Gait Cycle'], axis = 1)
AnkleAngle.columns = labelsx
AnkleAngle.index = indexes

## Extracting the data for the Ankle GRF 
AnkleGRF = Schwartz[1].loc[Schwartz[1]['Force  [% Body Weight]'] == 'Vertical'].drop(['Force  [% Body Weight]','% Gait Cycle'], axis = 1)
AnkleGRF.columns = labelsx
AnkleGRF.index = indexes

## Extracting the data for the Ankle Moments DJS
AnkleMoment = Schwartz[2].loc[Schwartz[2]['Moment'] == 'Ankle Dorsi/Plantarflexion'].drop(['Moment','% Gait Cycle'], axis = 1)
AnkleMoment.columns = labelsx
AnkleMoment.index = indexes

## Extracting the data for the Ankle Powers DJS
AnklePower = Schwartz[3].loc[Schwartz[3]['Power'] == 'Ankle'].drop(['Power','% Gait Cycle'], axis = 1)
AnklePower.columns = labelsx
AnklePower.index = indexes

## Loading classes
Energy= EF()
plot = Plotting()
reg = Reg()

### Dollowing procedure of Crenna to obtain DJS
Work_abs = pd.DataFrame([Energy.integration(AnkleAngle, AnkleMoment, i, dx=0.4, Max = np.argmax((AnkleAngle[i].values))) for i in labelsx], index=labelsx, columns = ['Work Abs (J/s)'])
Work_prod = pd.DataFrame([Energy.integration(AnkleAngle, AnkleMoment, i, dx=0.4, Min = np.argmax((AnkleAngle[i].values))) for i in labelsx], index=labelsx, columns = ['Work Prod (J/s)'])
Work_gen = pd.DataFrame([Energy.integration(AnkleAngle, AnkleMoment, i, dx=0.4, Max = 51) for i in labelsx], index=labelsx, columns = ['Work Generated (W)']) # DJS integration
TotalWork = pd.DataFrame(np.concatenate([np.abs(Work_prod.values), np.abs(Work_abs.values), -Work_prod.values - Work_abs.values], axis = 1), index=labelsx, columns = ['Work Prod (J/s)', 'Work Abs (J/s)','Net work (J/s)'])

### Following the integration of Power to obtain the DJS
Work_gen2 = pd.DataFrame([Energy.integrationPower(AnklePower, i, dx=0.4, Max = 51) for i in labelsx], index=labelsx[::-1], columns = ['Work Generated (J)']) # Power integration
zeros = Energy.zeros(AnklePower)
SubPhaseWork = Energy.work(zeros, AnklePower, labelsx[::-1])

# df.xs('price', level=1, drop_level=False) To call multirow, df.xs('price', axis=1, level=1, drop_level=False) to call multicolumn
power_plot_Schartz = plot.PowerLine(AnklePower.xs('mean', axis=1, level=1, drop_level=True), cycle_gait*100,zeros*2, title='Power plot in Schwaltz paper', vertical=False)

# =============================================================================
# Calculating the range of each gait subphase in the ankle DJS
# =============================================================================
points_Schartz = reg.CrennaPoints(AnkleMoment.xs('mean', axis=1, level=1, drop_level=True), AnkleAngle.xs('mean', axis=1, level=1, drop_level=True), level1, percent=0.07, threshold=1.7)
# =============================================================================
### Obtaining regression coefficients at each subphase
Regressions_Sch, MeanErrorSquares_Sch, R2_Sch, PredictedValues_Sch = reg.LinearRegression2(AnkleAngle.xs('mean', axis=1, level=1, drop_level=True), AnkleMoment.xs('mean', axis=1, level=1, drop_level=True), level1, points_Schartz, alpha= 0)

### Plotting every 
Regress_plot_Sch = plot.QuasiRectiLine(AnkleAngle.xs('mean', axis=1, level=1, drop_level=True), AnkleMoment.xs('mean', axis=1, level=1, drop_level=True), points_Schartz, PredictedValues_Sch, level1)
Clock_plot_sch= plot.QuasiLine(AnkleAngle.xs('mean', axis=1, level=1, drop_level=True), AnkleMoment.xs('mean', axis=1, level=1, drop_level=True), title='DJS direction overview')
#QuasiLine(self, name1, name2, x_label='Angle (Deg)', y_label='Moment (Nm/kg)', grid_space=450, title='No title Given',minN=None, maxN=None, size=2, leyend= 'top_left'):
Regress_plots = [Clock_plot_sch]

for i in range(len(Regress_plot_Sch)):
    Regress_plots.append(Regress_plot_Sch[i])
grid = gridplot([i for i in Regress_plots], ncols=2, plot_width=400, plot_height=400)
show(grid)
