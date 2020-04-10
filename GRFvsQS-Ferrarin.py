#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 10:02:40 2018
We are going to plot the GRF against quasi stiffness in all cases
@author: eprietop
"""

import pandas as pd
import warnings
import numpy as np
from bokeh.io import reset_output
from bokeh.layouts import row, column
from bokeh.plotting import show, gridplot
from DJSFunctions import Plotting
from DJSFunctions import EnergyFunctions
from DJSFunctions import Regressions
from curvature import smoothness
warnings.filterwarnings('ignore')
#output_notebook()
##Cleaning and selecting the data
reset_output()
localdir =  "/home/eprietop/Dropbox/Gait_Analysis_data/Downloaded/Ferrarin"
FerrarinData =  pd.ExcelFile(localdir+'/mmc3.xls')
labels =  ['Natural_y','XSy','Sy','My','Ly','Toe_y','Heel_y','Ascending_y',
           'Descending_y','Natural_a','XSa','Sa','Ma','La','Toe_a','Heel_a',
           'Ascending_a','Descending_a']



## Building the subtables

def tablex(sheet, minN, maxN, index):
    """Function to do the Dataframe of each variable"""
    var_name = FerrarinData.parse(sheet)
    var, SDminus, SDplus = var_name[var_name.columns[3:55:3]], var_name[var_name.columns[2:54:3]], var_name[var_name.columns[4:56:3]]
    var.columns, SDminus.columns, SDplus.columns = labels, labels, labels
    data, data1, data2 = var.loc[minN:maxN], SDminus.loc[minN:maxN], SDplus.loc[minN:maxN]
    data.index, data1.index, data2.index = index, index, index
    return data, data1, data2

# Setting Angles, moments, power
index = np.array(range(1,101))
Angles = tablex("Joint Rotations", 708, 807, index)
Moments = tablex("Joint Moments",404,503, index)
Powers = tablex("Joint Power", 203, 305, index)
GRF = tablex('Ground Reaction Forces',101,200,index)
cycle = pd.DataFrame({"cycle %":index}) #Defining Gait cycle array

### Defining the velocities of natural walking from the paper
Range_velsBH = {'Xs': [0, 0.6], 'S': [0,6, 0.8], 'M': [0.8, 1.0], 'L': [1.0, 1.5]}
Height_A = [1.61, 1.81]
Height_Y = [1.27, 1.67]
## To obtain velocities in m/s from the relation v/h, we will multiply the value given and multiply by h 
VelAdult = [k[j] * Height_A[z] for i,k in Range_velsBH.items() for j in range(2) for z in range(len(Height_A))]
VelYouths = [k[j] * Height_A[z] for i,k in Range_velsBH.items() for j in range(2) for z in range(len(Height_Y))] ## The results showed that none of the velocities is slower than 0.5 m/s

grid_space = 700

# Loadind the personalized functions
EF = EnergyFunctions()
plot = Plotting()
filter_ = smoothness()
reg = Regressions()

plots_winter = []
plots_tp = []

# =============================================================================
# Obtaining the sub-phases on the ankle DJS according to Winter's method
New_data_x, New_data_y, maxx, minn = filter_.max_min_local(cycle['cycle %'], GRF[0], labels=labels, win_len=1, plot = False)
# Due to dome points are not desired, we will eliminate the values before 9
maxx = [i[i>8] for i in maxx]
minn = [i[i>8] for i in minn]   
#minn = [i[i<60] for i in minn]
# ERP and LRP according to Winter, 1991.
ERP_points_y = np.array([i[0] for i in minn])
LRP_points_y = np.array([i[-1] for i in maxx])
points = []
for i in range(len(maxx)):
    points.append(np.sort(np.concatenate((maxx[i],minn[i]))))
    if points[i].shape[0] <= 3:
        # If the last minima is not detected, add 65 as an average point
        points[i] = np.append(points[i], [65])
# =============================================================================
# If the Winter point option is desired uncomment this    
DJS_winter = pd.DataFrame(points, index = labels,
                          columns=['Thres1','ERP', 'Thres2', 'DP'])
# Winter points figures
GRF_plot_W = plot.GRF(GRF[0], cycle['cycle %'], labels= labels, 
                    points= DJS_winter.T, std=[GRF[1], GRF[2]], 
                    y_label= 'GRF [Nm/kg %BH]', individual= True)
  

Regressions_w, MeanErrorSquares_w, R2_w, PredictedValues_w = reg.LinearRegression2(Angles[0],
                                            Moments[0], labels, DJS_winter.T, alpha= 0)
plots_winter = plot.QuasiRectiLine(Angles[0], Moments[0], DJS_winter.T, 
                                 PredictedValues_w, labels)

# =============================================================================
# if the turning point option is desired uncomment this
turning_point_DJS = [filter_.plot_turning_points(Angles[0][i].values, 
                             Moments[0][i].values, turning_points= 6, 
                             smoothing_radius = 4, cluster_radius= 15, plot=False) for i in labels]
turning_point_DJS = [np.sort(i[i<72]) for i in turning_point_DJS]

# adding missing points
for i, turning in enumerate(turning_point_DJS):
    if turning.shape[0] <= 3:
        # If the last minima is not detected, add 65 as an average point
        turning_point_DJS[i] = np.sort(np.append([8], turning))
        


DJS_turning = pd.DataFrame(np.transpose(turning_point_DJS), columns = labels, 
                           index=['Thres1','ERP', 'Thres2', 'DP'])

# =============================================================================
# Establishing the ERL and LRP phases 
# =============================================================================

GRF_plot_TP = plot.GRF(GRF[0], cycle['cycle %'], labels= labels, 
                    points= DJS_turning, std=[GRF[1], GRF[2]], 
                    y_label= 'GRF [Nm/kg %BH]', individual=True)

plots_tp.append(GRF_plot_TP)

Regressions_tp, MeanErrorSquares_tp, R2_tp, PredictedValues_tp = reg.LinearRegression2(Angles[0],
                                            Moments[0], labels, DJS_turning, alpha= 0)

plots_tp = plot.QuasiRectiLine(Angles[0], Moments[0], DJS_turning, 
                                 PredictedValues_w, labels)


#Ploting with TP
grid_tp = gridplot([i for i in np.concatenate(np.transpose(np.vstack((GRF_plot_TP, plots_tp))))], ncols=2, plot_width=300, plot_height=300)
show(grid_tp)

#Plotting with winter
grid_w = gridplot([i for i in np.concatenate(np.transpose(np.vstack((GRF_plot_W, plots_winter))))], ncols=2, plot_width=300, plot_height=300)
show(grid_w)



