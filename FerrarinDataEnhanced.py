#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 19:41:52 2017

@author: nikorose
"""

import pandas as pd
import warnings
import numpy as np
from bokeh.layouts import row, column
from bokeh.palettes import Spectral6, Set3
from bokeh.plotting import figure, show, gridplot
from bokeh.models import ColumnDataSource
from bokeh.models.glyphs import Text
from bokeh.io import reset_output
from scipy.integrate import simps
import scipy.optimize as optimization
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from DJSFunctions import Plotting
from DJSFunctions import EnergyFunctions
from DJSFunctions import Regressions
from curvature import smoothness
import matplotlib.pyplot as plt
from pathlib import PurePath
import os
warnings.filterwarnings('ignore')
#output_notebook()
##Cleaning and selecting the data
reset_output()
current_dir = PurePath(os.getcwd())
data_dir =  os.chdir('../Ferrarin')
FerrarinData =  pd.ExcelFile('mmc3.xls')
labels =  ['Natural_y','XSy','Sy','My','Ly','Toe_y','Heel_y','Ascending_y',
           'Descending_y','Natural_a','XSa','Sa','Ma','La','Toe_a','Heel_a',
           'Ascending_a','Descending_a']
os.chdir(current_dir)
## Building the subtables

def tablex(sheet, minN, maxN, index):
    """Function to do the Dataframe of each variable"""
    var_name = FerrarinData.parse(sheet)
    var, SDminus, SDplus = var_name[var_name.columns[3:55:3]],var_name[var_name.columns[2:54:3]], var_name[var_name.columns[4:56:3]]
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
EF = EnergyFunctions()

def QuasiLine(x_label, y_label, grid_space, title, name1, name2, minN, maxN, size, leyend= 'top_left'):
    """Bokeh plot of Quasi-stiffness plus the area beneath the curve"""
    f = figure(x_axis_label=x_label, y_axis_label=y_label, plot_width=grid_space, plot_height=grid_space, title = title)
    color = 0
    area = []
    for diff in name1.columns[minN:maxN]:
        f.circle(x = name1[diff],y=name2[diff], color = Spectral6[color], fill_color = Spectral6[color], size = size, legend = diff)
        color += 1
        area.append(EF.integration(name1, name2, diff, 0.5))
    f.legend.location = leyend
    return f, np.array(area, dtype= np.float64)

def QuasiRectiLine(x_label, y_label, title, name1, name2, compiled, regression, minN = 0, maxN = -1, grid_space= 250, leyend= "top_left", figures=[] ):
    """Bokeh plot of Quasi-stiffness points plus the linear regression"""
    color = 0
    for diff in name1.columns[minN:maxN]:
        f = figure(x_axis_label=x_label, y_axis_label=y_label, plot_width=grid_space, plot_height=grid_space, title = diff)
        for i in range(3):
            if name1[diff][compiled[diff][i]] > name1[diff][compiled[diff][i+1]]:
                space = -0.1
                plus = -5
            else:
                space = 0.1
                plus = 1.5
            x = np.arange(name1[diff][compiled[diff][i]], name1[diff][compiled[diff][i+1]] + plus, space) ## Obtaining array from X space for regression
            y = func (x, regression[i][color + minN][0], regression[i][color + minN][1])
            f.line(x, y, line_width=2, color=Spectral6[color])
        for i in range(4): f.circle(x = name1[diff][compiled[diff][i]], y=name2[diff][compiled[diff][i]], size=10, color= Spectral6[color], alpha=0.7)
        f.ray(x=-40, y=name2[diff][compiled[diff][0]], length=100, angle=0, color=Spectral6[color],line_dash="4 4")
        f.ray(x=-40, y=name2[diff][compiled[diff][2]], length=100, angle=0, color=Spectral6[color],line_dash="4 4")
        f.circle(x = name1[diff], y=name2[diff], size=3, color= Spectral6[color], alpha=0.5)
        color += 1
        figures.append(f)
    return figures

def PowerLine(x_label, y_label, grid_space, title, name1, name2, minN, maxN, size, leyend, vertices, vertical = True):
    """Bokeh plot of Power joint with respect to the gait cycle, plus zero points detection"""
    f = figure(x_axis_label=x_label, y_axis_label=y_label, plot_width=grid_space, plot_height=grid_space, title = title)
    color = 0
    a = np.array(np.zeros(vertices))
    for diff in name2.columns[minN:maxN]:
        f.line(x = name1,y=name2[diff], color = Spectral6[color], line_width = size, legend = diff)
        color += 1
        asign = np.sign(name2[diff][0:]) # To detect signs of numbers
        signchange = ((np.roll(asign,1)-asign) !=0).astype(int) # To detect the change of signs
        j, = np.where(signchange==1) # To detect the position where sign changes
        try: 
            a = np.vstack((a, j[:vertices]))
        except ValueError:
            print ("The number of times in which the data cross zeros is less than the number of vertices especified in %s" % diff)
            j1 = np.zeros(vertices-j.shape[0])
            j = np.append(j,j1)
            a = np.vstack((a,j))
    if vertical:
        instances = ColumnDataSource(dict(x = [-5, 15, 45, 75], y = np.ones(4)*1.5, text = ["Heel Strike", "Roll Over", "Push Off", "Swing"]))
        prom = np.true_divide(a.sum(0),(a!=0).sum(0))
        for i in np.true_divide(a[1:,1:].sum(0),(a[1:,1:]!=0).sum(0)): 
            f.ray(x=i, y=-1, length=100, angle=1.57079633, color=Spectral6[color],line_dash="4 4") ## np.true_divide excludes 0 values to do the average
        glyph = Text(x = "x", y = "y", text = "text", angle = 0.0, text_color = "black")
        f.add_glyph(instances, glyph)
        a = np.vstack((a,prom))
    f.legend.location = leyend
    return f, a[1:,1:].astype(int)

# Linear regression
def regress(name1,name2, percent = 1):
    """Function to do a linear regression"""
    train_x = name1[:int(len(name1)*percent)].reshape(-1, 1)
    train_y = name2[:int(len(name1)*percent)].reshape(-1, 1)
    test_x = name1[int(len(name1)*percent):].reshape(-1, 1)
    test_y = name2[int(len(name1)*percent):].reshape(-1, 1)
    y_linear_lr = linear_model.LinearRegression(n_jobs = -1)
    y_linear_lr.fit(train_x, train_y)
    R2 = y_linear_lr.score(train_x, train_y)
    pred = y_linear_lr.predict(test_x)
    meanSquare = mean_squared_error(test_y, pred)
    return np.array((y_linear_lr.intercept_.item(0), y_linear_lr.coef_.item(0))), meanSquare, R2

def func(x, a, b):
    return a + b*x

def minSquare(xdata,ydata):
    x0 = np.zeros(2) - 20
    sigma = np.ones(xdata.shape)
    return optimization.curve_fit(func, xdata, ydata, x0, sigma)
    #return optimization.leastsq(func, x0, args=(xdata, ydata))
    
# Plotting ankle quasi-stiffness in regular gait
quasiYoung, Ywork = QuasiLine('Right Ankle Angle (Deg)', 'Moment (Nm/kg)', 
                              grid_space, "Ankle Quasi-Stiffness in young people at different speeds", 
                              Angles[0], Moments[0], 0, 5, size= cycle['cycle %']*0.1)
quasiAdult, Awork = QuasiLine('Right Ankle Angle (Deg)', 'Moment (Nm/kg)', 
                              grid_space, "Ankle Quasi-Stiffness in Adult people at different speeds", 
                              Angles[0], Moments[0], 9, 14, size= cycle['cycle %']*0.1)

# Plotting ankle quasi-stiffness at irregular gait
quasiIrregularYoung, IYwork = QuasiLine('Right Ankle Angle (Deg)', 'Moment (Nm/kg)', 
                                        grid_space, "Ankle Quasi-Stiffness in young people in irregular conditions", 
                                        Angles[0], Moments[0], 5, 9, size= cycle['cycle %']*0.1)
quasiIrregularAdult, IAwork = QuasiLine('Right Ankle Angle (Deg)', 'Moment (Nm/kg)', 
                                        grid_space, "Ankle Quasi-Stiffness in Adult people in irregular conditions", 
                                        Angles[0], Moments[0], 14, 20, size= cycle['cycle %']*0.1)


# Plotting joint power 

PowerYoung, zerosY= PowerLine('Gait cycle (%)', 'Power (W/kg)', grid_space, 
                              "Trajectory of ankle power in young people", 
                              cycle["cycle %"], Powers[0], 0, 5, 2, "top_right", 4)
PowerAdult, zerosA= PowerLine('Gait cycle (%)', 'Power (W/kg)', grid_space, 
                              "Trajectory of ankle power in adult people", cycle["cycle %"], 
                              Powers[0], 9, 14, 2, "top_right",4)
IrregularPowerYoung, zerosIY= PowerLine('Gait cycle (%)', 'Power (W/kg)', 
                                        grid_space, "Trajectory of ankle power in young people at irregular gait", 
                                        cycle["cycle %"], Powers[0], 5, 9, 3, "top_right", 
                                        4, vertical = False)
IrregularPowerAdult, zerosIA= PowerLine('Gait cycle (%)', 'Power (W/kg)', 
                                        grid_space, "Trajectory of ankle power in adult people at irregular gait", 
                                        cycle["cycle %"], Powers[0], 14, 20, 3, 
                                        "top_right", 4, vertical = False)

zerosY = EF.zeros(Powers[0].iloc[:,:5])
zerosA = EF.zeros(Powers[0].iloc[:,9:14], vertices= 5)
#show(column(row(PowerYoung,PowerAdult),row(IrregularPowerYoung, IrregularPowerAdult)))
#show(column(row(quasiYoung, quasiAdult),row(quasiIrregularYoung, quasiIrregularAdult)))

## Calculating total power and sectional power in each gait
PowerJulius = [EF.integrationPower(Powers[0], diff, 0.5) for diff in Powers[0].columns]
total=pd.DataFrame(np.abs(np.concatenate((Ywork, IYwork, Awork, IAwork))), index= labels)
Net_Work=pd.DataFrame([total[0][0:9].values,total[0][9:].values], columns= labels[:9], index= ["Young", "Adult"])


## Calculating the work on the two greatest subphases
YoungPartialWork = EF.work(zerosY, Powers[0], labels[:5])  ## the zeros Y is crossing similarly on each velocity 
AdultPartialWork = EF.work(zerosY, Powers[0], labels[9:14])
PartialWork = pd.concat([YoungPartialWork.mean(),YoungPartialWork.std(),
                         AdultPartialWork.mean(),AdultPartialWork.std()],
                            keys=['Youths mean','Youths std','Adults mean','Adults std'], axis=1)


Work_abs = pd.DataFrame([EF.integration(Angles[0], Moments[0], diff, dx= 0.4, 
                                        Max= Angles[0][diff].idxmax()) for diff in Moments[0].columns], 
                                        index=labels)
Work_prod = total.values + Work_abs.values
Totalwork = pd.DataFrame(np.reshape(np.concatenate((total.values,Work_abs.values,Work_prod)),(3,len(labels))), 
                         index = ["Net Work","Work Absorbed","Work Produced"], columns = labels)
Totalwork_youths = Totalwork.iloc[:,:5]
Totalwork_adults = Totalwork.iloc[:,9:14]


# =============================================================================
## Calculating points to detect each sub-phase of gait
### Incremental Ratio between Moments and Angles
reg = Regressions()
plot = Plotting()


# =============================================================================
# Calculating the range of each gait subphase in the ankle DJS
# =============================================================================
# TO DO: create an algorithm that obtains the best threshold
points_y = reg.CrennaPoints(Moments[0], Angles[0], labels[:5], percent=0.07, threshold=1.7)
points_a = reg.CrennaPoints(Moments[0], Angles[0], labels[9:14], percent=0.07, threshold=1.7)
# =============================================================================
### Obtaining regression coefficients at each subphase
crenna_reg_y = reg.LinearRegression2(Angles[0].iloc[:,:5], Moments[0].iloc[:,:5], labels[:5], points_y, alpha= 0)
crenna_reg_a = reg.LinearRegression2(Angles[0].iloc[:,9:14], Moments[0].iloc[:,9:14], labels[9:14], points_a, alpha= 0)
### Plotting every 
Regress_plot_y = plot.QuasiRectiLine(Angles[0], Moments[0], points_y, crenna_reg_y['predicted'], labels[:5])
Regress_plot_a = plot.QuasiRectiLine(Angles[0], Moments[0], points_a, crenna_reg_a['predicted'], labels[9:14])
Regress_plots = []
for i in range(len(Regress_plot_y)):
    Regress_plots.append(Regress_plot_y[i])
    Regress_plots.append(Regress_plot_a[i])


# =============================================================================
# Focusing on edecting the appropiate threshold for S speed on the DJS
# =============================================================================
R2s, MSQs = np.zeros(1), np.zeros(1)
Regress_plots_S = []
for i in np.linspace(1.5, 2.8, num= 6):
    points_s = reg.CrennaPoints(Moments[0], Angles[0], [labels[2], labels[11]], percent=0.07, threshold=i)
    #print (points_s)
    new_crenna_reg = reg.LinearRegression2(Angles[0], Moments[0], [labels[2], labels[11]], points_s, alpha= 0)
    new_crenna_reg['MSE'][new_crenna_reg['MSE'] == 0], new_crenna_reg['R2'][new_crenna_reg['R2'] == 0] = np.nan, np.nan
    MSQs = np.vstack((MSQs, np.array(np.nanmean(new_crenna_reg['MSE']))))
    R2s = np.vstack((R2s, np.array(np.nanmean(new_crenna_reg['R2']))))
    Regress_plots_S.extend(plot.QuasiRectiLine(Angles[0], Moments[0], points_s, new_crenna_reg['predicted'], [labels[2], labels[11]]))

### In this piece of code we show the integrated area of the ankle joint Power in gait. Adapted from http://nbviewer.jupyter.org


### compilating all young and adult population data in order to see energetic differences
Angles_y =  [Angles[i].iloc[:,:9] for i in range(3)]
Angles_a = [Angles[i].iloc[:,9:] for i in range(3)]
Moments_y =  [Moments[i].iloc[:,:9] for i in range(3)]
Moments_a = [Moments[i].iloc[:,9:] for i in range(3)]

compiled_angles = {'Values_all':[Angles_y[0],Angles_a[0]], 
                   'all_min':[Angles_y[1],Angles_a[1]],'all_max':[Angles_y[2],Angles_a[2]]}
compiled_moments = {'Values_all':[Moments_y[0],Moments_a[0]], 
                    'all_min':[Moments_y[1],Moments_a[1]],'all_max':[Moments_y[2],Moments_a[2]]}
 

# Plotting  the average group
mean_angle, mean_moment, colors = [],[],['#9ecae1', '#3182bd', '#c7e9c0','#31a354']
f = figure(x_axis_label='Angle (Deg)', y_axis_label='Moment (Nm/kg)', plot_width=500, plot_height=500, title = 'Ankle avarage DJS per group of age')


#Concatenating Angles and moments
plot = Plotting()
Avg_loop= []
for i in range(5):
    Angles_all = pd.concat([Angles_y[0].iloc[:,i],Angles_a[0].iloc[:,i]], axis= 1)
    Moments_all = pd.concat([Moments_y[0].iloc[:,i],Moments_a[0].iloc[:,i]], axis=1)
    Avg_loop.append(plot.QuasiLine(name1=Angles_all, name2=Moments_all, size=cycle['cycle %']*0.1, title= 'Ankle DJS of youths vs Adults at '+str(Moments_all.columns[0])))

for i in range(2):  
    for keys in compiled_moments.keys():
        mean_angle.append(compiled_angles[keys][i].mean(axis=1))
        mean_moment.append(compiled_moments[keys][i].mean(axis=1))
x_band, x_band2 = np.append(mean_angle[1],mean_angle[2]), np.append(mean_angle[4],mean_angle[5])
y_band, y_band2 = np.append(mean_moment[1],mean_moment[2]), np.append(mean_moment[4],mean_moment[5]) 
f.patch(x_band,y_band, color=colors[0], fill_alpha=0.2)
f.patch(x_band2,y_band2, color=colors[2], fill_alpha=0.2)
f.line(mean_angle[0], mean_moment[0], line_width = 2, color= colors[1], legend= 'Youths')
f.line(mean_angle[3], mean_moment[3], line_width = 2, color= colors[3], legend= 'Adults')
f.grid.grid_line_alpha = 0.4
f.x_range.range_padding = 0
f.legend.location = 'top_left'
Avg_loop.append(f)


# =============================================================================
# Unnormalizing data to export 
# =============================================================================
parameters = FerrarinData.parse("Parameters", skiprows=[0,1])
parameters_names = parameters['Parameter']
parameters_names = parameters_names.dropna()
task_names = parameters['Task'].drop_duplicates()
parameters_y = parameters.iloc[:,2:10]
parameters_a = parameters.iloc[:,10:]
parameters_a.columns = parameters_y.columns
#finding stride time position in parameter df
str_time_pos = parameters_names[parameters_names == 'Stride time [s]'].index[0]
stride_y = parameters_y['Mean'][str_time_pos:str_time_pos+9] #seconds
stride_a = parameters_a['Mean'][str_time_pos:str_time_pos+9] #seconds
#GRF unnormalized
unnormalized_force = lambda var: var[0]*var[1]*9.81
unnormalized_moment = lambda var: var[0]*var[1] # The unnormalized moment will only be multiplied by the mass
#Population information
young = {'age':(6,10.8,17),'mass':(41.4,15.5), 'height':(1.47, 0.2),'sex':(9,11)}
adult = {'age':(22,41.4,72),'mass':(68.5,15.8), 'height':(1.71, 0.1),'sex':(9,11)}

# the given scores are the mean results plus the max standard deviation 

#Unnormalized GRF
GRF_y_un = GRF[0].iloc[:,:9]*(young['mass'][0]+young['mass'][1])*9.81
GRF_a_un = GRF[0].iloc[:,9:]*(adult['mass'][0]+adult['mass'][1])*9.81

#Unnormalized Moments

compiled_moments_un = {'Values_all':[unnormalized_moment((Moments_y[0], young['mass'][0])),
                                     unnormalized_moment((Moments_a[0], adult['mass'][0]))], 
                    'all_min':[unnormalized_moment((Moments_y[1], young['mass'][0]-young['mass'][1])),
                               unnormalized_moment((Moments_a[1], adult['mass'][0]-adult['mass'][1]))],
                    'all_max':[unnormalized_moment((Moments_y[2], young['mass'][0]+young['mass'][1])),
                               unnormalized_moment((Moments_a[2], adult['mass'][0]+adult['mass'][1]))]}
# =============================================================================
# Transforming gait cycle df to stride time dataframe
# =============================================================================
stride_time_y = pd.DataFrame(np.array([cycle['cycle %'].values*i/100.0 for i in stride_y.values]).T,
                             columns= labels[:9])
stride_time_a = pd.DataFrame(np.array([cycle['cycle %'].values*i/100.0 for i in stride_a.values]).T,
                             columns= labels[9:])
# =============================================================================
# Finding local minimum and maximum in GRF and making smoothly
# =============================================================================
filter_ = smoothness()

New_GRF_young_data = filter_.max_min_local(stride_time_y, GRF_y_un, 
                                        labels=labels[:9], win_len=5, plot = False)
New_GRF_adult_data = filter_.max_min_local(stride_time_a, GRF_a_un, 
                                        labels=labels[9:], win_len=5, plot = False)

New_angle_young_data = filter_.max_min_local(stride_time_y, Angles[0], low_limit=4,
                                        labels=labels[:9], win_len=5, plot = False)
New_angle_adult_data = filter_.max_min_local(stride_time_a, Angles[0], low_limit=4,
                                        labels=labels[9:], win_len=5, plot = False)


# =============================================================================
# Obtaining angular velocities
# =============================================================================
ang_min_max_young = pd.concat(([New_angle_young_data['max_data'], 
                                New_angle_young_data['min_data']]), axis = 1)
ang_min_max_adult = pd.concat(([New_angle_adult_data['max_data'], 
                                New_angle_adult_data['min_data']]), axis = 1)

#Removing Irregular gait datasets
#ang_min_max_young = ang_min_max_young.drop(ang_min_max_young.index[5:])
#ang_min_max_adult = ang_min_max_adult.drop(ang_min_max_adult.index[5:])

#Arranging according the order 
ang_min_max_young= ang_min_max_young[['min0','max0','min1','max1']]
ang_min_max_adult= ang_min_max_adult[['min0','max0','min1','max1']]

# =============================================================================
# Obtaining angular velocities from the extreme points to each other
# Angular velocity will be obtained through (ang[i+1]-ang[i])/(time[i+1]-time[i])
# =============================================================================
#regular Gait
angle_data = {'angle': [New_angle_young_data['y_data'], New_angle_adult_data['y_data']],
              'time': [New_angle_young_data['x_data'], New_angle_adult_data['x_data']],
              'points': [ang_min_max_young, ang_min_max_adult]}
#Youths and adults
points_in_angles= []
points_in_time=[]

for k, i in enumerate(labels):
    # If there are nan values, all the row is set in zeros
    if k >=9: u = 1 
    else: u=0
    try:
        points_in_angles.append([angle_data['angle'][u][i][angle_data['points'][u][j][i]] for j in angle_data['points'][u].columns])
    except TypeError: 
        points_in_angles.append(list(np.zeros(4)))
    try:
        points_in_time.append([angle_data['time'][u][i][angle_data['points'][u][j][i]] for j in angle_data['points'][u].columns])
    except TypeError: 
        points_in_time.append(list(np.zeros(4)))

#Dataframes
points_in_angles = pd.DataFrame(points_in_angles, columns=angle_data['points'][0].columns, 
                                  index=labels)
points_in_time = pd.DataFrame(points_in_time, columns=angle_data['points'][0].columns, 
                                index=labels)

#Obtaining angular velocities
#Values are given in rad/ms as LSDYNA ask for this in this way
ang_vel= [(points_in_angles.iloc[:,i+1]-points_in_angles.iloc[:,i]) /
          (points_in_time.iloc[:,i+1]-(points_in_time.iloc[:,i]))*
          (np.pi/180)/1000 for i in range(points_in_angles.shape[1]-1)]
ang_vel.append(pd.Series(np.zeros(ang_vel[0].shape[0]), index=labels))
ang_vel=pd.DataFrame(np.abs(ang_vel), index=['stage{}'.format(i) for i in range(len(ang_vel))], 
                     columns = labels)

# =============================================================================
# Plotting GRF to detect if the ERP and LRP limits match with the local minimum and maximum --> Conclusion: the subphase points does not match with local min or max
# Now, plotting the real max and min points
# =============================================================================

#Example of plotting crenna points -> do not delete
GRF_plot_y = plot.GRF(GRF[0], cycle['cycle %'], labels= labels[:5], 
                      points=points_y[:]['ERP':'Thres2'], y_label= 'GRF [Nm/kg %BH]')
GRF_plot_a = plot.GRF(GRF[0], cycle['cycle %'], labels= labels[9:14], 
                      points=points_a[:]['ERP':'Thres2'], y_label= 'GRF [Nm/kg %BH]')

# =============================================================================
# Plotting unnormalized GRF
# =============================================================================

GRF_plot_y_un = plot.GRF(New_GRF_young_data['y_data'], New_GRF_young_data['x_data'], 
                         labels= labels[:5], 
                         points= New_GRF_young_data['max_data'].T, 
                         points2= New_GRF_young_data['min_data'].T, 
                         x_label='Time [s]', y_label= 'GRF [N]',
                         title='GRF unnormalized for Youths at different gait speed')
GRF_plot_a_un = plot.GRF(New_GRF_adult_data['y_data'], New_GRF_adult_data['x_data'], 
                         labels= labels[9:14], 
                         points= New_GRF_adult_data['max_data'].T, 
                         points2= New_GRF_adult_data['min_data'].T, 
                         x_label='Time [s]', y_label= 'GRF [N]',
                         title='GRF unnormalized for Adults at different gait speed')

# =============================================================================
# Plotting Angles at different gait speed unnormalized
# =============================================================================
angles_plot_y_un = plot.GRF(New_angle_young_data['y_data'], New_angle_young_data['x_data'],
                        labels= labels[:5], 
                        points= New_angle_young_data['max_data'].T, 
                        points2= New_angle_young_data['min_data'].T, 
                        x_label='Time [s]', y_label= 'Angle [Deg]',
                        title='Ankle angle for youths')

angles_plot_a_un = plot.GRF(New_angle_adult_data['y_data'], New_angle_adult_data['x_data'],
                        labels= labels[9:14],
                        points= New_angle_adult_data['max_data'].T, 
                        points2= New_angle_adult_data['min_data'].T, 
                        x_label='Time [s]', y_label= 'Angle [Deg]',
                        title='Ankle angle for adults')

# =============================================================================
# Plotting unnormalized ankle angle 
# =============================================================================
angles_plot_y = plot.GRF(Angles[0], cycle['cycle %'], labels= labels[:5], 
                      y_label= 'Ankle Angle [Deg]', x_label='Time [s]',
                      title='Ankle angle for youths [normalized]')
angles_plot_a = plot.GRF(Angles[0], cycle['cycle %'], labels= labels[9:14], 
                      y_label= 'Ankle Angle [Deg]', x_label='Time [s]',
                      title='Ankle angle for adults [normalized]')

# =============================================================================
# No normal walking
# =============================================================================

i_angles_plot_y_un = plot.GRF(Angles[0], stride_time_y, labels= labels[5:9], 
                      y_label= 'Ankle Angle [Deg]', x_label='Time [s]',
                      points= New_angle_young_data['max_data'].T, 
                      points2= New_angle_young_data['min_data'].T, 
                      title='Ankle angle for youths for irregular gait intention [unnormalized]')
i_angles_plot_a_un = plot.GRF(Angles[0], stride_time_a, labels= labels[14:], 
                      y_label= 'Ankle Angle [Deg]', x_label='Time [s]',
                      points= New_angle_adult_data['max_data'].T, 
                      points2= New_angle_adult_data['min_data'].T, 
                      title='Ankle angle for adults [unnormalized]')

i_angles_plot_y = plot.GRF(Angles[0], cycle['cycle %'], labels= labels[5:9], 
                      y_label= 'Ankle Angle [Deg]', x_label='Time [s]',
                      title='Ankle angle for youths [normalized]')
i_angles_plot_a = plot.GRF(Angles[0], cycle['cycle %'], labels= labels[14:], 
                      y_label= 'Ankle Angle [Deg]', x_label='Time [s]',
                      title='Ankle angle for adults for irregular gait intention [normalized]')

# =============================================================================
# Exporting data to csv
# =============================================================================
#Exporting GRF
GRF_un_data_y = pd.concat([New_GRF_young_data['x_data'], 
                           New_GRF_young_data['y_data']], 
                            axis=1, keys=['time','force'])
GRF_un_data_a = pd.concat([New_GRF_adult_data['x_data'], 
                           New_GRF_adult_data['y_data']], 
                            axis=1, keys=['time','force'])

GRF_un_data = pd.concat([GRF_un_data_y, GRF_un_data_a], axis=1)
GRF_un_data.to_csv('GRF_ferrarin_un.csv', float_format='%.4f')
#GRF_un_data_y.to_csv('GRF_young_un.csv', float_format='%.4f')
#GRF_un_data_a.to_csv('GRF_adult_un.csv', float_format='%.4f')

#Exporting angles

angle_conca = pd.concat(angle_data['angle'], axis=1)
time_conca = pd.concat(angle_data['time'], axis=1)
angle_un_data = pd.concat([time_conca, angle_conca], axis=1, keys=['time','angle'])
angle_un_data.to_csv('angle_ferrarin_un.csv', float_format='%.4f')

#Exporting angular velocity

points_in_time.to_csv('Max_min_points_time.csv', float_format='%.4f')
ang_vel.to_csv('Angular_vel_ferrarin.csv', float_format='%.4f')



#Exporting moments
#concat dict 
moments_conca = {(i,k): j[k] for i in compiled_moments_un.keys() for j in compiled_moments_un[i] for k in j}
moments_conca_df = pd.DataFrame.from_dict(moments_conca)
moments_conca_df.index = range(moments_conca_df.shape[0])
moments_conca_df.to_csv('moment_ferrarin_un.csv', float_format='%.4f')
# =============================================================================
# Establishing new segmentation points for calculating the QS, according to the max and min
# local point in the GRF
# =============================================================================

## We choose the first local min because here dorsi-flexion begins
new_points_y = pd.concat([New_GRF_young_data['max_data'], 
                          New_GRF_young_data['min_data']], axis=1)
new_points_y.columns = ['Thres1', 'Thres2','ERP', 'DP'] #Labeling the DF
new_points_y = new_points_y[['Thres1','ERP', 'Thres2','DP']] #Changing the order
#new_points_y['Thres2'] = 43 # Taking threshold as 43 % of the CG
new_points_y['DP'] = 65 # Taking the DP point as 65 % of the CG
new_points_y =new_points_y.astype(int).T # <- The instances must be in the rows

#new_points_y = pd.DataFrame(np.vstack((Thres1_y,ERP_points_y, LRP_points_y,Thres2_y)), columns= labels[:9], index=['Thres1','ERP', 'Thres2', 'DP'])
GRF_regression = reg.LinearRegression2(Angles[0], Moments[0], labels[:9], 
                                       new_points_y, alpha= 0)
new_plot_GRF = plot.QuasiRectiLine(Angles[0], Moments[0], new_points_y, 
                                 GRF_regression['predicted'], labels[:9])

#turning_points_GRF = filter_.plot_turning_points(cycle['cycle %'].values, GRF[0]['Sy'].values, turning_points= 5, smoothing_radius = 4, cluster_radius= 6)

turning_points_DJS_y = [filter_.plot_turning_points(Angles[0][i].values, Moments[0][i].values, 
                                                    turning_points= 6, smoothing_radius = 4, 
                                                    cluster_radius= 15, plot=False) for i in labels[:9]]
turning_points_DJS_a = [filter_.plot_turning_points(Angles[0][i].values, Moments[0][i].values, 
                                                    turning_points= 6, smoothing_radius = 4, 
                                                    cluster_radius= 15, plot= False) for i in labels[9:]]
# =============================================================================
# Saving the turning points in dataframe
# =============================================================================

new_turning_points_y = [np.sort(i) for i in turning_points_DJS_y]
new_turning_points_y = pd.DataFrame(new_turning_points_y, index=labels[:9], columns=['Thres1','ERP', 'Thres2','DP','other']).T
new_turning_points_y = new_turning_points_y.drop(['other'])

turning_regression = reg.LinearRegression2(Angles[0], Moments[0], labels[:9], 
                                       new_points_y, alpha= 0)
new_plot_turning = plot.QuasiRectiLine(Angles[0], Moments[0], new_turning_points_y, 
                                 turning_regression['predicted'], labels[:9])

you_wanna_plot = True
if you_wanna_plot:
# =============================================================================
#     #    plotting power and QS with STD with matplotlib
# =============================================================================
    plot.plot_power_and_QS(Powers, Angles, Moments, cycle["cycle %"], labels[0])   
# =============================================================================
#     # Plotting Quasi-Stiffness
# =============================================================================
    #    Plotting Sy at many threshold points
    grid_S = gridplot([i for i in Regress_plots_S], ncols=2, plot_width=400, plot_height=400)
    show(grid_S)

    #   Plotting the transition from right to left of the youths QS compared with
    grid_1 = gridplot(Avg_loop, ncols=2, plot_width=300, plot_height=300)
    show(grid_1)

    #Plotting regressions with crenna points at 1.7
    grid_crenna17 = gridplot([i for i in Regress_plots], ncols=2, plot_width=400, plot_height=400)
    show(grid_crenna17)
    # Plotting QS regression with points taken from GRF
    grid_2 = gridplot([i for i in new_plot_GRF], ncols=2, plot_width=300, plot_height=300)
    show(grid_2) # Taking the Threshold 2 as the second max local in the GRF shows bad results
    # Plotting regression with turning points
    grid_3 = gridplot([i for i in new_plot_turning], ncols=2, plot_width=300, plot_height=300)
    show(grid_3) 
    
# =============================================================================
#   Plotting GRFs and Angles
# =============================================================================
    #     Plotting GRF normalized vs unnormalized
    show(column(row(GRF_plot_y,GRF_plot_a),row(GRF_plot_y_un, GRF_plot_a_un)))
    #   Plotting GRF against ankle angle normalized
#    show(column(row(GRF_plot_y,GRF_plot_a),row(angles_plot_y,angles_plot_a)))
    #   Plotting angles normalized vs withou normalization
    show(column(row(angles_plot_y, angles_plot_a),row(angles_plot_y_un, angles_plot_a_un)))
    #Irregular angles
    show(column(row(i_angles_plot_y, i_angles_plot_a),row(i_angles_plot_y_un, i_angles_plot_a_un)))
