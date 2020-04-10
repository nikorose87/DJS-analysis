#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:12:08 2018

@author: nikorose
"""
from scipy.integrate import simps
import pandas as pd
import numpy as np
import scipy.optimize as optimization
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from bokeh.palettes import Spectral6, Pastel2, Set3, Category20
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models.glyphs import Text
from scipy.constants import golden
import matplotlib.pyplot as plt


class EnergyFunctions:
    """ In this routine we are going to compile all the needed functions to obtain the total work in the Dynamic Joint Stiffness of any case"""
    def integration(self, angle, moment, columna, dx= 0.5, Min = 0, Max = None):
        """Simpson rule integration based on two variables"""
        return np.around(np.float64(simps(moment[columna][Min:Max].values, angle[columna][Min:Max].values, dx=dx)), decimals=4)
    
    def integrationPower(self, power, columna, dx= 0.5, Min = 0, Max = None):
        """Simpson rule integration based on one variable"""
        return np.around(np.float64(simps(power[columna][Min:Max].values, dx=dx)), decimals=4)
    
    def zeros(self, power, vertices =4):
        a = np.array(np.zeros(vertices))
        for diff in power.columns:
            asign = np.sign(power[diff]) # To detect signs of numbers
            signchange = ((np.roll(asign,1)-asign) !=0).astype(int) # To detect the change of signs
            j, = np.where(signchange==1) # To detect the position where sign changes
            try: 
                a = np.vstack((a, j[:vertices]))
            except ValueError:
                print ("The number of times in which the data cross zeros is less than the number of vertices especified in:"+str(diff))
                j1 = np.zeros(vertices-j.shape[0])
                j = np.append(j,j1)
                a = np.vstack((a,j))
        return a[1:]
    def work(self, zeros, Data, labels, columns = ['Heel strike','Roll over','Push off','Total']):
        """This piece of code gets the partial work between zero delimiters"""
        PartialWork = np.zeros(zeros.shape)
        for i in range(zeros.shape[0]):
            for j in range(zeros.shape[1]-1):
                try:
                    PartialWork[i,j]= self.integrationPower(Data,labels[i], Min = int(zeros[i,j]), Max = int(zeros[i,j+1]))
                except IndexError:
                    PartialWork[i,j] = 0
        PartialWork[:,-1] = np.sum(np.abs(PartialWork), axis = 1)
        PartialWorkdf = pd.DataFrame(PartialWork, index= labels , columns= columns)
        return PartialWorkdf

class Regressions:
    """Main functions to do linear regression in the DJS slope, also we're detecting points to obtain the phases of gait"""
    ## Calculating points to detect each sub-phase of gait
    def CrennaPoints(self, Moments, Angles, labels, percent= 0.05, threshold = 1.7):
        ### Incremental Ratio between Moments and Angles
        Thres1 = []
        Thres2 = []
        ERP = []
        LRP = []
        DP = []
        for columna in labels:
            ## Index value where the moment increase 2% of manimum moment
            Thres1.append(np.argmax(Moments[columna].values > np.max(Moments[columna].values, 
                                    axis=0) * percent, axis = 0)) 
            Si = np.concatenate([(Moments[columna][i:i+1].values - 
                                  Moments[columna][i+1:i+2].values) / (Angles[columna][i:i+1].values - 
                                  Angles[columna][i+1:i+2].values) 
                                  for i in range(Moments[columna].values.shape[0])])
            SiAvg = Si / np.abs(np.average(Si.astype('Float64'), axis=0))
            ## Moving the trigger 
            ERP.append(np.argmax(SiAvg[Thres1[labels.index(columna)]:].astype('Float64') 
                        > threshold, axis= 0) + Thres1[labels.index(columna)])
            asignLRP = np.sign(SiAvg[int(Moments[columna].shape[0]*0.25):].astype('Float64'))
            LRP.append(np.argmin(asignLRP.astype('Float64'), axis = 0) + 
                       int(Moments[columna].shape[0]*0.25))
            Thres2.append(np.argmax(Moments[columna].values > np.max(Moments[columna].values, axis=0) * (1-percent), axis = 0)) ## Index value where the moment increases 95% of maximum moment
            DP.append(np.argmax(Moments[columna][int(Moments[columna].shape[0]*0.55):].values < np.max(Moments[columna].values, axis=0) * percent, axis = 0) + int(Moments[columna].shape[0]*0.55)) ## Index value where the moment decreases to 2% of manimum moment after reaching the peak
        CompiledPoints = np.concatenate((Thres1, ERP, LRP, Thres2, DP)).reshape((5,len(Thres1)))
        CompiledPoints = pd.DataFrame(CompiledPoints, index= ['Thres1','ERP', 'LRP', 'Thres2', 'DP'], columns= labels)
        return CompiledPoints
    # Linear regression
    def regress(self, name1,name2):
        """Function to do a linear regression"""
        train_x = name1[:int(len(name1))].reshape(-1, 1)
        train_y = name2[:int(len(name1))].reshape(-1, 1)
        y_linear_lr = linear_model.LinearRegression(n_jobs = -1)
        y_linear_lr.fit(train_x, train_y)
        R2 = y_linear_lr.score(train_x, train_y)
        pred = y_linear_lr.predict(name1)
        meanSquare = mean_squared_error(name2, pred)
        return np.array((y_linear_lr.intercept_.item(0), y_linear_lr.coef_.item(0))), meanSquare, R2, np.array(np.concatenate([name1, pred], axis=1).reshape((len(name1),2)), dtype= 'Float64')

    # Ridge regression
    def ridge(self, name1,name2, alpha = 1.0):
        """Function to do a linear regression"""
        y_linear_lr = linear_model.Ridge(alpha= alpha)
        y_linear_lr.fit(name1, name2)
        R2 = y_linear_lr.score(name1, name2)
        pred = y_linear_lr.predict(name1)
        meanSquare = mean_squared_error(name2, pred)
        return np.array((y_linear_lr.intercept_.item(0), y_linear_lr.coef_.item(0))), meanSquare, R2, np.array(np.concatenate([name1, pred], axis=1).reshape((len(name1),2)), dtype= 'Float64')

    def func(self, x, a, b):
        return a + b*x
    
    def minSquare(self, xdata,ydata):
        x0 = np.zeros(2) - 20
        sigma = np.ones(xdata.shape)
        return optimization.curve_fit(self.func, xdata, ydata, x0, sigma)
        #return optimization.leastsq(func, x0, args=(xdata, ydata))
    
    def LinearRegression2(self, Angles, Moments, labels, points, 
                          train_test_coef = 0.9, alpha = 0):
        """Function to do Linear regressions in ankle quasi-stiffness, specifically """
        order = 2
        Regressions = np.zeros(order)
        Squares = np.zeros(1)
        R2 = np.zeros(1)
        AcumPred = []
        instances = points.index
        for diff in labels:
            for phase in range(len(instances)-1): # Because there are three main phases
                #Lineal regression to obtain the ERP coefficients
                fitAngles = Angles[diff][points[diff][instances[phase]]:points[diff][instances[phase+1]]].values.reshape(-1,1)
                fitMoments = Moments[diff][points[diff][instances[phase]]:points[diff][instances[phase+1]]].values.reshape(-1,1)
                try:
                    # If ridge regression or linear regression is chosen
                    if alpha == 0:
                        y_data, meanSq, R2D2, pred =  self.regress(fitAngles, fitMoments)
                    else:
                        y_data, meanSq, R2D2, pred =  self.ridge(fitAngles, fitMoments, alpha)
                except ValueError:
                    y_data = np.zeros(order)
                    meanSq = np.zeros(1)
                    R2D2 = np.zeros(1)
                    pred = [0,0]  
                Regressions = np.vstack((Regressions, y_data))
                Squares = np.vstack((Squares, meanSq))
                R2 = np.vstack((R2, R2D2))
                AcumPred.append(pred)
        return Regressions[1:], Squares[1:], R2[1:], AcumPred

class Plotting():
    def __init__(self):
        """The main purpose of this function is to plot with the bokeh library 
        the Quasi-stiffness slope """    
        Category20[20].pop(1)
        # Matplotlib params
        
        params = {'backend': 'ps',
                  'axes.labelsize': 11,
                  'axes.titlesize': 14,
                  'font.size': 14,
                  'legend.fontsize': 10,
                  'xtick.labelsize': 10,
                  'ytick.labelsize': 10,
                  'text.usetex': True,
                  'font.family': 'sans-serif',
                  'font.serif': ['Computer Modern'],
                  'figure.figsize': (6.0, 6.0 / golden),
                  }
        
        plt.rcParams.update(params)
    
    def QuasiLine(self, name1, name2, x_label='Angle (Deg)', y_label='Moment (Nm/kg)', 
                  grid_space=450, title='No title Given',minN=None, maxN=None, 
                  size=2, leyend= 'top_left'):
        """Bokeh plot of Quasi-stiffness plus the area beneath the curve"""
        f = figure(x_axis_label=x_label, y_axis_label=y_label, plot_width=grid_space, plot_height=grid_space, title = title)
        color = 0
        for diff in name1.columns[minN:maxN]:
            f.circle(x = name1[diff],y=name2[diff], color = Spectral6[color], fill_color = Spectral6[color], size = size, legend = diff)
            color += 1
        f.legend.location = leyend
        return f
    
    def QuasiRectiLine(self, name1, name2, compiled, regression, labels, 
                       minN = None, maxN = None, grid_space= 250, 
                       leyend= "top_left", x_label= 'Angle (Deg)', 
                       y_label= 'Moment (Nm/kg)', size=5, 
                       title= 'General Quasi-Stiffness plot'):
        """Bokeh plot of Quasi-stiffness points plus the linear regression"""
        color, count = 0, 0
        Points = compiled
        figures = []
        for diff in labels[minN:maxN]:
            f = figure(x_axis_label=x_label, y_axis_label=y_label, plot_width=grid_space, plot_height=grid_space, title = diff)
            for j in range(len(compiled.index)-1):
                try:
                    f.line(regression[j+count][:,0], regression[j+count][:,1], line_width=2, color=Category20[20][color])
                except TypeError:
                    pass
            count +=len(compiled.index)-1
            f.ray(x=np.min(name1[diff]), y=name2[diff][compiled[diff]['Thres1']], length=100, angle=0, color=Category20[20][color],line_dash="4 4")
            f.ray(x=np.min(name1[diff]), y=name2[diff][compiled[diff]['Thres2']], length=100, angle=0, color=Category20[20][color],line_dash="4 4")
            f.circle(x = name1[diff], y=name2[diff], size= size, color= Category20[20][color], alpha=0.5)
            f.circle(x = name1[diff][Points[diff]], y=name2[diff][Points[diff]], size= size*2, color= Category20[20][color], alpha=0.5)
            color += 1
            figures.append(f)
        return figures
    
    def PowerLine(self, name1, name2, vertices, x_label='Cycle (Percent.)', 
                  y_label= 'Power (W/kg)', grid_space=450, title='Power cycle plot at the joint',
                  minN=None, maxN=None, size=2, leyend= 'top_left', vertical = True, 
                  text = False):
        """Bokeh plot of Power joint with respect to the gait cycle, plus zero points detection"""
        f = figure(x_axis_label=x_label, y_axis_label=y_label, plot_width=grid_space, plot_height=grid_space, title = title)
        color = 0
        for diff in name1.columns[minN:maxN]:
            f.line(x = name2,y=name1[diff], color = Spectral6[color], line_width = size, legend = diff)
            if vertical:
                for i in vertices: 
                    f.ray(x=i, y=-1.5, length=100, angle=1.57079633, color=Pastel2[8][color],line_dash="4 4")
            if text:                  
                instances = ColumnDataSource(dict(x = [-5, 15, 45, 75], y = np.ones(4)*1.5, text = ["Heel Strike", "Roll Over", "Push Off", "Swing"]))                    
                glyph = Text(x = "x", y = "y", text = "text", angle = 0.0, text_color = "black")
                f.add_glyph(instances, glyph)
            color += 1
        f.legend.location = leyend
        return f

    def GRF(self, name1, name2, labels, points=[], std=None, x_label='Gait Cycle (%)', 
            y_label= 'GRF (Nm/kg)', grid_space=450, title='GRF plot at the Ankle joint', 
            minN=None, maxN=None, size=2, leyend= 'top_right', individual = False):
        plots = []
        """Bokeh plot of GRF in ankle joint with respect to the gait cycle"""
        if not individual:
            f = figure(x_axis_label=x_label, y_axis_label=y_label, plot_width=grid_space, plot_height=grid_space, title = title)
        color = 0
        for diff in labels[minN:maxN]:
            if individual:
                f = figure(x_axis_label=x_label, y_axis_label=y_label, plot_width=grid_space, plot_height=grid_space, title = title)
            f.line(x = name2, y=name1[diff], color = Category20[20][color], line_width = size, legend = diff)
            if std:
                y1 = std[0][diff].values 
                y2 = std[1][diff].values[::-1]
                y = np.hstack((y1,y2))
                x = np.hstack((name2.values, name2.values[::-1]))
                f.patch(x, y, alpha=0.5, line_width=0.1, color= Category20[20][color])
            if points.values != []:
                for j in points[diff]:
                    f.circle(x = name2[j], y=name1[diff][j], 
                             size= size*5, color= Category20[20][color], alpha=0.5)
            color += 1
            plots.append(f)
        f.legend.location = leyend
        if individual:
                return plots
        else:
                return f

class normalization():
    def __init__(self, height, mass):
        ''' Equations to transform data into normalized gait data'''
        self.g = 9.81    
        self.h = height
        self.mass = mass
    def power_n(self, power):
        return power/(self.mass*self.g**(3.0/2.0)*np.sqrt(self.h))
    def moment_n(self, moment):
        return moment / (self.mass*self.g*self.h)
    def force_n(self,force):
        return force / (self.mass * self.g)
    def all_norm(self, power, moment, force):
        return np.array((self.power_n(power), self.moment_n(moment), self.force_n(force))) 