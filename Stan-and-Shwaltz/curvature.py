#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 15:08:34 2018
From Jaime of Stackoverflow
Script to smooth the data and for detecting the highest curvature
@author: nikorose
"""
# =============================================================================
# If your curve is smooth enough, you could identify your turning points as 
# those of highest curvature. Taking the point index number as the curve parameter, 
# and a central differences scheme, you can compute the curvature with the following code
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.signal import argrelextrema

class smoothness:
    
    def first_derivative(self,x) :
        return x[2:] - x[0:-2]
    
    def second_derivative(self,x) :
        return x[2:] - 2 * x[1:-1] + x[:-2]
    
    def curvature(self, x, y) :
        x_1 = self.first_derivative(x)
        x_2 = self.second_derivative(x)
        y_1 = self.first_derivative(y)
        y_2 = self.second_derivative(y)
        return np.abs(x_1 * y_2 - y_1 * x_2) / np.sqrt((x_1**2 + y_1**2)**3)
    
    # =============================================================================
    # You will probably want to smooth your curve out first, then calculate the curvature, 
    # then identify the highest curvature points. The following function does just that:
    # =============================================================================
    def plot_turning_points(self, x, y, turning_points=10, smoothing_radius=3,
                            cluster_radius=10, plot = True) :
        """ =============================================================================
        # Some explaining is in order:
        #     turning_points is the number of points you want to identify
        #     smoothing_radius is the radius of a smoothing convolution to be applied to your data before computing the curvature
        #     cluster_radius is the distance from a point of high curvature selected as a turning point where no other point should be considered as a candidate.
        # You may have to play around with the parameters a little, but I got something like this:    
        """ 
        if smoothing_radius :
            weights = np.ones(2 * smoothing_radius + 1)
            new_x = scipy.ndimage.convolve1d(x, weights, mode='constant', cval=0.0)
            new_x = new_x[smoothing_radius:-smoothing_radius] / np.sum(weights)
            new_y = scipy.ndimage.convolve1d(y, weights, mode='constant', cval=0.0)
            new_y = new_y[smoothing_radius:-smoothing_radius] / np.sum(weights)
        else :
            new_x, new_y = x, y
        k = self.curvature(new_x, new_y)
        turn_point_idx = np.argsort(k)[::-1]
        t_points = []
        while len(t_points) < turning_points and len(turn_point_idx) > 0:
            t_points += [turn_point_idx[0]]
            idx = np.abs(turn_point_idx - turn_point_idx[0]) > cluster_radius
            turn_point_idx = turn_point_idx[idx]
        t_points = np.array(t_points)
        t_points += smoothing_radius + 1
        if plot:
            plt.plot(x,y, 'k-')
            plt.plot(new_x, new_y, 'r-')
            plt.plot(x[t_points], y[t_points], 'o')
            plt.show()
        return t_points
        

# =============================================================================
# From http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html  
# =============================================================================
    def max_min_local(self, name1, name2, labels, win_len= 11, plot = True):
        if plot: f, axes = plt.subplots(1,len(labels), sharey=True, figsize = (15,10)); 
        n=0
        maxn, minn = [], []
        New_data_x, New_data_y = [],[]
        for diff in labels:
            try:
                x = name1[diff].values
            except KeyError:
                x = name1.values
            except IndexError:
                x = name1.values
            y = name2[diff].values
            new_x = self.smooth(x, window_len= win_len)
            new_y = self.smooth(y, window_len= win_len)
            New_data_x.append(new_x)
            New_data_y.append(new_y)
            # sort the data in x and rearrange y accordingly
            sortId = np.argsort(x)
            x = x[sortId]
            y = y[sortId]
            maxn.append(argrelextrema(new_y, np.greater)[0])
            minn.append(argrelextrema(new_y, np.less)[0])
            # Set common labels

            if plot:
                f.text(0.5, 0.04, 'Gait Cycle', ha='center', va='center')
                f.text(0.06, 0.5, 'GRF Vertical [N/kg %BH]', ha='center', va='center', rotation='vertical')
                plt.suptitle('Vertical GRF')
                axes[n].plot(x,y, 'k-');
                axes[n].plot(new_x, new_y, 'r-');
                axes[n].plot(new_x[maxn[n]], new_y[maxn[n]], 'bo', linewidth = 3, label='Local max');
                axes[n].plot(new_x[minn[n]], new_y[minn[n]], 'g^', linewidth = 3, label='Local min');
                n += 1
        if plot: return New_data_x, New_data_y, maxn, minn, f
        else: return New_data_x, New_data_y, maxn, minn
    
    def smooth(self,x, window_len=11,window='hanning'):
        
        """smooth the data using a window with requested size.
        
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal 
        (with the window size) in both ends so that transient parts are minimized
        in the begining and end part of the output signal.
        
        input:
            x: the input signal 
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.
    
        output:
            the smoothed signal
            
        example:
    
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)
        
        see also: 
        
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
     
        TODO: the window parameter could be the window itself if an array instead of a string
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """
        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")
    
        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")
    
    
        if window_len<3:
            return x
    
    
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    
    
        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')
    
        y=np.convolve(w/w.sum(),s,mode='valid')
        return y
