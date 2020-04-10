#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:12:08 2018

@author: nikorose
"""
from scipy.integrate import simps
import pandas as pd
import numpy as np

class EnergyFunctions:
    """ In this routine we are going to compile all the needed functions to obtain the total work in the Dynamic Joint Stiffness of any case"""
    def integration(self, angle, moment, columna, dx= 0.5, Min = 0, Max = 101):
        """Simpson rule integration based on two variables"""
        return simps(moment[columna][Min:Max].values, angle[columna][Min:Max].values, dx=dx) 
    
    def integrationPower(self, power, columna, dx= 0.5, Min = 0, Max = 101):
        """Simpson rule integration based on one variable"""
        return simps(power[columna][Min:Max].values, dx=dx) 
    
    def zeros(self, power, vertices =4):
        a = np.array(np.zeros(vertices))
        for diff in power.columns:
            asign = np.sign(power[diff]) # To detect signs of numbers
            signchange = ((np.roll(asign,1)-asign) !=0).astype(int) # To detect the change of signs
            j, = np.where(signchange==1) # To detect the position where sign changes
            try: 
                a = np.vstack((a, j[:vertices]))
            except ValueError:
                print ("The number of times in which the data cross zeros is less than the number of vertices especified in %s" % diff)
                j1 = np.zeros(vertices-j.shape[0])
                j = np.append(j,j1)
                a = np.vstack((a,j))
        return a[1:]
    def work(self, zeros, Data, labels):
        """This piece of code gets the partial work between zero delimiters"""
        PartialWork = np.zeros(zeros.shape)
        for i in range(zeros.shape[0]):
            for j in range(zeros.shape[1]-1):
                try:
                    PartialWork[i,j]= self.integrationPower(Data,labels[i], Min = int(zeros[i,j]), Max = int(zeros[i,j+1]))
                except IndexError:
                    PartialWork[i,j] = 0
        PartialWork[:,-1] = np.sum(np.abs(PartialWork), axis = 1)
        PartialWorkdf = pd.DataFrame(PartialWork, index= labels , columns= ["Heel Strike", "Roll over","Push Off",'Total Energy'])
        return PartialWorkdf

