#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:21:35 2022

@author: sarasaib

"""

print('This script sets the initial conditions of electrons')
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.constants as cst
from scipy.integrate import simpson as simps
import random

T = 20  #initial trap temp
m=9.11E-31
v=np.linspace(0,100000,1000)  #velocity range

def ini_vel(num=2):
    vel = np.zeros([num,3])
    a = 4*math.pi*(math.sqrt(m/(2*math.pi*cst.Boltzmann*T)))**3
    maxwell = a * v**2 * np.exp(-m/(2*cst.Boltzmann*T)*(v**2))
    plt.plot(v,maxwell)
    plt.show()
    
    index = np.where(maxwell==max(maxwell))
    print(index)
    prob = maxwell
    prob[index]+=1-simps(maxwell)
    
    for n in range(num):
        vel[n][random.randint(0,2)] = np.random.choice(v,p=prob)
    return vel

