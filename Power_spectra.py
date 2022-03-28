#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import boris_simulation as boris
import matplotlib.pyplot as plt
import scipy.constants as cst
from math import pi
import time

q = -1*cst.e #not sure what to do with the increase in speed and the particle reaching the speed higher than c
m = cst.electron_mass
N = 1 # number of particles

mass_array = np.transpose(np.ones(N))*m
mass_array = mass_array[:,None]   #making an array of N x 1
print('mass_array',mass_array)

q_array = np.transpose(np.ones(N))*q
q_array = q_array[:,None]
print('q_array',q_array)

E_ext = np.array([0,0,0])  #needs to be updated according the penning trap potential
B_ext = np.array([0,0,1]) *0.7

#v_array_old = np.array([[10,0,0]])
v_array_old = np.ones(shape = (N,3))*14215 #This is the mean velocity when the total mean v = 24600 m/s from v=sqrt(2KT/m)
print('v_array_old',v_array_old)

v_array_new = np.zeros(shape = (N,3))

if N==2:
    r_array_old = np.array([[1,1,1],[-1,-1,-1]])  #for 2 particles
#for 1 particle only:
r_array_old = np.array([[0.0001,0.0001,0.0001]])

print('r_array_old',r_array_old)
r_array_new = np.zeros(shape = (N,3))
#calculating B field
B_array = B_ext
print('B_array = ',B_array)
    
Lamda = abs(cst.c/(q*0.7/m/(2*pi))) #resonance
frequency = abs(q*0.7/m/(2*pi))   #=19594742910.633125

def power(r_array_old,v_array_old,B_array,E_ext,freq):
    #Should we take the E_fin - E_ini? Or average over?
    [v, r, t, Time_max, E, totalKE, trapU, intU] = boris.calc_rvE(r_array_old,v_array_old,B_array,E_ext,cst.c/freq, injection_time=1E-6)   #readme store parameters for each lamda
    totalE1 = totalKE + trapU + intU
    totalE2 = totalKE + trapU
    #print('---',np.size(v))
    #print('totKE', totalKE[0],'trapU', trapU[0],'intU',intU[0])
    #print('totKE', totalKE[-1],'trapU', trapU[-1],'intU',intU[-1])
    #tot_energy.append(totalE)
    P1 = (totalE1[-1] - totalE1[0])/Time_max   #E_final/ total calc time
    P2 = (totalE2[-1] - totalE2[0])/Time_max
    #readme in reality, we want to change this by (1) increase initial v (what is the applicable scale?)
    #(2) P = change in E / time
    # what sould the initial v be? how fast are the electrons going?
    return Time_max, P1, P2, [v, r, t, Time_max, E, totalKE, trapU, intU]

def main():   
    start = time.time()
    power_array1 = []  #tracks power for each frequency of microwave
    power_array2 = []
    f_left = 19.55E9 #frequency*0.5
    f_right = 19.6E9 #frequency*1.5
    f_array = np.linspace(f_left,f_right,1) #resonance frequency for B=0.7 is lamda = 0.015321487301949693
    
    count = 1
    for freq in f_array: 
        #plot p vs w
        Time_max, P1, P2 , ar = power(r_array_old,v_array_old,B_array,E_ext,freq)
        power_array1.append(P1)
        power_array2.append(P2)
        power_calc_time = time.time()
        print('count = ', count, ' /', str(np.size(f_array)))
        print('time until now ='+str(power_calc_time - start))
        count+=1
        
    spectra(f_array,power_array1,label = 'including interaction')
    spectra(f_array,power_array2,label = 'no interaction')
    end = time.time()
    print('time took = ' + str(end - start))
    return power_array1,power_array2
        
def power_spectra():
    start = time.time()
    power_array1 = []  #tracks power for each frequency of microwave
    power_array2 = []
    data_ar = []  #contains data such as r,v, etc for each freq
    
    f_left = 19.55E9 #frequency*0.5
    f_right = 19.6E9 #frequency*1.5
    f_array = np.linspace(f_left,f_right,2) #resonance frequency for B=0.7 is lamda = 0.015321487301949693
    
    count = 1
    for freq in f_array: 
        #plot p vs w
        Time_max, P1, P2 , ar = power(r_array_old,v_array_old,B_array,E_ext,freq)
        power_array1.append(P1)
        power_array2.append(P2)
        data_ar.append(ar)
        power_calc_time = time.time()
        print('count = ', count, ' /', str(np.size(f_array)))
        print('time until now ='+str(power_calc_time - start))
        count+=1
        
    spectra(f_array,power_array1,label = 'including interaction')
    spectra(f_array,power_array2,label = 'no interaction')
    end = time.time()
    print('time took = ' + str(end - start))
    return power_array1,power_array2, data_ar

def spectra(f_array,power_array,label = '', f = cst.e*0.7/cst.electron_mass/(2*pi)):
    plt.plot(f_array,power_array,label = label)
    plt.axvline(x = f, color = 'r', label = 'cyclotron frequency')
    plt.title('Power Spectra',y=1.08)
    plt.xlabel('microwave frequency (Hz)')
    plt.ylabel('power')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()
    


# In[ ]:




