#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 09:17:09 2022

@author: sarasaib
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.constants as cst
from sympy import symbols, Eq, solve, Matrix, linsolve
import random 

import time
start_time = time.time()

#calculating electric field
eps0 = 8.8541878128E-12
k = 1/(4*math.pi*eps0)


# In[4]:

def circular_microwave_zt(r,t,lamda = 1E-3):  #refer to my derivation in paper "Microwave derivation 2"
    z = r[:,2:3]
    
    c = cst.c
    #lamda = 1E-3 #(set 0.001 to 0.3 m for microwave wavelength)
    power = 10E-3  #10dbm=10mW is standard in ECR paper
    area = math.pi*(2E-2)**2 #2cm radius
    frac = 1/math.sqrt(2)
    E_0 = np.sqrt(power/(2*eps0*area*c))*1000 #Readme this is a huge exaggeration. I just added *1000
    Ex = E_0 * (frac + math.sqrt(1 - frac**2))* np.cos(2*math.pi/lamda* (z - c*t))
    Ey = -E_0 * (frac - math.sqrt(1 - frac**2))* np.sin(2*math.pi/lamda* (z - c*t))
    Ez = z*0
    E = np.concatenate((Ex,Ey,Ez),axis=1)
    
    '''
    freq = c/lamda
    w = circular_microwave_zt_freq()
    if t%0.2==0: 
        print('freq',freq)
        print('microwave_omega', w)
        print('E_0',E_0)   
    '''
    return E


def circular_microwave_zt_freq(lamda = 1E-3):
    #(set lamda 0.001 to 0.3 m for microwave wavelength)
    c = cst.c
    freq = c/lamda
    #w = 2*math.pi*freq
    return freq


# This is for seeing the graph of microwave heating

# In[5]:


t = 0
micro=[]
time_track=[]

#later, add time frame to insert microwave to heat up plasma
def circular_microwaveZT_graph(r,t):
    dt = 0.0000001
    t = 0
    E_micro = []
    time = []
    while t <= 0.01:
        #print(t)
        m = circular_microwave_zt(r,t)
        #print(m)
        E_micro.append(m)
        time.append(t)
        t += dt
    E_microwave = np.array(E_micro)

    plt.subplot(1,2,1)
    plt.plot(time,E_microwave[:,0:1])
    plt.plot(time,E_microwave[:,1:2])
    plt.xlabel("time")
    plt.ylabel("Microwave amplitude")

    plt.subplot(1,2,2)
    plt.plot(E_microwave[:,0:1],E_microwave[:,1:2])
    plt.xlabel("Microwave x")
    plt.ylabel("Microwave y")
    plt.gca().set_aspect('equal')


def microwave_graph(r,T):
    t = 0
    micro=[]
    time=[]

    while t<T:
        micro.append(microwave(r,t))
        time.append(t)
        t += 1E-5
    print('micro size',np.shape(micro))
    #print('micro',micro)
    plt.plot(time,np.array(micro)[:,0])
    plt.show()
    
def circular_microwaveZT_graph(r,T):
    t = 0
    micro=[]
    time=[]

    while t<T:
        micro.append(circular_microwave_zt(r,t))
        time.append(t)
        t += 1E-5
    plt.plot(time,np.array(E_microwave)[:,:,0:1][:,0])
    plt.plot(time,np.array(E_microwave)[:,:,1:2][:,0])
    plt.xlabel("time")
    plt.ylabel("Microwave amplitude")
    plt.show()    
    

#Kinetic Energy vs. time graph
def total_kinetic(m,v):
    return np.sum(1/2*m*v**2)

#Kinetic parallel to B field (assuming B is in z direction only)
def par_kinetic(m,v):
    v_par = v[:,2:3]
    K_par = np.sum(1/2*m*v_par**2)
    return K_par

#Kinetic perpendicular to B field (assuming B is in z direction only)
def perp_kinetic(m,v):
    v_perp = v[:,0:2]
    K_perp = np.sum(1/2*m*v_perp**2)
    return K_perp

def trap_potential(r):
    k_2 = 5000
    r=np.copy(r)
    r[:,0:1],r[:,1:2],r[:,2:3] = 1/4*np.square(r[:,0:1]),1/4*np.square(r[:,1:2]), -1/2*np.square(r[:,2:3])
    U = q*k_2*np.sum(r)
    return U

def interaction_potential(dr_mag_array):
    U = k*q/dr_mag_array
    return U


# Calculation of trap potential is done as follows:
# E = k*(z -1/2x -1/2y)

# In[8]:


def E_trap(r):
    k_2 = 5000
    r=np.copy(r)
    r[:,0:1],r[:,1:2],r[:,2:3] = -1/2*r[:,0:1],-1/2*r[:,1:2],r[:,2:3]
    E= r*k_2
    return E


# In[9]:
q = -1.6E-19 #not sure what to do with the increase in speed and the particle reaching the speed higher than c
m = 9.11E-31
N = 1 # number of particles

mass_array = np.transpose(np.ones(N))*m
mass_array = mass_array[:,None]   #making an array of N x 1

q_array = np.transpose(np.ones(N))*q
q_array = q_array[:,None]

E_ext = np.array([0,0,0])  #needs to be updated according the penning trap potential
Bmag = 0.7
B_ext = np.array([0,0,1]) *Bmag

#v_array_old = np.array([[10,0,0]])
v_array_old = np.ones(shape = (N,3))*14215 #for N particles with v = mean velocity = 24600

v_array_new = np.zeros(shape = (N,3))

if N==2:
    r_array_old = np.array([[1,1,1],[-1,-1,-1]])  #for 2 particles
#for 1 particle only:
r_array_old = np.array([[0.0001,0.0001,0.0001]])
r_array_new = np.zeros(shape = (N,3))
#calculating B field
B_array = B_ext

inj = 0

def initial_conditions(N=2): #number of particles
    q = -1*cst.e #not sure what to do with the increase in speed and the particle reaching the speed higher than c
    m = cst.m_e
    
    mass_array = np.transpose(np.ones(N))*m
    mass_array = mass_array[:,None]   #making an array of N x 1
    
    q_array = np.transpose(np.ones(N))*q
    q_array = q_array[:,None]
    
    E_ext = np.array([0,0,0])  #needs to be updated according the penning trap potential
    Bmag = 0.7
    B_ext = np.array([0,0,1]) *Bmag
    
    #v_array_old = np.array([[10,0,0]])
    v_array_old = np.ones(shape = (N,3))*14215 #for N particles with v = 0
    print('v_array_old',v_array_old)
    
    v_array_new = np.zeros(shape = (N,3))
    
    #r_array_old = np.array([[1,0,0],[-1,0,0]])*8.37E-6 #for 2 particles when intU ~ KE
    #for 1 particle only:
    r_array_old = np.random.uniform(low=0.0, high=1.0, size=(2,3))*8.37E-6   #uniformly distributed array between 0 and 8.37E-6 
    print('r_array_old',r_array_old)
    r_array_new = np.zeros(shape = (N,3))
    #calculating B field
    B_array = B_ext
    print('B_array = ',B_array)
    return [N, q_array, mass_array, E_ext, B_array, r_array_old, r_array_new, v_array_old, v_array_new] 

def energy_investigation(injection_time = inj,lamda=cst.c/(cst.e*0.7/cst.m_e/(2*math.pi)), plot=True):
    [N, q_array, mass_array, E_ext, B_array, r_array_old, r_array_new, v_array_old, v_array_new]=initial_conditions()
    [v_track, r_track, t_track, Time_max, E_track, totalKE, trapU, intU]=calc_rvE(N, q_array, mass_array, E_ext, B_array, r_array_old, r_array_new, v_array_old, v_array_new,lamda,injection_time)
    if plot:
        plot_totE(t_track, Time_max, totalKE, trapU, intU=np.zeros(np.size(t_track)))
    #plt.subplot(223)
    #plot_potential(t_track,trapU,intU)
    return [N,v_track, r_track, t_track, Time_max, E_track, totalKE, trapU, intU]

def ini_fin(totalKE,trapU, intU):
    totalU = trapU + intU
    iniKE = totalKE[:20]
    finKE = totalKE[-20:]
    iniU = totalU[:20]
    finU = totalU[-20:]
    print('initialKE = ', totalKE[:20])
    print('finalKE = ', totalKE[-20:])
    print('initialU = ', totalU[:20])
    print('finalU = ', totalU[-20:])
    return [iniKE, finKE, iniU, finU]
    
#v_array_old = ranmdom.choices()
def cyclotron_freq(q,B_array,m):
    w_c = q*B_array/m
    return w_c

def step_Max(q,B,m):
    w_c = cyclotron_freq(q,B,m)
    f_c = np.linalg.norm(w_c/(2*math.pi))
    #print('ECR freq f_c = ', f_c)
    step_max = 1/f_c
    if f_c == 0:
        step_max = 1e-10
        print('fc=0')
    #print('step_max = ', step_max)
    return step_max

current_time = time.time()

#calculate forces between particles, velocities, and position

microwave_period = 0.5 #between 0 and 3/4
def calc_rvE(N, q_array, mass_array, E_ext, B_array, r_array_old, r_array_new, v_array_old, v_array_new,lamda,injection_time = inj):
    step_max = step_Max(q,B_array,mass_array)   #this is the full cyclotron rotation period

    start_op_time = time.time()
    #delta_t = step_max/100
    delta_t = step_max/50
    T = 0
    #Time_max = 1.05E-6
    Time_max = delta_t#2500*step_max
    
    dr_array = np.array([np.zeros(shape=(N-1,3)),]*N)
    dr_mag_array = np.zeros(shape=(N,1))
    E_array = np.array([np.array([0.,0.,0.]) for i in range (N)])   #initialized E_array
    #E_array_new = np.array([np.array([0.,0.,0.]) for i in range (N)])

    E_track = []
    v_track = []
    r_track = []
    t_track = []
    trapU = []
    interactionU = []
    totalKE = []
    parE = []
    perpE = []

    test_r_final = []  # this list stores final positions after each loop for comparison
    test_t_final = []
    
    
    print('---',N, q_array, mass_array, E_ext, B_array, 'r_old =',r_array_old, r_array_new, v_array_old, v_array_new,lamda,injection_time)
    count  = 0
    while(T<Time_max):#0):#0.5*1/f_c):
        count += 1
        if T==5*delta_t:
            print(T)
        for i in range(N):
        
            if N == 1:
                break
            r_i = r_array_old[i]
            #print('r_i',r_i)
            r_others = np.append(r_array_old[0:i,:],r_array_old[i+1:N,:],axis = 0)  #get rid of the self particle from the array
            #print('r_others=',r_others)
            dr = -1*(r_others - r_i)
            dr_mag = np.linalg.norm(dr)
            E_others = k*q*dr/(dr_mag**3)
            E_i = E_ext + np.sum(E_others, axis = 0)  #sums up all the E-field from other particles and external E-field
            #store the calculated dr in array of size N
            #print(dr)
            
            dr_array[i] = dr
            dr_mag_array[i] = dr_mag

            E_array[i] = E_i 
           
        micro_start = (Time_max - injection_time)/2
        micro_end = micro_start + injection_time
        if ((T> micro_start) and (T<(micro_end))):  #turn on microwave from t =  T/2 to 3T/4 #README if possible, change this to a period I can extract in other codes
            E_microwave = circular_microwave_zt(r_array_old,T-micro_start,lamda)
            #print('----')
            #print(E_microwave)
        else:
            E_microwave = 0
        #print(E_microwave)
        E_array_new = np.copy(E_array) + E_microwave + np.copy(E_trap(r_array_old))
        E_track.append(np.copy(E_array) + E_microwave + np.copy(E_trap(r_array_old)))
 

        #At the end of the array, we have as a main thing: E-field whose rows correspond to each particle
        totalKE.append(total_kinetic(m,v_array_old))
        trapU.append(trap_potential(r_array_old))
        if N>1:
            interactionU.append(interaction_potential(dr_mag_array))
        parE.append(par_kinetic(m,v_array_old))
        perpE.append(perp_kinetic(m,v_array_old))

        v_minus = v_array_old + delta_t/2*(q_array/mass_array)*E_array_new
        
        for i in range(N): 
            
            c = float(-(q_array[i]*delta_t)/(2*mass_array[i]))

            B1= B_array[0]
            B2 = B_array[1]
            B3 = B_array[2]

            a1 = np.array([[1, c*B3, -c*B2], [-c*B3, 1, -c*B1], [c*B2, c*B1, 1]])
            b1 = np.array(-c*np.cross(v_minus[i],B_array) + v_minus[i])
            v_plus =np.linalg.solve(a1, b1)

            v_array_new[i] = v_plus + (q_array[i]*delta_t)/(2*mass_array[i])*E_array_new[i]
        
        r_array_new = v_array_new * delta_t + r_array_old
        #print(r_array_old)
        v_array_old = v_array_new
        r_array_old = r_array_new
        v_track.append(np.copy(v_array_old))
        r_track.append(r_array_old)
        t_track.append(T)

        T += delta_t

    print('---end of calculation for all particles---')
    totalKE = np.array(totalKE)
    interactionU = np.array(interactionU)

    if N == 1:
        interactionU = totalKE*0   #just to make an array of same size as totalKE but elements all being 0 for no interaction
    #print(interactionU[0:10])

    intU = [np.linalg.norm(interactionU[i]) for i in range(np.size(totalKE))] 

    totalE = np.array(totalKE) + np.array(trapU) + np.array(intU)
    totalU = np.array(trapU) + np.array(intU)
    trapU = np.array(trapU)
    v_track = np.array(v_track)
    r_track = np.array(r_track)
    t_track = np.array(t_track)
    E_track = np.array(E_track)

    current_time = time.time()-start_time
    time_took = time.time() - start_op_time

    print('current time = ', current_time)
    #print('time took = ', time_took)
    print('total points',count, np.size(t_track))
    return [v_track, r_track, t_track, Time_max, E_track, totalKE, trapU, intU]



# In[12]:


#plotting x vs y position
def plot_xy(r_track, v_track):  #readme needs modifying
    start_op_time = time.time()
    fig, (ax0,ax01,ax02) = plt.subplots(1,3)
    fig.tight_layout()

    r1_track = r_track[:,0,:]
    v1_track = v_track[:,0,:]
    ##get_ipython().run_line_magic('store', 'r1_track')
    ##get_ipython().run_line_magic('store', 'v1_track')
    x1 = r1_track[:,0]
    y1 = r1_track[:,1]
    size = np.size(y1)
    #ax1.margins(-0.1,-0.1)
    print(np.size(x1))
    #1st particle
    #ax01 = plt.subplot(132)
    ax0.plot(x1,y1,'r-')
    ax0.set_xlabel('x (m)')
    ax0.set_ylabel('y (m)')
    ax0.title.set_text('2 particles pos')

    ax01.plot(x1[0:1000],y1[0:1000],'-')
    ax01.set_xlabel('x (m)')
    ax01.title.set_text('1st x-y pos')
    ax01.text(0.5, 0.5, '1st 1000 points', horizontalalignment='center',
     verticalalignment='center', transform=ax01.transAxes)
    #plt.gca().set_aspect('equal')

    '''
    if N==2:
        r2_track = r_track[:,1,:]
        v2_track = v_track[:,1,:]
        ##get_ipython().run_line_magic('store', 'r2_track')
        ##get_ipython().run_line_magic('store', 'v2_track')
        x2 = r2_track[:,0]
        y2 = r2_track[:,1]
        ax0.plot(x2,y2)
        ax02.plot(x2[0:1000],y2[0:1000],'-')
        ax02.set_xlabel('x (m)')
        ax02.title.set_text('2nd x-y pos')
       # plt.gca().set_aspect('equal')
       '''
    plt.show()

    current_time = time.time()-start_time
    time_took = time.time() - start_op_time
    print('current time = ', current_time)
    print('time took = ', time_took)


# In[13]:


#plotting z vs t
def plot_position(t_track,r_track):  #readme needs modifying
    start_op_time = time.time()

    r1_track = r_track[:,0,:]
    
    plt.subplot(3,1,1)
    x1 = r1_track[:,0]
    if N == 2:
        r2_track = r_track[:,1,:]
        x2  = r2_track[:,0]
    plt.xlabel('time (s)')
    plt.ylabel('x (m)')
    plt.plot(t_track,x1)
    plt.title('1st particle x vs t')
    plt.show()

    #plotting y vs t
    plt.subplot(3,1,2)
    y1 = r1_track[:,1]
    if N == 2:
        y2 = r2_track[:,1]
    plt.xlabel('time (s)')
    plt.ylabel('y (m)')
    plt.plot(t_track,y1)
    plt.title('1st particle y vs t,',y=1.1)
    plt.show()

    #plotting z vs t
    plt.subplot(3,1,3)
    z1 = r1_track[:,2]
    if N == 2:
        z2 = r2_track[:,2]
    plt.xlabel('time (s)')
    plt.ylabel('z (m)')
    plt.plot(t_track,z1)
    plt.title('1st particle z vs t',y=1.1)
    plt.show()

    current_time = time.time()-start_time
    time_took = time.time() - start_op_time
    print('current time = ', current_time)
    print('time took = ', time_took)


# In[14]:


#plotting Vx vs t position
def plot_Vx_t(t_track, v_track):
    start_op_time = time.time()

    fig1, (ax1,ax11) = plt.subplots(1,2)
    fig1.tight_layout()
    l = 500001
    v1_track = v_track[:,0,:]
    vx1 = v1_track[:,0]
    vy1 = v1_track[:,1]
    vz1 = v1_track[:,2]
    print(np.size(vy1))
    ax1 = plt.subplot(221)
    #ax1.margins(-0.1,-0.1)
    Vx_graph, = ax1.plot(t_track[0:l],vx1[0:l],'-',label = 'Vx')
    Vy_graph, = ax1.plot(t_track[0:l],vy1[0:l],'-',label = 'Vy')
    Vz_graph, = ax1.plot(t_track[0:l],vz1[0:l],'-',label = 'Vz')

    plt.xlabel('t')
    plt.ylabel('v (m/s)')
    ax1.legend(handles=[Vx_graph, Vy_graph,Vz_graph])
    plt.title('1st particle velocities',y=1.05)

    ax11 = plt.subplot(222)
    Vtotal_graph, = ax11.plot(t_track[0:l], np.sqrt(vx1**2 + vy1**2 + vz1**2)[0:l],'g-', label = 'V total magnitude')
    plt.xlabel('t')
    plt.ylabel('v (m/s)')
    ax11.legend(handles=[Vtotal_graph])
    plt.title('1st particle v mag',y=1.05)
    #plt.gca().set_aspect('equal')

    #plotting Vy vs t position
    if N==2:
        fig2, (ax2,ax22) = plt.subplots(1,2)
        fig2.tight_layout()
        ax2 = plt.subplot(223)
        v2_track = v_track[:,1,:]
        vx2 = v2_track[:,0]
        vy2 = v2_track[:,1]
        vz2 = v2_track[:,2]
        print(np.size(vy2))
        plt.title('2nd particle velocities',y=1.05)

        #ax2.margins(-0.1,-0.1)
        Vx_graph, = ax2.plot(t_track[0:l],vx2[0:l],'-',label = 'Vx')
        Vy_graph, = ax2.plot(t_track[0:l],vy2[0:l],'-',label = 'Vy')
        Vz_graph, = ax2.plot(t_track[0:l],vz2[0:l],'-',label = 'Vz')
        plt.xlabel('t')
        plt.ylabel('v (m/s)')
        ax2.legend(handles=[Vx_graph, Vy_graph, Vz_graph])

        ax22 = plt.subplot(224)
        Vtotal_graph, = ax22.plot(t_track[0:l], np.sqrt(vx2**2 + vy2**2 + vz1**2)[0:l],'g-', label = 'V total magnitude')
        plt.xlabel('t')
        plt.ylabel('v (m/s)')
        ax22.legend(handles=[Vtotal_graph])
        plt.title('2nd particle v mag',y=1.05)


    plt.show()

    current_time = time.time()-start_time
    time_took = time.time() - start_op_time
    print('current time = ', current_time)
    print('time took = ', time_took)


# In[15]:


#plotting guiding center
def plot_guidingR(w_c, ):
    start_op_time = time.time()
    fig3, (ax3,ax4,ax5) = plt.subplots(1,3)
    fig3.tight_layout()

    if np.linalg.norm(w_c) == 0:
        w_c = 1e-10
    Rx1 = x1 - vy1/np.linalg.norm(w_c)
    Ry1 = y1 - vx1/np.linalg.norm(w_c)
    Rz1 = z1

    ax3 = plt.subplot(131)
    ax4 = plt.subplot(132)
    #ax1.margins(-0.1,-0.1)
    R1_graph, = ax3.plot(Rx1,Ry1,label = 'particle 1')
    ax4.plot(Rx1[0:1000],Ry1[0:1000])
    ax4.title.set_text('1st guiding center')

    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.legend(handles=[R1_graph])
    ax3.title.set_text('guiding centers')
    #plt.gca().set_aspect('equal')

    #plotting Vy vs t position
    if N==2:
        Rx2 = x2 - vy2/np.linalg.norm(w_c)
        Ry2 = y2 - vx2/np.linalg.norm(w_c)
        Rz2 = z2
        ax5 = plt.subplot(133)
        ax5.title.set_text('2nd guiding center')

        #ax2.margins(-0.1,-0.1)
        R2_graph, = ax3.plot(Rx2,Ry2,label = 'particle 2')
        ax5.plot(Rx2[0:1000],Ry2[0:1000])
        ax5.set_xlabel('x (m)')
        #plt.ylabel('position (m)')
        ax3.legend(handles=[R2_graph])

    plt.show()

    current_time = time.time()-start_time
    time_took = time.time() - start_op_time
    print('current time = ', current_time)
    print('time took = ', time_took)
    '''
    for plotting R vs t
    ax3 = plt.subplot(121)
    #ax1.margins(-0.1,-0.1)
    Rx1_graph, = ax3.plot(t_track[0:200],Rx1[0:200],'-',label = 'Rx')
    Ry1_graph, = ax3.plot(t_track[0:200],Ry1[0:200],'-',label = 'Ry')
    Rz1_graph, = ax3.plot(t_track[0:200],Rz1[0:200],'-',label = 'Rz')

    plt.xlabel('t')
    plt.ylabel('position (m)')
    ax3.legend(handles=[Rx1_graph, Ry1_graph,Rz1_graph])
    plt.title('1st particle guiding center')
    #plt.gca().set_aspect('equal')

    #plotting Vy vs t position
    if N==2:
        Rx2 = x2 - vy2/np.linalg.norm(w_c)
        Ry2 = y2 - vx2/np.linalg.norm(w_c)
        Rz2 = z2
        ax4 = plt.subplot(122)
        plt.title('2nd particle guiding center')

        #ax2.margins(-0.1,-0.1)
        Rx2_graph, = ax4.plot(t_track[0:200],Rx2[0:200],'-',label = 'Rx')
        Ry2_graph, = ax4.plot(t_track[0:200],Ry2[0:200],'-',label = 'Ry')
        Rz2_graph, = ax3.plot(t_track[0:200],Rz2[0:200],'-',label = 'Rz')
        plt.xlabel('t')
        #plt.ylabel('position (m)')
        ax4.legend(handles=[Rx2_graph, Ry2_graph, Rz2_graph])

    plt.show()

    print('current time = ', time.time()-start_time)
    '''


# In[16]:


#plotting x vs z position
def plot_xz(r_track, v_track): #readme needs modifying
    start_op_time = time.time()

    fig2, ax = plt.subplots()
    fig2.tight_layout()
    r1_track = r_track[:,0,:]
    v1_track = v_track[:,0,:]
    ##get_ipython().run_line_magic('store', 'r1_track')
    ##get_ipython().run_line_magic('store', 'v1_track')
    x1 = r1_track[:,0]
    z1 = r1_track[:,2]

    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    #plt.gca().set_aspect('equal')
    ax.plot(x1,z1)

    plt.subplot(1,3,2)
    if N==2:
        r2_track = r_track[:,1,:]
        v2_track = v_track[:,1,:]
        ##get_ipython().run_line_magic('store', 'r2_track')
        #get_ipython().run_line_magic('store', 'v2_track')
        x2 = r2_track[:,0]
        z2 = r2_track[:,2]
        plt.plot(x2,z2)
        plt.xlabel('x (m)')
        plt.ylabel('z (m)')
        #plt.gca().set_aspect('equal')
        plt.show()



    current_time = time.time()-start_time
    time_took = time.time() - start_op_time
    print('current time = ', current_time)
    print('time took = ', time_took)


# In[17]:


#plotting E_array vs t
def plot_Et(t_track, E_track): #readme needs modifying
    start_op_time = time.time()

    E_1 = E_track[:,0,:]
    #%store E_1
    if N==2:
        E_2 = E_track[:,1,:]
        #%store E_2

    fig1, axs = plt.subplots(1,3)
    fig1.tight_layout()

    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('E field')
    axs[0].set_title('Ex')
    #plt.gca().set_aspect('equal')
    axs[0].plot(t_track,E_1[:,0])

    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('Ey')
    axs[1].set_title('Ey')
    axs[1].plot(t_track,E_1[:,1])

    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('Ez')
    axs[2].set_title('Ez')
    axs[2].plot(t_track,E_1[:,2])

    if N==2:
        plt.plot(t_track,E_2[:,0])

    '''
    plt.subplot(1,3,2)
    plt.xlabel('t (s)')
    plt.ylabel('Ey')
    plt.title('Ey')
    plt.plot(t_track,E_1[:,1])
    if N==2:
        plt.plot(t_track,E_2[:,0])
    #plt.plot(t_track,E_2[:,0])
    #plt.plot(t_track,E_1[:,1])
    #plt.plot(t_track,E_1[:,2])
    '''

    current_time = time.time()-start_time
    time_took = time.time() - start_op_time
    print('current time = ', current_time)
    print('time took = ', time_took)


# This plots kinetic energies

# In[18]:


from operator import add
def plot_KE(t_track, Time_max, totalKE, trapU, intU,l=250000, injection_time = inj): #readme needs modifying
    l =  np.size(t_track)
    start_op_time = time.time()
    totalE = totalKE + trapU + intU
    
    #plot for totalKE
    plt.subplot(121)
    plt.plot(t_track[0:l],totalKE[0:l])
    if injection_time>0:
        micro_start = (Time_max - injection_time)/2
        micro_end = micro_start+injection_time
        plt.axvline(x = micro_start, color = 'r')#, label = 'start of microwave heating')
        plt.axvline(x = micro_end, color = 'r')#, label = 'end of microwave heating')
    plt.ylabel('energy')
    plt.xlabel('time')
    plt.title('Total KE',y=1.05)
    
    #plot for total E
    plt.subplot(122)
    plt.plot(t_track[0:l],totalE[0:l])
    if injection_time>0:
        micro_start = (Time_max - injection_time)/2
        micro_end = micro_start+injection_time
        plt.axvline(x = micro_start, color = 'r')#, label = 'start of microwave heating')
        plt.axvline(x = micro_end, color = 'r')#, label = 'end of microwave heating')
    plt.title('Total E',y=1.05)
    plt.xlabel('time')

    current_time = time.time()-start_time
    time_took = time.time() - start_op_time
    print('current time = ', current_time)
    print('time took = ', time_took)


# In[19]:


#plotting total energy
def plot_totE(t_track, Time_max, totalKE, trapU, intU, l=500000, injection_time = inj): 
    l = np.size(t_track)
    start_op_time = time.time()
    
    totalU = trapU + intU
    totalE = totalKE + totalU
    fig_energy, (ax_energy1, ax_energy2) = plt.subplots(1,2)

    #l = 500001
    ax_energy1 = plt.subplot(121)
    totalKE_graph, = ax_energy1.plot(t_track[0:l],totalKE[0:l],'-',label = 'total KE')
    totalU_graph, = ax_energy1.plot(t_track[0:l],totalU[0:l],'-',label = 'total U')
    #totalE_graph, = ax_energy1.plot(t_track[0:l],totalE[0:l],'-',label = 'total energy')
    #totalE_graph_scaled, = ax_energy1.plot(t_track[0:l],totalE[0:l]*(max(totalKE[0:l])/max(totalE[0:l])),'g--',label = 'total energy enlarged')
    if injection_time>0:
        micro_start = (Time_max - injection_time)/2
        micro_end = micro_start+injection_time
        plt.axvline(x = micro_start, color = 'r')#, label = 'start of microwave heating')
        plt.axvline(x = micro_end, color = 'r')#, label = 'end of microwave heating')
    plt.xlabel('t')
    plt.ylabel('energy')
    ax_energy1.legend(handles=[totalKE_graph, totalU_graph])#, totalE_graph])
    plt.title('1st particle Energy',y=1.05)
    plt.text(0.50, -0.35, "KE initial and final = " + str(totalKE[:3]) +'\n' + str(totalKE[-3:]) 
             + '\n' + "trapU initial and final = " + str(trapU[:3]) +'\n' + str(trapU[-3:])
             + '\n' + "calcualtion time  = " + str(Time_max)
             + '\n Microwave applied for ' + str(injection_time) + ' s'
             , transform=plt.gcf().transFigure, fontsize=14, ha='center', color='blue')

    ax_energy2 = plt.subplot(122)
    plt.plot(t_track[0:l],totalE[0:l],'g-',label = 'total energy')
    #plt.text(1,1,str(lamda))
    plt.xlabel('t')
    plt.ylabel('energy')
    plt.title('1st total energy',y=1.05)

    print('fluctuation in total E: totalE/totalKE = ', max(totalE[0:l])/max(totalKE[0:l]))
    ##ASK about this phenomena of total energy change
    ##it is small compared to the KE and U (order of 10^-5) so it should not be much problem. Is this a computation error?

    current_time = time.time()-start_time
    time_took = time.time() - start_op_time
    #print('current time = ', current_time)
    #print('time took = ', time_took)
    print('initial and final KE = ', [totalKE[:3],totalKE[-3:]])
    print('initial and final trap U = ', [trapU[:3],trapU[-3:]])
    #print('initial and final intU = ', [intU[:3],intU[-3:]])


# In[20]:


#plotting potentials
def plot_potential(t_track, Time_max, trapU, intU, injection_time= inj): #readme needs modifying
    start_op_time = time.time()
    
    if np.size(trapU) == np.size(t_track):
        plt.subplot(121)
        plt.title('V_trap',y=1.05)
        plt.plot(t_track,np.array(trapU))
    if injection_time>0:
        micro_start = (Time_max - injection_time)/2
        micro_end = micro_start+injection_time
        plt.axvline(x = micro_start, color = 'r')#, label = 'start of microwave heating')
        plt.axvline(x = micro_end, color = 'r')#, label = 'end of microwave heating')
        
        plt.xlabel('t')
        plt.ylabel('V')

    if np.size(intU) == np.size(t_track):
        plt.subplot(122)
        plt.title('V_int',y=1.05)
        plt.xlabel('time (s)')
        plt.ylabel('V (J)')
        plt.plot(t_track,np.array(intU)*1e5)
    if injection_time>0:
        micro_start = (Time_max - injection_time)/2
        micro_end = micro_start+injection_time
        plt.axvline(x = micro_start, color = 'r')#, label = 'start of microwave heating')
        plt.axvline(x = micro_end, color = 'r')#, label = 'end of microwave heating')
        plt.xlabel('time (s)')

    current_time = time.time()-start_time
    time_took = time.time() - start_op_time
    print('current time = ', current_time)
    print('time took = ', time_took)


# In[21]:


#plotting potentials
def plot_potential_closedup(t_track, Time_max, trapU, intU):
    start_op_time = time.time()

    plt.subplot(121)
    #plt.title('V_trap')
    plt.plot(t_track[0:1000],np.array(trapU)[0:1000])
    plt.xlabel('t')
    plt.ylabel('V')

    plt.subplot(122)
    #plt.title('V_interaction')
    plt.plot(t_track[0:1000],np.array(intU)[0:1000])
    plt.xlabel('t')

    current_time = time.time()-start_time
    time_took = time.time() - start_op_time
    print('current time = ', current_time)
    print('time took = ', time_took)


