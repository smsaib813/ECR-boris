#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.constants as cst


# In[2]:


import time
start_time = time.time()


# In[3]:


#calculating electric field
eps0 = 8.8541878128E-12
k = 1/(4*math.pi*eps0)
print(k)


# In[4]:


def circular_microwave(t):   #no z dependence
    c = cst.c*1E-10
    lamda = 1E-3 #(set 0.001 to 0.3 m for microwave wavelength)
    power = 10E-3  #10dbm=10mW is standard in ECR paper
    area = math.pi*(2E-2)**2 #2cm radius
    E_0 = np.sqrt(2*power/(eps0*area*c))
    E = [E_0 *math.cos(2*math.pi/lamda*t),E_0 *math.sin(2*math.pi/lamda*t),0]
    freq = c/lamda
    if t%0.2==0: 
        print('freq',freq)
        print('E_0',E_0)
    return E

def circular_microwave_zt(r,t):  #refer to my derivation in paper "Microwave derivation 2"
    z = r[:,2:3]
    
    c = cst.c*1E-10
    lamda = 1E-3 #(set 0.001 to 0.3 m for microwave wavelength)
    power = 10E-3  #10dbm=10mW is standard in ECR paper
    area = math.pi*(2E-2)**2 #2cm radius
    frac = 1/math.sqrt(2)
    E_0 = np.sqrt(power/(2*eps0*area*c))
    Ex = E_0 * (frac + math.sqrt(1 - frac**2))* np.cos(2*math.pi/lamda* (z - c*t))
    Ey = -E_0 * (frac - math.sqrt(1 - frac**2))* np.sin(2*math.pi/lamda* (z - c*t))
    Ez = z*0

    E = np.concatenate((Ex,Ey,Ez),axis=1)
    freq = c/lamda
    if t%0.2==0: 
        print('freq',freq)
        print('E_0',E_0)
    return E


# This is for seeing the graph of microwave heating

# In[5]:


t = 0
micro=[]
time_track=[]

#later, add time frame to insert microwave to heat up plasma
def circular_microwave_graph(t):
    dt = 0.00001
    t = 0
    E_micro = []
    time = []
    while t <= 0.01:
        #print(t)
        m = circular_microwave(t)
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
    
#microwave_graph(np.array([[-1,1,1],[0.01,0.01,0.01]]),1)


# The following is for graphing the energy of the particles. 

# In[ ]:


#Energy vs. time graph


# Calculation of trap potential is done as follows:
# E = k*(z -1/2x -1/2y)

# In[6]:


def E_trap(r):
    E_0 = 10000
    r=np.copy(r)
    r[:,0:1],r[:,1:2],r[:,2:3] = -1/2*r[:,0:1],-1/2*r[:,1:2],r[:,2:3]
    E= r*E_0
    return r*E_0


# In[7]:


q = -1.6E-19 #not sure what to do with the increase in speed and the particle reaching the speed higher than c
m = 9.11E-31
N = 2 # number of particles

mass_array = np.transpose(np.ones(N))*m
mass_array = mass_array[:,None]   #making an array of N x 1
print('mass_array',mass_array)

q_array = np.transpose(np.ones(N))*q
q_array = q_array[:,None]
print('q_array',q_array)

delta_t = 0.0001
E_ext = np.array([0,0,0])  #needs to be updated according the penning trap potential
B_ext = np.array([0,0,1]) *0.7


v_array_old = np.zeros(shape = (N,3)) #for N particles with v = 0
#v_array_old = np.array([[0.00,0,0] for _ in range(N)])
print('v_array_old',v_array_old)

v_array_new = np.zeros(shape = (N,3))

r_array_old = np.array([[0.001,0,0],[-0.001,0,0]])  #for 2 particles
#for 1 particle only:
#r_array_old = np.array([[1,0,0]])
print('r_array_old',r_array_old)

r_array_new = np.zeros(shape = (N,3))


# In[8]:


#calculating B field
B_array = B_ext
print('B_array = ',B_array)

w_c = q*B_array/m
f_c = np.linalg.norm(w_c/(2*math.pi))
print('ECR freq f_c = ', f_c)
Time_max = 1/f_c
print('Time_max = ', Time_max)


# In[57]:


#calculate forces between particles, velocities, and position

from sympy import symbols, Eq, solve, Matrix, linsolve

T = 0
dr_array = np.array([np.zeros(shape=(N-1,3)),]*N)
dr_mag_array = np.zeros(shape=(N,1))
E_array = np.array([np.array([0.,0.,0.]) for i in range (N)])   #initialized E_array
v_track = []
r_track = []
t_track = []

test_r_final = []  # this list stores final positions after each loop for comparison
test_t_final = []

E_track = []
    
while(T<5):#0):#0.5*1/f_c):
    for i in range(N):
        r_i = r_array_old[i]
        r_others = np.append(r_array_old[0:i,:],r_array_old[i+1:N,:],axis = 0)  #get rid of the self particle from the array
        dr = -1*(r_others - r_i)
        dr_mag = np.linalg.norm(dr)
        E_others = k*q*dr/(dr_mag**3)
        E_i = E_ext + np.sum(E_others, axis = 0)  #sums up all the E-field from other particles and external E-field
        #store the calculated dr in array of size N
        dr_array[i] = dr
        dr_mag_array[i] = dr_mag
        E_array[i] = E_i 
    #print('E_array',E_array)
    #E_microwave = [circular_microwave(T),]*int(N)
    
    E_microwave = 0#circular_microwave_zt(r_array_old,T)
    
    #print('E_microwave',E_microwave)
    #print('addition----',E_array + E_microwave)
    E_track.append(np.copy(E_array)+ np.copy(E_trap(r_array_old)) + E_microwave)
    #At the end of the array, we have as a main thing: E-field whose rows correspond to each particle

    v_minus = v_array_old + delta_t/2*(q_array/mass_array)*E_array
    
    for i in range(N): 
        c = float(-(q_array[i]*delta_t)/(2*mass_array[i]))

        B1= B_array[0]
        B2 = B_array[1]
        B3 = B_array[2]

        a1 = np.array([[1, c*B3, -c*B2], [-c*B3, 1, -c*B1], [c*B2, c*B1, 1]])
        b1 = np.array(-c*np.cross(v_minus[i],B_array) + v_minus[i])
        v_plus =np.linalg.solve(a1, b1)
        
        v_array_new[i] = v_plus + (q_array[i]*delta_t)/(2*mass_array[i])*E_array[i]
    
    r_array_new = v_array_new * delta_t + r_array_old
    #print(r_array_old)
    v_array_old = v_array_new
    r_array_old = r_array_new
    #print(r_array_old)
    v_track.append(np.copy(v_array_old))
    r_track.append(r_array_old)
    t_track.append(T)
    
    T += delta_t
    
print('end of updating v and r')

v_track = np.array(v_track)
r_track = np.array(r_track)
t_track = np.array(t_track)
E_track = np.array(E_track)

print('current time = ', time.time()-start_time)


# In[58]:


#print(E_track)
#%store E_track


# In[59]:


#print(v_track)


# In[60]:


#print(r_track)


# In[61]:


#plotting x vs y position
r1_track = r_track[:,0,:]
v1_track = v_track[:,0,:]
get_ipython().run_line_magic('store', 'r1_track')
get_ipython().run_line_magic('store', 'v1_track')
x1 = r1_track[:,0]
y1 = r1_track[:,1]

ax1 = plt.subplot(131)
ax1.margins(-0.1,-0.1)
ax1.plot(x1,y1)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
#plt.gca().set_aspect('equal')

#plt.subplot(1,3,2)
if N==2:
    r2_track = r_track[:,1,:]
    v2_track = v_track[:,1,:]
    get_ipython().run_line_magic('store', 'r2_track')
    get_ipython().run_line_magic('store', 'v2_track')
    x2 = r2_track[:,0]
    y2 = r2_track[:,1]
    ax1.plot(x2,y2)
    #plt.gca().set_aspect('equal')

plt.show()

print('current time = ', time.time()-start_time)


# In[ ]:


#plotting x vs z position
plt.subplot(1,3,1)
r1_track = r_track[:,0,:]
v1_track = v_track[:,0,:]
get_ipython().run_line_magic('store', 'r1_track')
get_ipython().run_line_magic('store', 'v1_track')
x1 = r1_track[:,0]
z1 = r1_track[:,2]

plt.xlabel('x (m)')
plt.ylabel('z (m)')
#plt.gca().set_aspect('equal')
plt.plot(x1,z1)

plt.subplot(1,3,2)
if N==2:
    r2_track = r_track[:,1,:]
    v2_track = v_track[:,1,:]
    get_ipython().run_line_magic('store', 'r2_track')
    get_ipython().run_line_magic('store', 'v2_track')
    x2 = r2_track[:,0]
    z2 = r2_track[:,2]
    plt.plot(x2,z2)
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    #plt.gca().set_aspect('equal')
    plt.show()



print('current time = ', time.time()-start_time)


# In[ ]:


#plotting E_array vs t 
E_1 = E_track[:,0,:]
get_ipython().run_line_magic('store', 'E_1')
if N==2:
    E_2 = E_track[:,1,:]
    get_ipython().run_line_magic('store', 'E_2')
    
plt.subplot(1,3,1)
plt.xlabel('t (s)')
plt.ylabel('E')
#plt.gca().set_aspect('equal')
plt.plot(t_track,E_1[:,0])
plt.plot(t_track,E_2[:,0])
#plt.plot(t_track,E_2[:,0])
plt.title('Ex')
#plt.plot(t_track,E_1[:,1])
#plt.plot(t_track,E_1[:,2])


# In[34]:


#plotting E_array vs z 

plt.plot(z1,E_1[:,0])
plt.plot(z2,E_2[:,0])
#plt.gca().set_aspect('equal')
plt.title("Ex vs z")
plt.xlabel("z")
plt.ylabel("Ex")


# In[17]:


#plotting z vs t
plt.subplot(3,1,3)
z1 = r1_track[:,2]
plt.xlabel('time (s)')
plt.plot(t_track,z1)
plt.show()


# In[18]:


#testing the final positions of the particle after each loop
'''
import statistics
x = np.array(test_r_final)[:,:,0]
y = np.array(test_r_final)[:,:,1]
z = np.array(test_r_final)[:,:,2]
t = test_t_final
print('r',test_r_final)
print('time',test_t_final)
plt.subplot(1,2,1)
plt.scatter(x,y)
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.subplot(1,2,2)
plt.scatter(t,z)
plt.xlabel('t (s)')
plt.ylabel('z (m)')
print(np.std(x),np.std(y),np.std(z))
'''


# In[19]:


#do this to avoid this notebook freezing and not opening

clear = input("clear output? y/n:")

from IPython.display import clear_output
clear_output(wait=True)
print("output cleared")

