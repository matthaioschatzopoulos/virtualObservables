#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 12:42:42 2022

@author: xatzman1999
"""
### Importing Libraries ###
import numpy as np
import math
import matplotlib as plt
import random 
import pandas as pd
from scipy.stats import multivariate_normal
from numpy.linalg import inv
import matplotlib.pyplot as plt

### Data Generation ###
#import data_generation as d
random.seed(42) #always the same random numbers for validation
N_train=50
beta=20
alpha=10
fixed=1

w_true=[-0.3,0.5]

x_data=np.random.uniform(0,1,N_train)
x_data=np.reshape(x_data,(N_train,1))

if fixed==1:
    df=pd.read_csv('x_data',delimiter='\s+')
    x_data=np.reshape(np.array(df),(N_train,1))
    #x_data=np.squeeze(x_data,1)
    df2=pd.read_csv('noise',delimiter='\s+')
    noise=np.reshape(np.array(df2),(N_train,1))
    #noise=np.squeeze(noise,1)

Phi=np.concatenate((np.ones((N_train,1)), x_data), axis=1)


y_data=np.zeros((N_train,1))
if fixed==0:
    noise=np.zeros((N_train,1))
for i in range(1,N_train):
    if fixed==0:
        noise[i,0]=np.random.normal(0,1/beta)
    y_data[i,0]=x_data[i]*w_true[1]+w_true[0]+noise[i,0]



#phi_data=[np.ones(N_train,1),x_data]

#noise=np.squeeze(noise,1)
#x_data=np.squeeze(x_data,1)
#for i in range(0,N_train):
#    print(x_data[i])

### Data Generation ###

### Prior Distribution Parameters ###
Lambda_inv=1/alpha*np.eye(2)
Lambda=alpha*np.eye(2)
mu=np.zeros((2,1))


### Likelihood Parameters ###
L_inv=1/beta*np.eye(N_train)
L=beta*np.eye(N_train)
A=Phi

#x=np.transpose(np.ones((d.N_train,1))*ww1[0,j]) #No need for calculating x

### Extra Required Parameters ###
b=np.zeros((N_train,1))
y=y_data
     


### Calculating the Posterior Analytically ###
sigmaa=inv((Lambda+np.matmul(np.matmul(np.transpose(A),L),A)))
muu=np.matmul(sigmaa,(np.matmul(np.matmul(np.transpose(A),L),(y-b))+np.matmul(Lambda,mu)))
Posterior=multivariate_normal(np.squeeze(muu,1),sigmaa)

### Creating the Parameters grid for plotting ###
w0=np.linspace(-1, 1,1001)
w1=np.linspace(-1, 1,1001)
ww0, ww1=np.meshgrid(w0,w1)
pos = np.dstack((ww0, ww1))

### Ploting ###
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.grid()
plot1=plt.contourf(w0, w1, Posterior.pdf(pos))
plt.colorbar()
posterior_vals=Posterior.pdf(pos)

fig2.savefig('analytical.png', dpi=150)

res=Posterior.cdf([1,1]) ### cdf at 1,1 is equal to 1 as expected

aaa=np.sqrt(sigmaa[0,0])
bbb=sigmaa[0,1]/aaa
ccc=np.sqrt(sigmaa[1,1]-bbb**2)

print(muu)
print(sigmaa)
print(N_train)