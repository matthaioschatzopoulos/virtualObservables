#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:17:31 2022

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
import data_generation as d



### Parameters initiallization ###
Iterations=10000
N_samples=100
eps=10**(-8)
lrate=10**(-5)
lr1=10**(-5)
lr2=10**(-5)
lr3=10**(-5)
perc=0.5
M=2 ### By definition because we only have w0 and w1



mq=np.ones((M,1))
L=np.array([[0.5,1],[0.2,0.7]])
phi=np.zeros((M,1))
b=np.zeros((M,1))
A=np.zeros((M,M))

dT_m=np.zeros((2,Iterations))
dT_L=np.zeros((4,Iterations))
for j in range(0,d.N_train):
    phi[0,0]=1
    phi[1,0]=d.x_data[j,0]
    b=b+d.y_data[j,0]*phi
    A=A+np.outer(phi,phi)


for ii in range(0,Iterations):

    ep=np.zeros((2,N_samples))
    ep_exp=np.zeros((M,1))
    ep_cov=np.zeros((M,M))

    ep[0,:]=np.random.normal(0,1,N_samples)
    ep[1,:]=np.random.normal(0,1,N_samples)

    for k in range(0,N_samples):
        ep_exp[0,0]=ep_exp[0,0]+ep[0,k]
        ep_exp[1,0]=ep_exp[1,0]+ep[1,k]
        for i in range(0,M):
            for j in range(0,M):
                ep_cov[i,j]=ep_cov[i,j]+ep[i,k]*ep[j,k]

    ep_exp=1/N_samples*ep_exp
    ep_cov=1/N_samples*ep_cov
    
    dT1_m=-d.beta/2*(-2*b+np.matmul(A,2*mq+2*np.matmul(L,ep_exp)))
    dT1_L=-d.beta/2*(-2*np.outer(ep_exp,b)+np.matmul(A,(2*np.inner(mq, ep_exp)+2*np.inner(L,ep_cov)))) ### careful with transpose matrix L
    dT2_m=-d.alpha*mq
    dT2_L=-d.alpha*np.sum(ep_exp)*np.identity(M)
    dT3_m=np.zeros((M,1))
    dT3_L=np.linalg.inv(L)
    
    #dT_m[:,ii]=dT1_m+dT2_m+dT3_m
    dT_m[0,ii]=dT1_m[0,0]+dT2_m[0,0]+dT3_m[0,0]
    dT_m[1,ii]=dT1_m[1,0]+dT2_m[1,0]+dT3_m[1,0]
    dT_L[0,ii]=dT1_L[0,0]+dT2_L[0,0]+dT3_L[0,0]
    dT_L[1,ii]=dT1_L[1,0]+dT2_L[1,0]+dT3_L[1,0]
    dT_L[2,ii]=dT1_L[0,1]+dT2_L[0,1]+dT3_L[0,1]
    dT_L[3,ii]=dT1_L[1,1]+dT2_L[1,1]+dT3_L[1,1]
    
   
    
    mq=mq+np.array([[dT_m[0,ii]],[dT_m[1,ii]]])*lrate
    L=L+np.array([[dT_L[0,ii],dT_L[2,ii]],[dT_L[1,ii],dT_L[3,ii]]])*lrate
    S=np.matmul(L,np.transpose(L))
 
    
