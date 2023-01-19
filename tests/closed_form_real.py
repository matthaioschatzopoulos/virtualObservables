#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 18:24:19 2022

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
Iterations=100
N_samples=100
eps=10**(-8)
lrate=10**(-5)
lr1=10**(-5)
lr2=10**(-5)
lr3=10**(-5)
perc=0.5
M=2 ### By definition because we only have w0 and w1



mq=np.zeros((M,1))
phi=np.zeros((M,1))
b=np.zeros((M,1))
A=np.zeros((M,M))

for j in range(0,d.N_train):
    phi[0,0]=1
    phi[1,0]=d.x_data[j,0]
    b=b+d.y_data[j,0]*phi
    A=A+np.outer(phi,phi)
    
mq=np.matmul(np.linalg.inv(d.beta*A+d.alpha*np.identity(M)),d.beta*b)
S=np.linalg.inv(d.beta*A+d.alpha*np.identity(M))


### Testing derivatives for MC ###
m_in=np.ones((M,1))
L=np.array([[0.5,1],[0.2,0.7]])
cdT1_m=-d.beta/2*(-2*b+2*np.matmul(A,m_in))
cdT1_L=-d.beta/2*np.matmul(A,2*L)
