### Importing Libraries ###
import numpy as np
import math
import random
import pandas as pd
from scipy.stats import multivariate_normal
from numpy.linalg import inv
import matplotlib.pyplot as plt

### Import Pyro/Torch Libraries ###
import argparse
import torch
import torch.nn as nn
from torch.nn.functional import normalize

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.optim import Adam

pyro.set_rng_seed(1)

from pyro.nn import PyroSample
from torch import nn
from pyro.nn import PyroModule
import pyro.optim as optim
import os
import logging
from torch.distributions import constraints
smoke_test = ('CI' in os.environ)
from torch.distributions import constraints

### Data Generation ###

random.seed(42) #always the same random numbers for validation
N_train=500
beta=20
alpha=10
fixed=0
M=2 ### By definition because we only have w0 and w1

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

x_data=np.squeeze(x_data,1)
y_data=np.squeeze(y_data,1)


### Data Generation ###




### Main program ###



beta=math.sqrt(20)
alpha=10


def model(x_data,y_data):
    w=pyro.sample("w", dist.MultivariateNormal(loc=torch.zeros(1,2),
                        covariance_matrix=(1./alpha)*torch.eye(2,2))) ### Prior distribution P(w)
    mean = w[0,0]+x_data*w[0,1]
    with pyro.plate("data", len(y_data)):
            pyro.sample("y_data", dist.Normal(mean, (1./beta)), obs=y_data)

"""
def model():
    x = pyro.sample("x", dist.Normal(0,1))
    pred = 0.5*x+1
    y = pyro.sample("y", dist.Normal(pred,0.01))
    w=pyro.sample("w", dist.MultivariateNormal(loc=torch.zeros(1,2),
                        covariance_matrix=(1./alpha)*torch.eye(2,2)))
    mean = w[0,0]+x*w[0,1]### Prior distribution P(w)
    pyro.sample("y_data", dist.Normal(mean, (1./beta)), obs=y)
"""
def guide(x_data,y_data):
    mq = pyro.param('mq', torch.zeros(1,2))
    Sigma = pyro.param('Sigma', torch.tensor([[0.05,0.01],[0.02,0.01]]), constraints.positive_definite)
    w = pyro.sample("w", dist.MultivariateNormal(loc=mq, covariance_matrix=Sigma))

"""
def guide():
    x = pyro.sample("x", dist.Normal(0, 1))
    mq = pyro.param('mq', torch.zeros(1,2))
    Sigma = pyro.param('Sigma', torch.tensor([[0.05,0.01],[0.02,0.01]]), constraints.positive_definite)
    w = pyro.sample("w", dist.MultivariateNormal(loc=mq, covariance_matrix=Sigma))
    pred = 0.5*x+1
    y = pyro.sample("y", dist.Normal(pred, 0.01))
"""
#guide = AutoMultivariateNormal(model)


x=torch.from_numpy(x_data)
y=torch.from_numpy(y_data)

svi = SVI(model,
          guide,
          optim.Adam({"lr": 0.001}),
          loss=Trace_ELBO())

hist_elbo=[]
pyro.clear_param_store()
num_iters = 5000 if not smoke_test else 2
for i in range(num_iters):
    elbo, current_grad = svi.stepGrad(x,y)
    hist_elbo.append(elbo)
    if i % 500 == 0:
        logging.info("Elbo loss: {}".format(elbo))
        print("Step: ",i,"   ELBO: ",(sum(hist_elbo[-500:])/len(hist_elbo[-500:])))


    from pyro.infer import Predictive


for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))
#print(name)

plt.scatter(x,y)
x_test = np.linspace(0,1,100)
plt.plot(x_test,0.4908*x_test-0.2899, 'r')
plt.show()
#print(guide.get_posterior())
#print(guide.scale_constraint)
#print(guide.scale_tril_constraint)
#L=guide.scale_tril_constraint
#D=pyro.get_param_store().items()
#print(pyro.param("AutoMultivariateNormal.loc"))
