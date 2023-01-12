### Importing Libraries ###
import numpy as np
import math
import random
import pandas as pd
import sys
from numpy.linalg import inv
import matplotlib.pyplot as plt

### Import Pyro/Torch Libraries ###
import argparse
import torch
import torch.nn as nn
from torch.nn.functional import normalize

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoMultivariateNormal, AutoDelta
from pyro.optim import Adam
from pyro.nn import PyroSample
from torch import nn
from pyro.nn import PyroModule
import pyro.optim as optim
import os
import logging
from torch.distributions import constraints

smoke_test = ('CI' in os.environ)
from torch.distributions import constraints
from pyro.infer import Predictive
from utils.powerIteration import powerIteration
import time
from textwrap import wrap

class phiOptimizer:
    def __init__(self, pde, poly_pow=None, eigRelax=None, powIterTol=None):
        self.pde = pde
        self.eigRelax = eigRelax
        self.gradLr = 0.1
        self.epoch = 100
        self.powIterTol = powIterTol
        self.nele = pde.effective_nele
        self.mean_px = pde.mean_px
        self.sigma_px = pde.sigma_px
        self.Nx_samp = pde.Nx_samp
        self.Nx_samp_phi = self.Nx_samp
        self.x = torch.zeros(self.Nx_samp_phi, 1)
        self.y = torch.zeros(self.Nx_samp_phi, self.nele)
        self.Nx_samp = self.Nx_samp
        self.Nx_counter = 0
        self.phi_max = torch.rand(self.nele)*0.1+torch.ones(self.nele)
        self.phi_max = self.phi_max/torch.linalg.norm(torch.ones(self.nele))
        self.poly_pow = poly_pow
        self.temp_res = torch.rand(self.Nx_samp).tolist()
        self.full_res_temp = torch.rand(self.Nx_samp).tolist()
        for i in range(0, len(self.full_res_temp)):
            self.full_res_temp[i] = torch.rand(self.nele)*0.1

    def phiGradOptMvnNx(self): # Correct 3/11/22
        phi_max = torch.reshape(self.phi_max, (-1, 1))
        phi_max_leaf = phi_max.clone().detach().requires_grad_(True)
        model_phi = [phi_max_leaf]
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model_phi, lr=self.gradLr)
        loss_history = []

        ### Gradient optimization loops
        for epoch in range(0, self.epoch):
            optimizer.zero_grad()
            resid22 = torch.zeros(1, 1)
         #   if y[0, 4] > 5:
         #       print(y[0,4])
            for jj in range(0, self.Nx_samp_phi):
                model_phi_nml = model_phi[0] / torch.linalg.norm(model_phi[0])
                b = self.pde.calcResKernel(self.x[jj,:], self.y[jj,:])
                C = torch.matmul(b, torch.transpose(b, 0, 1))
                resid2 = torch.matmul(torch.transpose(model_phi_nml, 0, 1), C)
                resid2 = torch.matmul(resid2, model_phi_nml)
                resid22 = resid22 + resid2 / self.Nx_samp_phi
            resid2 = resid22
            #manDer = -2*resid2/torch.linalg.norm(model_phi[0])*b # manual derivative for Nx = 1 for comparison reasons
            #manDer = -1/2/torch.sqrt(torch.matmul(torch.transpose(model_phi[0], 0, 1),model_phi[0]))*2*model_phi[0]
            ### Manual derivative implementation for comparison ###
            #manDer = -1/2*(torch.sum(model_phi[0]**2))**(-3/2)*2*model_phi[0]*torch.matmul(torch.transpose(model_phi[0],0,1),b)
            #manDer = manDer + 1/torch.linalg.norm(model_phi[0])*b
            #manDer = -manDer * 2 * torch.sqrt(resid2)
            ### Manual derivative implementation for comparison ###
            #if resid2 > 2:
            #    print("Warning: Residual= ","{:2f}".format(float(resid2))," > 1 in phiGradOptDelta()")
            #loss = criterion(resid2, torch.tensor([[0]]).to(torch.float32))
            #loss = - resid2
            loss = -torch.linalg.norm(model_phi[0])
            loss = -resid2
            loss.backward(retain_graph=True)
            #gradd = torch.autograd.grad(loss, model_phi_nml, retain_graph=True)
            optimizer.step()
            #print("Is model_phi[0] leaf tensor:", model_phi[0].is_leaf)
            #print("Is loss leaf tensor:", loss.is_leaf)
            #print("resid2: ","{:8f}".format(float(resid2)))
            #print("loss: ","{:8f}".format(float(loss)))
            #print("grad: ",model_phi[0].grad) ## Attention! This is dloss/dphi
            #print("manual grad:", manDer)
            # print("phi_max: ",model_phi[0])
            # print("loss: ",loss)
            loss_history.append(loss)
        phi_max = model_phi[0] / torch.linalg.norm(model_phi[0])
        phi_max = torch.squeeze(phi_max, 1)
        self.phi_max = phi_max
        return phi_max.clone().detach()


    def phiEigOpt(self): ## Correct-3/11/22
        """
        :param phi_maxx:
        :param sample_x:
        :param sample_y:
        :param pow_iter_tol:
        :param residualcalc:
        :param eigRelax:
        :return:
        """
        resid2 = 0
        C = 0
        ### For Relaxation ###
        if len(self.phi_max.size()) > 1:
            phi_max_old = torch.reshape(self.phi_max, (-1, )).clone().detach() ### If you don't detach it everything will become slower
        else:
            phi_max_old = self.phi_max
        ### For Relaxation ###
        phi_max = torch.reshape(self.phi_max, (-1, 1))
        for jj in range(0, self.Nx_samp_phi):
            #C = torch.matmul(torch.reshape(self.full_res_temp[jj], (-1, 1)),
            #                 torch.reshape(self.full_res_temp[jj], (1, -1)))/self.Nx_samp ### Without the **2 it could behave even better
            b = self.pde.calcResKernel(self.x[jj], self.y[jj,:])
            C = C + torch.matmul( torch.reshape(b, (-1, 1)), torch.reshape(b, (1, -1)))/self.Nx_samp_phi
        #C = calcResidual("Polynomial", sample_x, sample_y, Nx_samp, nele, u, A, poly_pow=3)
        res = powerIteration(C, self.powIterTol)
        #print("Res", C)
        ### In case that Nx_samp = 1, then b = sqrt(lambda)*phi, however this doesn't help because we again need to find lambda
        #res2 = np.linalg.eig(C)
        #tess2 = res2[1][:,0]
        #res3 = res2[1][:,0]*np.sqrt(res2[0][0])
        ### In case that Nx_samp = 1, then b = sqrt(lambda)*phi, however this doesn't help because we again need to find lambda
        phi_max = res[0]
        phi_max = torch.squeeze(phi_max, 1)
        ### For Relaxation ###
        if self.eigRelax is not None:
            phi_max_return = phi_max_old + self.eigRelax*(phi_max - phi_max_old)
            #phi_max_return = torch.add(phi_max_old, torch.mul(torch.add(phi_max, torch.mul(phi_max_old,-1)), eigRelax))
            phi_max_return = phi_max_return / torch.linalg.norm(phi_max_return)
        else:
            phi_max_return = phi_max
        self.phi_max = phi_max_return
        #print(self.phi_max)
        ### For Relaxation ###
        return phi_max_return


