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

def phiGradOptDelta(phi_max, lrr, nele, mean_px, sigma_px, Nx_samp, psi, A, u, iter_grad, sample_x, sample_y):

            phi_max_leaf = phi_max.clone().detach().requires_grad_(True)
            model_phi = [phi_max_leaf]
            criterion = nn.MSELoss()
            optimizer = torch.optim.SGD(model_phi, lr=lrr)
            loss_history = []
            x = torch.exp(sample_x)

            ### Gradient optimization loops
            for epoch in range(0, iter_grad):
                optimizer.zero_grad()
                C = torch.zeros((nele, nele))
                for jj in range(0, Nx_samp):
                    for kk in range(0, Nx_samp):
                        model_phi_nml = model_phi[0] / torch.linalg.norm(model_phi[0])
                        b = (u - torch.matmul(x[jj, 0] * A, torch.transpose(psi, 0, 1) * 1 / x[jj, 0]))
                        C = C + torch.matmul(b, torch.transpose(b, 0, 1))
                C = C / Nx_samp
                resid2 = torch.matmul(torch.transpose(model_phi_nml, 0, 1), C)
                resid2 = torch.matmul(resid2, model_phi_nml)

                if resid2 > 1:
                    print("Warning: Residuals > 1 in phiGradOptDelta()")
                # loss = criterion(resid2, torch.tensor([[0]]).to(torch.float32))
                loss = - resid2
                loss.backward(retain_graph=True)
                gradd = torch.autograd.grad(loss, model_phi[0], retain_graph=True)
                optimizer.step()
                # print("grad: ",gradd)
                # print("phi_max: ",model_phi[0])
                # print("loss: ",loss)
                loss_history.append(loss)
            #testnormphi = torch.linalg.norm(model_phi[0])
            # Projection on the constraint manifold!!! It works now but it could cause problems
            phi_max = model_phi[0] / torch.linalg.norm(model_phi[0])
            return phi_max


def phiGradOptMvn(phi_max, lrr, nele, mean_px, sigma_px, Nx_samp, psi, A, u, iter_grad, sample_x, sample_y):
    phi_max = torch.reshape(phi_max, (-1, 1))
    phi_max_leaf = phi_max.clone().detach().requires_grad_(True)
    model_phi = [phi_max_leaf]
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model_phi, lr=lrr)
    loss_history = []
    x = torch.exp(sample_x)

    ### Gradient optimization loops
    for epoch in range(0, iter_grad):
        optimizer.zero_grad()
        C = torch.zeros((nele, nele))
        for jj in range(0, Nx_samp):
            for kk in range(0, Nx_samp):
                model_phi_nml = model_phi[0] / torch.linalg.norm(model_phi[0])
                #psi_tess = psi
                #with_psi = torch.transpose(psi, 0, 1) * 1 / x[jj, 0]
                #with_y = torch.reshape(sample_y[kk, :], (-1, 1))
                #y_tess = torch.reshape(sample_y[kk, :], (-1, 1))
                #u_min_tess = torch.matmul(x[jj, 0] * A, torch.reshape(sample_y[kk, :], (-1, 1)))
                #b22 = (u - torch.matmul(x[jj, 0] * A, torch.transpose(psi, 0, 1) * 1 / x[jj, 0]))
                b = (u - torch.matmul(x[jj, 0] * A, torch.reshape(sample_y[kk, :], (-1, 1))))
                C = C + torch.matmul(b, torch.transpose(b, 0, 1))
        C = C / Nx_samp / Nx_samp
        #tess_phiii = torch.transpose(model_phi_nml, 0, 1)
        resid2_inter = torch.matmul(torch.transpose(model_phi_nml, 0, 1), C)
        resid2 = torch.matmul(resid2_inter, model_phi_nml)

        if resid2 > 1:
            print("Warning: Residuals > 1 in phiGradOptDelta()")
        # loss = criterion(resid2, torch.tensor([[0]]).to(torch.float32))
        loss = - resid2
        loss.backward(retain_graph=True)
        gradd = torch.autograd.grad(loss, model_phi[0], retain_graph=True)
        optimizer.step()
        # print("grad: ",gradd)
        # print("phi_max: ",model_phi[0])
        # print("loss: ",loss)
        loss_history.append(loss)
    # testnormphi = torch.linalg.norm(model_phi[0])
    # Projection on the constraint manifold!!! It works now but it could cause problems
    phi_max = model_phi[0] / torch.linalg.norm(model_phi[0])
    phi_max = torch.squeeze(phi_max, 1)
    return phi_max


def phiGradOptDeltaDeb(phi_max, lrr, nele, mean_px, sigma_px, Nx_samp, psi, A, u, sample_x):
            phi_max = torch.reshape(phi_max, (-1, 1))
            phi_max_leaf = phi_max.clone().detach().requires_grad_(True)
            model_phi = [phi_max_leaf]
            criterion = nn.MSELoss()
            optimizer = torch.optim.SGD(model_phi, lr=lrr)
            loss_history = []
            x = torch.exp(torch.reshape(sample_x, (-1, 1)))

            ### Gradient optimization loops
            for epoch in range(0, 5):
                optimizer.zero_grad()
                C = torch.zeros((nele, nele))
                for jj in range(0, Nx_samp):
                    model_phi_nml = model_phi[0] / torch.linalg.norm(model_phi[0])
                    norm_model_phi_nml = torch.linalg.norm(model_phi_nml)
                    b = (u - torch.matmul(x[jj, 0] * A, torch.transpose(psi, 0, 1) * 1 / x[jj, 0]))
                    C = C + torch.matmul(b, torch.transpose(b, 0, 1))
                C = C / Nx_samp
                resid2 = torch.matmul(torch.transpose(model_phi_nml, 0, 1), C)
                resid2 = torch.matmul(resid2, model_phi_nml)
                if resid2 > 1:
                    print("Warning: Residuals > 1 in phiGradOptDelta()")

                #resid2 = resid2 / Nx_samp
                # loss = criterion(resid2, torch.tensor([[0]]).to(torch.float32))
                loss = - resid2
                loss.backward(retain_graph=True)
                gradd = torch.autograd.grad(loss, model_phi[0], retain_graph=True)
                optimizer.step()
                # print("grad: ",gradd)
                # print("phi_max: ",model_phi[0])
                # print("loss: ",loss)
                loss_history.append(loss)
            phi_max = model_phi[0] / torch.linalg.norm(model_phi[0])
            phi_max = torch.squeeze(phi_max, 1)
            return phi_max


def phiGradOptMvnDeb(phi_max, lrr, nele, mean_px, sigma_px, Nx_samp, psi, A, u, sample_x, sample_y):
    #if sample_y[1, 4]> 2:
    #    print("Here!")
    phi_max = torch.reshape(phi_max, (-1, 1))
    phi_max_leaf = phi_max.clone().detach().requires_grad_(True)
    model_phi = [phi_max_leaf]
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model_phi, lr=lrr)
    loss_history = []
    x = torch.exp(torch.reshape(sample_x, (-1, 1)))
    y = sample_y

    ### Gradient optimization loops
    for epoch in range(0, 5):
        optimizer.zero_grad()
        resid22 = torch.zeros((1, 1))
        for jj in range(0, Nx_samp):
            for kk in range(0, Nx_samp):
                model_phi_nml = model_phi[0] / torch.linalg.norm(model_phi[0])
                norm_model_phi_nml = torch.linalg.norm(model_phi_nml)
                b = (u - torch.matmul(x[jj, 0] * A, torch.reshape(sample_y[kk, :], (-1, 1))))
                C =  torch.matmul(b, torch.transpose(b, 0, 1))
                resid2 = torch.matmul(torch.transpose(model_phi_nml, 0, 1), C)
                resid2 = torch.matmul(resid2, model_phi_nml)/ Nx_samp / Nx_samp
                resid22 = resid22 + resid2
        resid2 = resid22
        if resid2 > 1:
            print("Warning: Residual= ","{:2f}".format(float(resid2))," > 1 in phiGradOptDelta()")


        # resid2 = resid2 / Nx_samp
        # loss = criterion(resid2, torch.tensor([[0]]).to(torch.float32))
        loss = - resid2
        loss.backward(retain_graph=True)
        gradd = torch.autograd.grad(loss, model_phi[0], retain_graph=True)
        optimizer.step()
        # print("grad: ",gradd)
        # print("phi_max: ",model_phi[0])
        # print("loss: ",loss)
        loss_history.append(loss)
    phi_max = model_phi[0] / torch.linalg.norm(model_phi[0])
    phi_max = torch.squeeze(phi_max, 1)
    return phi_max

def phiGradOptMvnDebCh(phi_max, lrr, nele, mean_px, sigma_px, Nx_samp, psi, A, u, sample_x, sample_y):
    phi_max = torch.reshape(phi_max, (-1, 1))
    phi_max_leaf = phi_max.clone().detach().requires_grad_(True)
    model_phi = [phi_max_leaf]
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model_phi, lr=lrr)
    loss_history = []
    x = torch.exp(torch.reshape(sample_x, (-1, 1)))
    y = sample_y

    ### Gradient optimization loops
    for epoch in range(0, 5):
        optimizer.zero_grad()
        resid2 = torch.zeros((nele, nele))
        for jj in range(0, Nx_samp):
            for kk in range(0, Nx_samp):
                model_phi_nml = model_phi[0] / torch.linalg.norm(model_phi[0])
                norm_model_phi_nml = torch.linalg.norm(model_phi_nml)
                b = (u - torch.matmul(x[jj, 0] * A, torch.reshape(y[kk, :], (-1, 1))))
                C = torch.matmul(b, torch.transpose(b, 0, 1))
                resid2 = torch.matmul(torch.transpose(model_phi_nml, 0, 1), C)
                resid2 = torch.matmul(resid2, model_phi_nml)
                resid22 = resid22 + resid2
        resid2 = resid22
        if resid2 > 2:
            print("Warning: Residual= ","{:2f}".format(float(resid2))," > 1 in phiGradOptDelta()")

        # resid2 = resid2 / Nx_samp
        # loss = criterion(resid2, torch.tensor([[0]]).to(torch.float32))
        loss = - resid2
        loss.backward(retain_graph=True)
        gradd = torch.autograd.grad(loss, model_phi[0], retain_graph=True)
        optimizer.step()
        # print("grad: ",gradd)
        # print("phi_max: ",model_phi[0])
        # print("loss: ",loss)
        loss_history.append(loss)
    phi_max = model_phi[0] / torch.linalg.norm(model_phi[0])
    phi_max = torch.squeeze(phi_max, 1)
    return phi_max


def phiGradOptMvnNx(phi_max, lrr, nele, mean_px, sigma_px, Nx_samp, psi, A, u, sample_x, sample_y, num_epoch, residualcalc): # Correct 3/11/22
    phi_max = torch.reshape(phi_max, (-1, 1))
    phi_max_leaf = phi_max.clone().detach().requires_grad_(True)
    model_phi = [phi_max_leaf]
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model_phi, lr=lrr)
    loss_history = []
    x = torch.exp(torch.reshape(sample_x, (-1, 1)))
    y = sample_y

    ### Gradient optimization loops
    for epoch in range(0, num_epoch):
        optimizer.zero_grad()
        resid22 = torch.zeros(1, 1)
     #   if y[0, 4] > 5:
     #       print(y[0,4])
        for jj in range(0, Nx_samp):
            model_phi_nml = model_phi[0] / torch.linalg.norm(model_phi[0])
            norm_model_phi_nml = torch.linalg.norm(model_phi_nml)
            b = (u - torch.matmul(x[jj, 0] * A, torch.reshape(y[jj, :], (-1, 1))))
            C = torch.matmul(b, torch.transpose(b, 0, 1))
            resid2 = torch.matmul(torch.transpose(model_phi_nml, 0, 1), C)
            resid2 = torch.matmul(resid2, model_phi_nml)
            resid22 = resid22 + resid2 / Nx_samp
        resid2 = resid22
        #manDer = -2*resid2/torch.linalg.norm(model_phi[0])*b # manual derivative for Nx = 1 for comparison reasons
        #manDer = -1/2/torch.sqrt(torch.matmul(torch.transpose(model_phi[0], 0, 1),model_phi[0]))*2*model_phi[0]
        ### Manual derivative implementation for comparison ###
        manDer = -1/2*(torch.sum(model_phi[0]**2))**(-3/2)*2*model_phi[0]*torch.matmul(torch.transpose(model_phi[0],0,1),b)
        manDer = manDer + 1/torch.linalg.norm(model_phi[0])*b
        manDer = -manDer * 2 * torch.sqrt(resid2)
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
    residualcalc.residual = resid2
    return phi_max.clone().detach()


def phiEigOpt(phi_maxx, sample_x, sample_y, pow_iter_tol, residualcalc, eigRelax=None): ## Correct-3/11/22
    """
    :param phi_maxx:
    :param sample_x:
    :param sample_y:
    :param pow_iter_tol:
    :param residualcalc:
    :param eigRelax:
    :return:
    """
    ### For Relaxation ###
    if len(phi_maxx.size()) > 1:
        phi_max_old = torch.reshape(phi_maxx, (-1, )).clone().detach() ### If you don't detach it everything will become slower
    else:
        phi_max_old = phi_maxx
    ### For Relaxation ###
    phi_max = torch.reshape(phi_maxx, (-1, 1))
    C = residualcalc.calcResidual(sample_x, sample_y)
    resid2 = torch.matmul(torch.transpose(phi_max, 0, 1), C)
    residualcalc.residual = torch.matmul(resid2, phi_max)
    #C = calcResidual("Polynomial", sample_x, sample_y, Nx_samp, nele, u, A, poly_pow=3)
    res = powerIteration(C, pow_iter_tol)
    ### In case that Nx_samp = 1, then b = sqrt(lambda)*phi, however this doesn't help because we again need to find lambda
    #res2 = np.linalg.eig(C)
    #tess2 = res2[1][:,0]
    #res3 = res2[1][:,0]*np.sqrt(res2[0][0])
    ### In case that Nx_samp = 1, then b = sqrt(lambda)*phi, however this doesn't help because we again need to find lambda
    phi_max = res[0]
    phi_max = torch.squeeze(phi_max, 1)
    ### For Relaxation ###
    if eigRelax is not None:
        phi_max_return = phi_max_old + eigRelax*(phi_max - phi_max_old)
        #phi_max_return = torch.add(phi_max_old, torch.mul(torch.add(phi_max, torch.mul(phi_max_old,-1)), eigRelax))
        phi_max_return = phi_max_return / torch.linalg.norm(phi_max_return)
    else:
        phi_max_return = phi_max
    ### For Relaxation ###
    return phi_max_return

class residualCalc:
    def __init__(self, mode, Nx_samp, pde, poly_pow=None):
        self.mode = mode
        self.pde = pde
        self.Nx_samp = Nx_samp
        self.phi_max = 0
        self.A = pde.A
        self.u = pde.u
        self.poly_pow = poly_pow
        self.nele = pde.effective_nele
        self.C = torch.zeros(self.nele, self.nele)
        self.residual = torch.tensor(0)
        if mode == "TrueSol":
            self.psi_init = torch.rand(1, self.nele) * 0.01  # Initialization of psi
            self.psi_history_init = np.zeros((self.nele, 1))
        elif mode == "Polynomial":
            self.psi_init = torch.ones(self.nele, poly_pow + 1) * 10 # Initialization of psi
            self.psi_history_init = np.zeros((self.nele * (poly_pow + 1), 1))

    def calcResidual(self, sample_x, sample_y):
        x = torch.reshape(sample_x, (-1, 1))
        y = sample_y
        CC = torch.zeros(self.nele, self.nele)
        if self.mode == "TrueSol":
            for jj in range(0, self.Nx_samp):
                b = self.pde.calcResKernel(x[jj, 0], y[jj, :])
                C = torch.matmul(b, torch.transpose(b, 0, 1))
                CC = CC + C / self.Nx_samp
            C = CC
        elif self.mode == "Polynomial":
            for jj in range(0, self.Nx_samp):
                b = self.pde.calcResKernel(x[jj, 0], y[jj, :])
                C = torch.matmul(b, torch.transpose(b, 0, 1))
                CC = CC + C / self.Nx_samp
            C = CC
        else:
            print("Warning calcResidual(): This mode isn't available")
            C = CC
        self.C = C
        return C
