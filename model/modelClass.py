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
from pyro.nn import PyroModule, PyroParam
import pyro.optim as optim
import os
import logging
from torch.distributions import constraints

smoke_test = ('CI' in os.environ)
from torch.distributions import constraints
from pyro.infer import Predictive
import time
# from numba import jit
from textwrap import wrap


### Classes and Functions ###


class modelMvn:
    def __init__(self, Nx_samp, nele, lr_for_phi, Iter_grad, phi_max):
        self.Nx_samp = Nx_samp
        self.lr_for_phi = lr_for_phi
        self.Iter_grad = Iter_grad
        self.x = torch.zeros(Nx_samp, 1)
        self.y = torch.zeros(Nx_samp, nele)
        # self.psi = torch.zeros(1, nele)
        self.Nx_counter = 0
        self.phi_max = phi_max
        # self.x = torch.rand(Nx_samp, 1) * 0.01  # Initialization of sample_x
        # self.y = torch.rand(Nx_samp, nele) * 0.01  # Initialization of sample_y

    def removeSamples(self):
        self.Nx_counter = 0

    def executeModel(self, phi_max, residuals, A, u, sigma_r, sigma_w, psii, nele, sigma_px, mean_px):
        x = pyro.sample("x", dist.Normal(loc=mean_px, scale=sigma_px))
        y = pyro.sample("y", dist.MultivariateNormal(loc=torch.zeros((1, nele)),
                                                     covariance_matrix=sigma_w ** 2 * torch.eye(nele, nele)))
        y = torch.transpose(y, 0, 1)
        # y = torch.transpose(y, 0, 1)
        b = (u - torch.matmul(torch.exp(x) * A, y))
        c = torch.matmul(b, torch.transpose(b, 0, 1))
        phi_max = torch.reshape(phi_max, (-1, 1))
        rwmax2 = torch.matmul(torch.matmul(torch.transpose(phi_max, 0, 1), c), phi_max)
        rwmax2 = torch.squeeze(rwmax2, 1)
        rwmax2 = torch.squeeze(rwmax2, 0)

        with pyro.plate("data", 1):
            residuals = pyro.sample("residuals", dist.Normal(rwmax2, sigma_r), obs=residuals)

    def executeGuide(self, phi_max, residuals, A, u, sigma_r, sigma_w, psii, nele, sigma_px, mean_px):
        # psi = pyro.param("psi", torch.rand(1, 10)*0.01)
        psi = pyro.param("psi", psii)
        # phi_max = pyro.param("phi_max", phi_max)
        x = pyro.sample("x", dist.Normal(loc=mean_px, scale=sigma_px))
        self.x[self.Nx_counter, 0] = x
        # mq = pyro.param('mq', 1 / torch.exp(x) * torch.ones(1, 10))
        # Sigma = pyro.param('Sigma', torch.eye(10, 10), constraints.positive_definite)
        # y = pyro.sample("y", dist.MultivariateNormal(loc=mq, covariance_matrix=Sigma))
        y = pyro.sample("y", dist.Delta(psi / torch.exp(x), event_dim=1))
        for i in range(0, nele):
            self.y[self.Nx_counter, i] = y[0, i]
        # self.psi = psi
        self.Nx_counter = self.Nx_counter + 1
        # self.phi_max = phiGradOptMvn(self.phi_max, self.lr_for_phi, nele, mean_px, sigma_px, self.Nx_samp, psi, A, u,
        #                               self.Iter_grad, self.x, self.y)  ## Deactivate for original Delta to work
        self.phi_max = phiGradOptDelta(torch.reshape(self.phi_max, (-1, 1)), self.lr_for_phi, nele, mean_px,
                                       sigma_px, self.Nx_samp, psi, A, u,
                                       self.Iter_grad, self.x, self.y)  ## Deactivate for original Delta to work
        # print(self.phi_max)
        # return y
        # residuals = x*y


class modelDelta:
    def __init__(self, pde, phi_max):
        self.pde = pde
        self.nele = pde.effective_nele
        self.mean_px = pde.mean_px
        self.sigma_px = pde.sigma_px
        self.Nx_samp = pde.Nx_samp
        self.x = torch.zeros(self.Nx_samp, 1)
        self.y = torch.zeros(self.Nx_samp, self.nele)
        self.Nx_samp = self.Nx_samp
        self.Nx_counter = 0
        self.phi_max = phi_max
        self.x2 = torch.zeros(self.Nx_samp, 1)

    def removeSamples(self):
        self.Nx_counter = 0

    def executeModel(self, phi_max, residuals, sigma_r, sigma_w, psii):
        x = pyro.sample("x", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
        y = pyro.sample("y", dist.MultivariateNormal(loc=torch.zeros((1, self.nele)),
                                                     covariance_matrix=sigma_w ** 2 * torch.eye(self.nele, self.nele)))

        b = self.pde.calcResKernel(x, y)
        c = torch.matmul(b, torch.transpose(b, 0, 1))
        phi_max = torch.reshape(phi_max, (-1, 1))
        rwmax2 = torch.matmul(torch.matmul(torch.transpose(phi_max, 0, 1), c), phi_max)
        rwmax2 = torch.squeeze(rwmax2, 1)
        rwmax2 = torch.squeeze(rwmax2, 0)

        with pyro.plate("data", 1):
            residuals = pyro.sample("residuals", dist.Normal(rwmax2, sigma_r), obs=residuals)

    def executeGuide(self, phi_max, residuals, sigma_r, sigma_w, psii):
        psi = pyro.param("psi", psii)
        x = pyro.sample("x", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
        y = pyro.sample("y", dist.Delta(psi / torch.exp(x), event_dim=1))
        # y = pyro.sample("y", dist.Delta(psi * x, event_dim=1))

    def sample(self, phi_max, residuals, sigma_r, sigma_w, psii):
        for i in range(0, self.Nx_samp):
            xx = pyro.sample("xx", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
            yy = pyro.sample("yy", dist.Delta(psii / torch.exp(xx), event_dim=1))
            # yy = pyro.sample("yy", dist.Delta(psii * xx, event_dim=1))

            self.x[self.Nx_counter, 0] = xx
            for j in range(0, self.nele):
                self.y[self.Nx_counter, j] = yy[0, j]
            self.Nx_counter = self.Nx_counter + 1
        #        if self.y[0, 4] > 10:
        #            print("Here!")
        self.removeSamples()


class modelDeltaAllres:
    def __init__(self, pde, phi_max):
        self.pde = pde
        self.nele = pde.effective_nele
        self.mean_px = pde.mean_px
        self.sigma_px = pde.sigma_px
        self.Nx_samp = pde.Nx_samp
        self.x = torch.zeros(self.Nx_samp, 1)
        self.y = torch.zeros(self.Nx_samp, self.nele)
        self.Nx_counter = 0
        self.phi_max = phi_max
        self.x2 = torch.zeros(self.Nx_samp, 1)

    def removeSamples(self):
        self.Nx_counter = 0

    def executeModel(self, phi_max, residuals, sigma_r, sigma_w, psii):
        x = pyro.sample("x", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
        y = pyro.sample("y", dist.MultivariateNormal(loc=torch.zeros((1, self.nele)),
                                                     covariance_matrix=sigma_w ** 2 * torch.eye(self.nele, self.nele)))

        b = self.pde.calcResKernel(x, y)
        b = torch.squeeze(b, 1)
        with pyro.plate("data", len(residuals)):
            residuals = pyro.sample("residuals", dist.Normal(b, sigma_r), obs=residuals)

    def executeGuide(self, phi_max, residualss, sigma_r, sigma_w, psii):
        psi = pyro.param("psi", psii)
        x = pyro.sample("x", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
        y = pyro.sample("y", dist.Delta(psi / torch.exp(x), event_dim=1))
        # y = pyro.sample("y", dist.Delta(psi * x, event_dim=1))

    def sample(self, phi_max, residuals, sigma_r, sigma_w, psii):
        for i in range(0, self.Nx_samp):
            xx = pyro.sample("xx", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
            yy = pyro.sample("yy", dist.Delta(psii / torch.exp(xx), event_dim=1))
            # yy = pyro.sample("yy", dist.Delta(psii * xx, event_dim=1))

            self.x[self.Nx_counter, 0] = xx
            for j in range(0, self.nele):
                self.y[self.Nx_counter, j] = yy[0, j]
            self.Nx_counter = self.Nx_counter + 1
        #        if self.y[0, 4] > 10:
        #            print("Here!")
        self.removeSamples()


class modelMvnDeb:
    def __init__(self, Nx_samp, nele, phi_max):
        self.x = torch.zeros(Nx_samp, 1)
        self.y = torch.zeros(Nx_samp, nele)
        self.Nx_samp = Nx_samp
        self.Nx_counter = 0
        self.phi_max = phi_max
        self.x2 = torch.zeros(Nx_samp, 1)

    def removeSamples(self):
        self.Nx_counter = 0

    def executeModel(self, phi_max, residuals, A, u, sigma_r, sigma_w, psii, nele, sigma_px, mean_px, Sigmaa):
        x = pyro.sample("x", dist.Normal(loc=mean_px, scale=sigma_px))
        y = pyro.sample("y", dist.MultivariateNormal(loc=torch.zeros((1, nele)),
                                                     covariance_matrix=sigma_w ** 2 * torch.eye(nele, nele)))

        y = torch.transpose(y, 0, 1)
        # y = torch.transpose(y, 0, 1)
        b = (u - torch.matmul(torch.exp(x) * A, y))
        c = torch.matmul(b, torch.transpose(b, 0, 1))
        phi_max = torch.reshape(phi_max, (-1, 1))
        rwmax2 = torch.matmul(torch.matmul(torch.transpose(phi_max, 0, 1), c), phi_max)
        rwmax2 = torch.squeeze(rwmax2, 1)
        rwmax2 = torch.squeeze(rwmax2, 0)

        with pyro.plate("data", 1):
            residuals = pyro.sample("residuals", dist.Normal(rwmax2, sigma_r), obs=residuals)

    def executeGuide(self, phi_max, residuals, A, u, sigma_r, sigma_w, psii, nele, sigma_px, mean_px, Sigmaa):
        # psi = pyro.param("psi", torch.rand(1, 10)*0.01)
        # psi = pyro.param("psi", psii)
        # phi_max = pyro.param("phi_max", phi_max)
        x = pyro.sample("x", dist.Normal(loc=mean_px, scale=sigma_px))
        mq = pyro.param('mq', psii)
        ##sig_mat = torch.rand(nele, nele)
        ##sig_mat = torch.mm(sig_mat, sig_mat.t())+ torch.eye(nele, nele)
        ##Sigma = pyro.param('Sigma', sig_mat/10**7, constraints.positive_definite)
        # Sigma = pyro.param('Sigma', Sigmaa, constraints.positive_definite)
        Sigma = pyro.param('Sigma', Sigmaa, constraints.positive)
        Sigmam = torch.diag(Sigma)
        y = pyro.sample("y", dist.MultivariateNormal(loc=mq / torch.exp(x),
                                                     covariance_matrix=Sigmam))
        # y = pyro.sample("y", dist.Delta(psi / torch.exp(x), event_dim=1))
        # self.x[self.Nx_counter, 0] = x
        # for i in range(0, nele):
        #    self.y[self.Nx_counter, i] = y[0, i]
        # self.Nx_counter = self.Nx_counter + 1
        # self.phi_max = phiGradOptDeltaDeb(phi_max, 0.1, nele, mean_px, sigma_px, 1, psi, A, u)
        # if self.Nx_counter == self.Nx_samp:
        #    self.phi_max = phiGradOptDeltaDeb(phi_max, 0.1, nele, mean_px, sigma_px, self.Nx_counter, psi, A, u, self.x)
        # print(y)
        # return y
        # residuals = x*y

    def sample(self, phi_max, residuals, A, u, sigma_r, sigma_w, psii, nele, sigma_px, mean_px, Sigmaa):
        for i in range(0, self.Nx_samp):
            # psi = pyro.param("psi", torch.rand(1, 10)*0.01)
            # phi_max = pyro.param("phi_max", phi_max)
            xx = pyro.sample("xx", dist.Normal(loc=mean_px, scale=sigma_px))
            # mq = pyro.param('mq', 1 / torch.exp(x) * torch.ones(1, 10))
            # Sigma = pyro.param('Sigma', torch.eye(10, 10), constraints.positive_definite)
            # y = pyro.sample("y", dist.MultivariateNormal(loc=mq, covariance_matrix=Sigma))
            yy = pyro.sample("yy", dist.MultivariateNormal(loc=psii / torch.exp(xx),
                                                           covariance_matrix=torch.diag(Sigmaa)))

            # yy = pyro.sample("yy", dist.Delta(psii / torch.exp(xx), event_dim=1))
            self.x[self.Nx_counter, 0] = xx
            for j in range(0, nele):
                self.y[self.Nx_counter, j] = yy[0, j]
            self.Nx_counter = self.Nx_counter + 1
        self.removeSamples()


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class modelDeltaNn:
    def __init__(self, pde, phi_max, poly_pow=None, allRes=True):
        self.pde = pde
        self.allRes = allRes
        self.nele = pde.effective_nele
        self.mean_px = pde.mean_px
        self.sigma_px = pde.sigma_px
        self.Nx_samp = pde.Nx_samp
        self.x = torch.zeros(self.Nx_samp, 1)
        self.y = torch.zeros(self.Nx_samp, self.nele)
        self.Nx_samp = self.Nx_samp
        self.Nx_counter = 0
        self.phi_max = phi_max
        self.x2 = torch.zeros(self.Nx_samp, 1)
        self.poly_pow = poly_pow
        if not self.allRes:
            self.residuals = torch.tensor(0)
            self.iter_plate = 1
        else:
            self.residuals = torch.zeros(self.nele)
            self.iter_plate = len(self.residuals)
        self.psiiweight = torch.rand(self.nele, poly_pow + 1) * 0.01  # Initialization of psi
        self.psiiweight = self.psiiweight[:, 1:]
        self.psiibias = self.psiiweight[:, 0]
        self.psii = [self.psiiweight, self.psiibias]
        self.temp_res = None

    def removeSamples(self):
        self.Nx_counter = 0

    def polynomial(self, x):
        y = torch.zeros(self.poly_pow + 1, 1)
        for k in range(0, self.poly_pow + 1):
            y[k, 0] = x ** k
        return y

    def executeModel(self, phi_max, sigma_r, sigma_w):
        x = pyro.sample("x", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
        y = pyro.sample("y", dist.MultivariateNormal(loc=torch.zeros((1, self.nele)),
                                                     covariance_matrix=sigma_w ** 2 * torch.eye(self.nele, self.nele)))

        b = self.pde.calcResKernel(x, y)
        self.temp_res = b
        if not self.allRes:
            c = torch.matmul(b, torch.transpose(b, 0, 1))
            phi_max = torch.reshape(phi_max, (-1, 1))
            rwmax2 = torch.matmul(torch.matmul(torch.transpose(phi_max, 0, 1), c), phi_max)
            rwmax2 = torch.squeeze(rwmax2, 1)
            rwmax2 = torch.squeeze(rwmax2, 0)
            res = rwmax2

        else:
            b = torch.squeeze(b, 1)
            res = b

        with pyro.plate("data", self.iter_plate):
            residuals = pyro.sample("residuals", dist.Normal(res, sigma_r), obs=self.residuals)

    def executeGuide(self, phi_max, sigma_r, sigma_w):
        # mq = pyro.param('mq', self.psii)
        # self.linear.weight = PyroParam(torch.zeros(4,2))

        self.linear.weight = PyroParam(self.psii[0])
        self.linear.bias = PyroParam(self.psii[1])

        tess = self.linear.weight
        tesss = self.linear.bias

        x = pyro.sample("x", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
        # if True: # If NN mode is on
        #    self.nn.bias = mq[:, 0]
        #    self.nn.weight = mq[:, 1:]
        polvecx = self.polynomial(x)
        polvecx = torch.squeeze(polvecx, 1)
        # mq_final = torch.matmul(mq, polvecx)
        # mq_final = torch.transpose(mq_final, 0, 1)
        mq_final = self.linear(polvecx[1:]).squeeze(-1)
        y = pyro.sample("y", dist.Delta(mq_final, event_dim=1))
        tess = y
        tsss = y

    def sample(self, phi_max, sigma_r, sigma_w):
        for i in range(0, self.Nx_samp):
            xx = pyro.sample("xx", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
            """
            polvecxx = self.polynomial(xx)
            mqq = torch.matmul(self.psii, polvecxx)
            mqq = torch.transpose(mqq, 0, 1)
            """
            polvecxx = self.polynomial(xx)
            polvecxx = torch.squeeze(polvecxx, 1)
            # mq_final = torch.matmul(mq, polvecx)
            # mq_final = torch.transpose(mq_final, 0, 1)
            mqq = self.linear(polvecxx[1:])
            mqq = torch.reshape(mqq, (1, -1))

            yy = pyro.sample("yy", dist.Delta(mqq, event_dim=1))
            self.x[self.Nx_counter, 0] = xx
            for j in range(0, self.nele):
                self.y[self.Nx_counter, j] = yy[0, j]
            self.Nx_counter = self.Nx_counter + 1
        self.removeSamples()


class modelDeltaPolynomial:
    def __init__(self, pde, phi_max, poly_pow=None, allRes=True):
        self.pde = pde
        self.allRes = allRes
        self.nele = pde.effective_nele
        self.mean_px = pde.mean_px
        self.sigma_px = pde.sigma_px
        self.Nx_samp = pde.Nx_samp
        self.x = torch.zeros(self.Nx_samp, 1)
        self.y = torch.zeros(self.Nx_samp, self.nele)
        self.Nx_samp = self.Nx_samp
        self.Nx_counter = 0
        self.phi_max = phi_max
        self.x2 = torch.zeros(self.Nx_samp, 1)
        self.poly_pow = poly_pow
        if not self.allRes:
            # self.residuals = torch.tensor(0) # old options
            # self.iter_plate = 1 # old options
            self.residuals = torch.zeros(self.nele)
            self.iter_plate = len(self.residuals)
        else:
            self.residuals = torch.zeros(self.nele)
            self.iter_plate = len(self.residuals)
        self.psii = torch.rand(self.nele, poly_pow + 1) * 0.01  # Initialization of psi
        self.temp_res = []
        self.model_time = 0
        self.guide_time = 0
        self.sample_time = 0

    def removeSamples(self):
        self.Nx_counter = 0
        self.temp_res = []

    def polynomial(self, x):
        y = torch.zeros(self.poly_pow + 1, 1)
        for k in range(0, self.poly_pow + 1):
            y[k, 0] = x ** k
        return y

    def executeModel(self, phi_max, sigma_r, sigma_w):
        t0 = time.time()
        x = pyro.sample("x", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
        y = pyro.sample("y", dist.MultivariateNormal(loc=torch.zeros((1, self.nele)),
                                                     covariance_matrix=sigma_w ** 2 * torch.eye(self.nele, self.nele)))

        b = self.pde.calcResKernel(x, y)
        self.temp_res.append(torch.linalg.norm(b.clone().detach()))
        if not self.allRes:
            c = torch.matmul(b, torch.transpose(b, 0, 1))
            phi_max = torch.reshape(phi_max, (-1, 1))
            ### Original Implementation ###
            # rwmax2 = torch.matmul(torch.matmul(torch.transpose(phi_max, 0, 1), c), phi_max)
            # rwmax2 = torch.squeeze(rwmax2, 1)
            # rwmax2 = torch.squeeze(rwmax2, 0)
            ### Original Implementation ###

            rwmax2 = torch.matmul(torch.transpose(phi_max, 0, 1), c)
            rwmax2 = torch.squeeze(rwmax2, 1)
            rwmax2 = torch.squeeze(rwmax2, 0)
            res = rwmax2

        else:
            b = torch.squeeze(b, 1)
            res = b
        with pyro.plate("data", self.iter_plate):
            residuals = pyro.sample("residuals", dist.Normal(res, sigma_r), obs=self.residuals)
        self.model_time += time.time() - t0

    def executeGuide(self, phi_max, sigma_r, sigma_w):
        t0 = time.time()
        x = pyro.sample("x", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
        mq = pyro.param('mq', self.psii)
        polvecx = self.polynomial(x)
        mq_final = torch.matmul(mq, polvecx)
        mq_final = torch.transpose(mq_final, 0, 1)
        mq_final = torch.squeeze(mq_final, 0)
        y = pyro.sample("y", dist.Delta(mq_final, event_dim=1))
        self.guide_time = self.guide_time + time.time() - t0

    def sample(self, phi_max, sigma_r, sigma_w):
        t0 = time.time()
        for i in range(0, self.Nx_samp):
            xx = pyro.sample("xx", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
            polvecxx = self.polynomial(xx)
            mqq = torch.matmul(self.psii, polvecxx)
            mqq = torch.transpose(mqq, 0, 1)
            yy = pyro.sample("yy", dist.Delta(mqq, event_dim=1))
            self.x[self.Nx_counter, 0] = xx
            for j in range(0, self.nele):
                self.y[self.Nx_counter, j] = yy[0, j]
            self.Nx_counter = self.Nx_counter + 1
        self.sample_time = self.sample_time + time.time() - t0


class modelPolynomial:
    def __init__(self, pde, phi_max, poly_pow=None, allRes=True, surgtType='Delta', stdInit=4.3):  # It was stdInit=8
        self.pde = pde
        self.allRes = allRes
        self.surgtType = surgtType
        self.nele = pde.effective_nele
        self.mean_px = pde.mean_px
        self.sigma_px = pde.sigma_px
        self.Nx_samp = pde.Nx_samp
        self.Nx_samp_phi = self.Nx_samp
        self.x = torch.zeros(self.Nx_samp_phi, 1)
        self.y = torch.zeros(self.Nx_samp_phi, self.nele)
        self.Nx_counter = 0
        self.phi_max = phi_max
        self.poly_pow = poly_pow
        self.guideExecCounter = 0
        self.modelExecCounter = 0
        if not self.allRes:
            self.residuals = torch.tensor(0)
            self.iter_plate = 1
        else:
            self.residuals = torch.zeros(self.nele)
            self.iter_plate = len(self.residuals)

        self.psi_init = torch.tensor([[1, -1.4195196367299, 0.6010709401081]]) * torch.tensor(0.125)
        self.psi_init = torch.tensor([[1, -1.4195196367299, 0.6010709401081], [1, -1.4195196367299,
                                                                               0.6010709401081]]) * torch.tensor(1 / 9)

        self.psi_init = torch.tensor(
            [[1 * 3 / 32, -1.4195196367299 * 3 / 32, 0.6010709401081 * 3 / 32], [1 * 0.125, -1.4195196367299 * 0.125,
                                                                                 0.6010709401081 * 0.125],
             [1 * 3 / 32, -1.4195196367299 * 3 / 32, 0.6010709401081 * 3 / 32]])

        self.psi_init = torch.rand(self.nele, poly_pow + 1) * 0.01  # Initialization of psi


        """
            self.psi_init = torch.tensor([[0.0316, -0.0177, 0.0110],
                    [0.0606, -0.0306, 0.0142],
                    [0.0829, -0.0494, 0.0217],
                    [0.1019, -0.0640, 0.0247],
                    [0.1153, -0.0793, 0.0300],
                    [0.1248, -0.0996, 0.0414],
                    [0.1315, -0.1133, 0.0473],
                    [0.1334, -0.1163, 0.0467],
                    [0.1331, -0.1232, 0.0521],
                    [0.1301, -0.1171, 0.0507],
                    [0.1309, -0.1120, 0.0455],
                    [0.1302, -0.1036, 0.0374],
                    [0.1241, -0.0995, 0.0398],
                    [0.1124, -0.0846, 0.0347],
                    [0.0896, -0.0658, 0.0295],
                    [0.0685, -0.0427, 0.0150],
                    [0.0440, -0.0254, 0.0134],
                    [0.0270, -0.0203, 0.0104]])
            """
        """
            self.psi_init = torch.tensor([[0.0085, -0.0006, -0.0025, -0.0005, 0.0062],
                    [0.0102, 0.0016, 0.0057, -0.0024, 0.0046],
                    [0.0169, 0.0005, 0.0085, -0.0030, 0.0069],
                    [0.0194, 0.0001, 0.0130, 0.0013, 0.0048],
                    [0.0230, -0.0003, 0.0141, -0.0016, 0.0059],
                    [0.0228, 0.0022, 0.0134, -0.0016, 0.0042],
                    [0.0223, 0.0002, 0.0091, 0.0008, 0.0074],
                    [0.0250, -0.0018, 0.0170, -0.0030, 0.0033],
                    [0.0227, 0.0008, 0.0121, -0.0059, 0.0061],
                    [0.0236, 0.0024, 0.0138, -0.0066, 0.0020],
                    [0.0224, 0.0008, 0.0096, -0.0013, 0.0013],
                    [0.0221, 0.0040, 0.0055, -0.0023, 0.0038],
                    [0.0221, 0.0014, 0.0095, -0.0046, 0.0026],
                    [0.0175, 0.0033, 0.0098, -0.0070, 0.0045],
                    [0.0174, 0.0029, 0.0131, -0.0092, 0.0025],
                    [0.0186, -0.0012, 0.0095, -0.0091, 0.0028],
                    [0.0152, -0.0025, 0.0062, -0.0052, 0.0011],
                    [0.0114, -0.0020, 0.0083, -0.0049, -0.0036]])
            """
        """
            self.psi_init = torch.tensor([[1.3241e-02, -3.2873e-03, 2.1914e-03, -3.5665e-03, 3.0396e-03],
                    [2.2630e-02, -4.5659e-03, 8.4805e-03, -9.8064e-03, 4.8920e-03],
                    [3.0546e-02, -6.8578e-03, 1.6058e-02, -1.3890e-02, 3.6751e-03],
                    [3.5994e-02, -6.5442e-03, 1.9254e-02, -1.5745e-02, 2.5962e-03],
                    [4.1836e-02, -7.6067e-03, 1.7880e-02, -1.7123e-02, 4.0315e-03],
                    [4.7332e-02, -9.9279e-03, 1.6646e-02, -1.5698e-02, 3.2995e-03],
                    [4.7046e-02, -9.5479e-03, 1.7541e-02, -1.4100e-02, 2.4895e-03],
                    [5.1201e-02, -1.0664e-02, 1.8088e-02, -1.7150e-02, 2.5443e-03],
                    [5.1577e-02, -1.0687e-02, 1.9934e-02, -1.8237e-02, 1.8204e-03],
                    [5.3111e-02, -1.0175e-02, 1.9042e-02, -2.1886e-02, 1.8629e-03],
                    [5.2803e-02, -9.4859e-03, 1.5879e-02, -2.1794e-02, 1.7772e-03],
                    [5.1623e-02, -9.5832e-03, 1.4279e-02, -2.1145e-02, 2.0649e-03],
                    [5.0558e-02, -8.5654e-03, 1.3740e-02, -2.4224e-02, 1.6591e-03],
                    [4.6013e-02, -8.9082e-03, 1.6332e-02, -2.2734e-02, -9.1706e-04],
                    [4.0235e-02, -9.1908e-03, 1.5203e-02, -2.1442e-02, -9.5916e-06],
                    [3.2870e-02, -1.0070e-02, 1.4380e-02, -1.8260e-02, 2.3187e-04],
                    [2.4728e-02, -8.1625e-03, 1.2901e-02, -1.5636e-02, -1.2447e-03],
                    [1.3228e-02, -5.1193e-03, 7.7156e-03, -7.8512e-03, -1.7013e-03]])
            """
        self.readData = 0
        self.xPoolSize = 11
        ### Input Data Generation ###
        if self.readData == 1:
            print("Pool of input data x was read from file.")
        else:
            print("Pool of input data x generated for this run.")
            self.data_x = torch.normal(self.mean_px, self.sigma_px, size=(self.xPoolSize,))
            self.data_x = 4 * self.sigma_px * torch.rand(size=(self.xPoolSize,)) - 2 * self.sigma_px
            print(self.data_x)
            print(torch.exp(-self.data_x))
            self.data_x = torch.linspace(-1, 1, self.xPoolSize)
        """
            self.psi_init= torch.tensor([[ 0.0822, -0.0846,  0.0489],
            [ 0.1196, -0.1376,  0.0576],
            [ 0.1226, -0.1307,  0.0732],
            [ 0.0778, -0.0930,  0.0460]])
            """
        self.Sigmaa_init = torch.eye(self.nele, self.nele) / 10 ** stdInit  # 10**-8 is good with sigma r = 1
        self.Sigmaa_init = torch.diag(self.Sigmaa_init, 0)
        self.psii = [self.psi_init, self.Sigmaa_init]
        self.temp_res = []
        self.full_temp_res = []
        self.model_time = 0
        self.guide_time = 0
        self.sample_time = 0

    def removeSamples(self):
        self.temp_res = []
        self.full_temp_res = []

    def polynomial(self, x):
        y = torch.zeros(self.poly_pow + 1, 1)
        for k in range(0, self.poly_pow + 1):
            y[k, 0] = x ** k
        return y

    def polynomialResExp(self, x, Nx):
        y = torch.zeros(self.poly_pow + 1, Nx)
        for k in range(0, self.poly_pow + 1):
            y[k, :] = torch.pow(x, k)
        return y

    def executeModel(self, phi_max, sigma_r, sigma_w):
        t0 = time.time()
        if True:
            x = self.data_x[self.modelExecCounter]
        if self.modelExecCounter + 1 < self.Nx_samp:
            self.modelExecCounter = self.modelExecCounter + 1
        else:
            self.modelExecCounter = 0

        if False:
            #x = pyro.sample("x", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
            ii = pyro.sample("ii", dist.Categorical(probs=torch.ones(self.xPoolSize) / self.xPoolSize))
            x = self.data_x[ii]
        y = pyro.sample("y", dist.MultivariateNormal(loc=torch.zeros((1, self.nele)),
                                                     covariance_matrix=sigma_w ** 2 * torch.eye(self.nele, self.nele)))

        phi_max = torch.reshape(phi_max, (1, -1))
        time_test = time.time()
        b = self.pde.calcResKernel(x, y)
        res = torch.matmul(phi_max, b)
        # rwmax2 = torch.matmul(torch.transpose(phi_max, 0, 1), c)
        res = torch.squeeze(res, 1)
        res = torch.squeeze(res, 0)
        # rwmax2 = torch.squeeze(rwmax2, 0)
        res = res
        time_test1 = time.time() - time_test
        start_int = time.time()
        b_test = self.pde.calcSingleRes(x, y, phi_max)
        time_test2 = time.time() - start_int

        self.full_temp_res.append(torch.linalg.norm(b.clone().detach()))
        # self.temp_res.append(torch.linalg.norm(b.clone().detach()))
        if not self.allRes:
            # c = torch.matmul(b, torch.transpose(b, 0, 1))
            #phi_max = torch.reshape(phi_max, (1, -1))
            res = torch.matmul(phi_max, b)
            # rwmax2 = torch.matmul(torch.transpose(phi_max, 0, 1), c)
            res = torch.squeeze(res, 1)
            res = torch.squeeze(res, 0)
            # rwmax2 = torch.squeeze(rwmax2, 0)
            res = res
        else:
            b = torch.squeeze(b, 1)
            res = b
        time_test2 = time.time() - time_test
        self.temp_res.append(abs(res))
        # tess = y.event_shape()
        with pyro.plate("data", self.iter_plate):
            residuals = pyro.sample("residuals", dist.Normal(res, sigma_r), obs=self.residuals)
        self.model_time += time.time() - t0

    def executeGuideDelta(self, phi_max, sigma_r, sigma_w):
        t0 = time.time()
        # x = pyro.sample("x", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
        ii = pyro.sample("ii", dist.Categorical(probs=torch.ones(self.xPoolSize) / self.xPoolSize))
        x = self.data_x[ii]
        mq = pyro.param('mq', self.psii[0])

        polvecx = self.polynomial(x)
        mq_final = torch.matmul(mq, polvecx)

        mq_final = torch.transpose(mq_final, 0, 1)
        # Sigma = pyro.param('Sigma', self.psii[1], constraints.positive)
        # Sigmam = torch.diag(Sigma)
        # y = pyro.sample("y", dist.MultivariateNormal(loc=mq_final,
        #                                            covariance_matrix=Sigmam))
        y = pyro.sample("y", dist.Delta(mq_final, event_dim=1))
        # tess = y.event_shape()
        self.guide_time = self.guide_time + time.time() - t0

    def executeGuideDeltaNoiseless(self, phi_max, sigma_r, sigma_w):
        t0 = time.time()
        x = self.data_x[self.guideExecCounter]
        if self.guideExecCounter + 1 < self.Nx_samp:
            self.guideExecCounter = self.guideExecCounter + 1
        else:
            self.guideExecCounter = 0
        # x = pyro.sample("x", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
        #ii = pyro.sample("ii", dist.Categorical(probs=torch.ones(self.xPoolSize) / self.xPoolSize))

        mq = pyro.param('mq', self.psii[0])

        polvecx = self.polynomial(x)
        mq_final = torch.matmul(mq, polvecx)

        mq_final = torch.transpose(mq_final, 0, 1)
        # Sigma = pyro.param('Sigma', self.psii[1], constraints.positive)
        # Sigmam = torch.diag(Sigma)
        # y = pyro.sample("y", dist.MultivariateNormal(loc=mq_final,
        #                                            covariance_matrix=Sigmam))
        y = pyro.sample("y", dist.Delta(mq_final, event_dim=1))
        # tess = y.event_shape()
        self.guide_time = self.guide_time + time.time() - t0

    def executeGuideMvn(self, phi_max, sigma_r, sigma_w):
        t0 = time.time()
        # x = pyro.sample("x", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
        ii = pyro.sample("ii", dist.Categorical(probs=torch.ones(self.xPoolSize) / self.xPoolSize))
        x = self.data_x[ii]
        mq = pyro.param('mq', self.psii[0])

        polvecx = self.polynomial(x)
        mq_final = torch.matmul(mq, polvecx)

        mq_final = torch.transpose(mq_final, 0, 1)
        Sigma = pyro.param('Sigma', self.psii[1], constraints.positive)
        Sigmam = torch.diag(Sigma)
        y = pyro.sample("y", dist.MultivariateNormal(loc=mq_final,
                                                     covariance_matrix=Sigmam))
        self.guide_time = self.guide_time + time.time() - t0

    def sampleResExpValidation(self):
        # iiii = pyro.sample("iiii", dist.Categorical(probs=torch.ones(self.xPoolSize) / self.xPoolSize))
        # xxx = self.data_x[iiii]
        xxx = self.data_x
        yfinal = torch.zeros((xxx.size(dim=0), self.psii[0].size(dim=0)))
        for i in range(0, xxx.size(dim=0)):
            polvecxx = self.polynomialResExp(xxx[i], 1)
            mqq = torch.matmul(self.psii[0], polvecxx)
            mqq = torch.transpose(mqq, 0, 1)
            yyy = pyro.sample("yyy", dist.Delta(mqq, event_dim=1))
            yfinal[i, :] = yyy
        return yfinal

    def sampleResExpDelta(self, Nx):
        with pyro.plate("data", Nx):
            iiii = pyro.sample("iiii", dist.Categorical(probs=torch.ones(self.xPoolSize) / self.xPoolSize))
            xxx = self.data_x[iiii]
            polvecxx = self.polynomialResExp(xxx, Nx)
            mqq = torch.matmul(self.psii[0], polvecxx)
            mqq = torch.transpose(mqq, 0, 1)
            yyy = pyro.sample("yyy", dist.Delta(mqq, event_dim=1))
            return xxx, yyy

    def sampleResExpMvn(self, Nx):
        with pyro.plate("data", Nx):
            iiii = pyro.sample("iiii", dist.Categorical(probs=torch.ones(self.xPoolSize) / self.xPoolSize))
            xxx = self.data_x[iiii]
            polvecxx = self.polynomialResExp(xxx, Nx)
            mqq = torch.matmul(self.psii[0], polvecxx)
            # mqq = torch.reshape(mqq, (-1, 1))
            mqq = torch.transpose(mqq, 0, 1)
            Sigmaa = self.psii[1]
            # tesss = torch.distributions.multivariate_normal.MultivariateNormal(loc=mqq, covariance_matrix=torch.diag(Sigmaa))
            # tess2 = tesss.batch_shape
            # tess3 = tesss.event_shape
            yyy = pyro.sample("yyy", dist.MultivariateNormal(loc=mqq,
                                                             covariance_matrix=torch.diag(Sigmaa)))
            return xxx, yyy

    def sqResGrad(self, x, phi, A, Psi, v, f):  ### Valid only when all derichlet bcs are zero
        t0 = torch.exp(-x)
        AT = torch.transpose(A, 0, 1)
        grad = 2 * t0 * torch.matmul(torch.reshape(phi, (1, -1)),
                                     torch.matmul(t0 * torch.matmul(A, torch.matmul(Psi, v)) - f))
        grad = grad * torch.outer(torch.matmul(AT, phi), torch.transpose(v, 0, 1))
        return grad

    def validateLikelihoodGrad(self):
        return


class modelDelta2Polynomial:
    def __init__(self, pde, phi_max, poly_pow=None, allRes=True, stdInit=0):
        self.pde = pde
        self.allRes = allRes
        self.nele = pde.effective_nele
        self.mean_px = pde.mean_px
        self.sigma_px = pde.sigma_px
        self.Nx_samp = pde.Nx_samp
        self.Nx_samp_phi = self.Nx_samp
        self.x = torch.zeros(self.Nx_samp_phi, 1)
        self.y = torch.zeros(self.Nx_samp_phi, self.nele)
        self.Nx_counter = 0
        self.phi_max = phi_max
        self.poly_pow = poly_pow
        if not self.allRes:
            self.residuals = torch.tensor(0)
            self.iter_plate = 1
        else:
            self.residuals = torch.zeros(self.nele)
            self.iter_plate = len(self.residuals)
        self.psi_init = torch.rand(self.nele, poly_pow + 1) * 0.01  # Initialization of psi
        self.readData = 0
        self.xPoolSize = 3
        ### Input Data Generation ###
        if self.readData == 1:
            print("Pool of input data x was read from file.")
        else:
            print("Pool of input data x generated for this run.")
            self.data_x = torch.normal(self.mean_px, self.sigma_px, size=(self.xPoolSize,))
            self.data_x = 4 * self.sigma_px * torch.rand(size=(self.xPoolSize,)) - 2 * self.sigma_px
            print(self.data_x)
            print(torch.exp(-self.data_x))
            self.data_x = torch.linspace(-2, 2, self.xPoolSize)
        """
        self.psi_init= torch.tensor([[ 0.0822, -0.0846,  0.0489],
        [ 0.1196, -0.1376,  0.0576],
        [ 0.1226, -0.1307,  0.0732],
        [ 0.0778, -0.0930,  0.0460]])
        """
        self.Sigmaa_init = torch.eye(self.nele, self.nele) / 10 ** stdInit  # 10**-8 is good with sigma r = 1
        self.Sigmaa_init = torch.diag(self.Sigmaa_init, 0)
        self.psii = [self.psi_init, self.Sigmaa_init]
        self.temp_res = []
        self.full_temp_res = []
        self.model_time = 0
        self.guide_time = 0
        self.sample_time = 0

    def removeSamples(self):
        self.Nx_counter = 0
        self.temp_res = []

    def polynomial(self, x):
        y = torch.zeros(self.poly_pow + 1, 1)
        for k in range(0, self.poly_pow + 1):
            y[k, 0] = x ** k
        return y

    def polynomialResExp(self, x, Nx):
        y = torch.zeros(self.poly_pow + 1, Nx)
        for k in range(0, self.poly_pow + 1):
            y[k, :] = torch.pow(x, k)
        return y

    def executeModel(self, phi_max, sigma_r, sigma_w):
        t0 = time.time()
        # x = pyro.sample("x", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
        ii = pyro.sample("ii", dist.Categorical(probs=torch.ones(self.xPoolSize) / self.xPoolSize))
        x = self.data_x[ii]
        y = pyro.sample("y", dist.MultivariateNormal(loc=torch.zeros((1, self.nele)),
                                                     covariance_matrix=sigma_w ** 2 * torch.eye(self.nele,
                                                                                                self.nele)))

        b = self.pde.calcResKernel(x, y)
        self.full_temp_res.append(b.clone().detach())
        # self.temp_res.append(torch.linalg.norm(b.clone().detach()))
        """
        if not self.allRes:
            c = torch.matmul(b, torch.transpose(b, 0, 1))
            phi_max = torch.reshape(phi_max, (-1, 1))
            rwmax2 = torch.matmul(torch.transpose(phi_max, 0, 1), c)
            rwmax2 = torch.squeeze(rwmax2, 1)
            rwmax2 = torch.squeeze(rwmax2, 0)
            res = rwmax2
        """
        if not self.allRes:
            # c = torch.matmul(b, torch.transpose(b, 0, 1))
            phi_max = torch.reshape(phi_max, (1, -1))
            res = torch.matmul(phi_max, b)
            # rwmax2 = torch.matmul(torch.transpose(phi_max, 0, 1), c)
            res = torch.squeeze(res, 1)
            res = torch.squeeze(res, 0)
            # rwmax2 = torch.squeeze(rwmax2, 0)
            res = res
        else:
            b = torch.squeeze(b, 1)
            res = b

        self.temp_res.append(abs(res))

        with pyro.plate("data", self.iter_plate):
            residuals = pyro.sample("residuals", dist.Normal(res, sigma_r), obs=self.residuals)
        self.model_time += time.time() - t0

    def executeGuide(self, phi_max, sigma_r, sigma_w):
        t0 = time.time()
        # x = pyro.sample("x", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
        ii = pyro.sample("ii", dist.Categorical(probs=torch.ones(self.xPoolSize) / self.xPoolSize))
        x = self.data_x[ii]
        mq = pyro.param('mq', self.psii[0])

        polvecx = self.polynomial(x)
        mq_final = torch.matmul(mq, polvecx)

        mq_final = torch.transpose(mq_final, 0, 1)
        mq_final = torch.squeeze(mq_final, 0)
        y = pyro.sample("y", dist.Delta(mq_final, event_dim=1))
        # Sigma = pyro.param('Sigma', self.psii[1], constraints.positive)
        # Sigmam = torch.diag(Sigma)
        # y = pyro.sample("y", dist.MultivariateNormal(loc=mq_final,
        #                                             covariance_matrix=Sigmam))
        self.guide_time = self.guide_time + time.time() - t0

    def sample(self, phi_max, sigma_r, sigma_w):
        t0 = time.time()
        for i in range(0, self.Nx_samp_phi):
            # xx = pyro.sample("xx", dist.Normal(loc=self.mean_px, scale=self.sigma_px))
            iii = pyro.sample("iii", dist.Categorical(probs=torch.ones(self.xPoolSize) / self.xPoolSize))
            xx = self.data_x[iii]
            polvecxx = self.polynomial(xx)
            mqq = torch.matmul(self.psii[0], polvecxx)
            mqq = torch.transpose(mqq, 0, 1)
            Sigmaa = self.psii[1]
            yy = pyro.sample("yy", dist.MultivariateNormal(loc=mqq,
                                                           covariance_matrix=torch.diag(Sigmaa)))

            self.x[self.Nx_counter, 0] = xx.clone().detach()
            for j in range(0, self.nele):
                self.y[self.Nx_counter, j] = yy[0, j].clone().detach()
            self.Nx_counter = self.Nx_counter + 1
        self.sample_time = self.sample_time + time.time() - t0
        self.removeSamples()

    def sampleResExp(self, Nx):
        with pyro.plate("data", Nx):
            iiii = pyro.sample("iiii", dist.Categorical(probs=torch.ones(self.xPoolSize) / self.xPoolSize))
            xxx = self.data_x[iiii]
            polvecxx = self.polynomialResExp(xxx, Nx)
            mqq = torch.matmul(self.psii[0], polvecxx)
            mqq = torch.transpose(mqq, 0, 1)
            mqq = torch.squeeze(mqq, 0)
            yyy = pyro.sample("yyy", dist.Delta(mqq, event_dim=1))
            # Sigmaa = self.psii[1]
            # yyy = pyro.sample("yyy", dist.MultivariateNormal(loc=mqq,
            #                                                 covariance_matrix=torch.diag(Sigmaa)))
            return xxx, yyy

    def sampleResExpValidation(self):
        xxx = pyro.sample("iiii", dist.Delta(self.data_x))
        polvecxx = self.polynomialResExp(xxx, self.xPoolSize)
        mqq = torch.matmul(self.psii[0], polvecxx)
        mqq = torch.transpose(mqq, 0, 1)
        # Sigmaa = self.psii[1]
        mqq = torch.squeeze(mqq, 0)
        yyy = pyro.sample("yyy", dist.Delta(mqq, event_dim=1))
        # yyy = pyro.sample("yyy", dist.MultivariateNormal(loc=mqq,
        #                                                 covariance_matrix=torch.diag(Sigmaa)))
        return xxx, yyy
