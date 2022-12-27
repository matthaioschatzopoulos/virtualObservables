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
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, Predictive, TraceGraph_ELBO
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
import time
from textwrap import wrap


class pdeForm:
    def __init__(self, nele, mean_px, sigma_px, Nx_samp, lBoundDir=None, rBoundDir=None, lBoundNeu=None, rBoundNeu=None, rhs=None):
        self.nele = nele
        self.mean_px = mean_px
        self.sigma_px = sigma_px
        self.Nx_samp = Nx_samp
        self.A = torch.zeros((self.nele+1, self.nele+1))
        self.u = torch.zeros((self.nele +1, 1))
        self.a = torch.zeros((self.nele +1, 1))
        self.lBoundDir = lBoundDir
        self.rBoundDir = rBoundDir
        self.lBoundNeu = lBoundNeu
        self.rBoundNeu = rBoundNeu
        self.rhs = rhs
        self.effective_nele = None
        self.f = None
        self.dl = 1/self.nele
        self.s = torch.linspace(0, 1, nele+1)
        self.systemRhs = None

        ### Building matrix A ###

        for i in range(0, nele+1):
            for j in range(0, nele + 1):
                if i == j:
                    self.A[i, j] = 2/self.dl
                elif j == i + 1:
                    self.A[i, j] = -1/self.dl
                elif j == i - 1:
                    self.A[i, j] = -1/self.dl
        self.A[self.nele, self.nele] = self.A[self.nele, self.nele] -1/self.dl
        self.A[0, 0] = self.A[0, 0] -1/self.dl

        #for numerical stability
        #self.A = self.A * self.dl


        ### Building rhs matrix f ###
        if isinstance(self.rhs, int) or isinstance(self.rhs, float):
            self.f = torch.reshape(self.A[:, 0], (-1, 1)) * 0.
            self.f[0, 0] = self.dl/2
            self.f[-1, 0] = self.dl/2
            for i in range(1, nele):
                self.f[i] = self.dl
            self.f = self.f * self.rhs
        # for numerical stability
        #self.f = self.f * self.dl
        self.createEquations()


    def createEquations(self):
        if self.rBoundNeu is not None:
            self.u[self.nele, 0] = self.rBoundNeu
        if self.lBoundNeu is not None:
            self.u[0, 0] = -self.lBoundNeu
        if self.rBoundDir is not None:
            self.A = self.A[0:self.nele, 0:self.nele]
            self.u = torch.reshape(self.u[0:self.nele, 0], (-1, 1))
            self.a = torch.reshape(self.a[0:self.nele, 0], (-1, 1))
            if isinstance(self.rhs, int) or isinstance(self.rhs, float):
                self.f = torch.reshape(self.f[0:self.nele, 0], (-1, 1))
            self.a[-1, 0] = self.rBoundDir/self.dl
        if self.lBoundDir is not None:
            self.A = self.A[1:, 1:]
            self.u = torch.reshape(self.u[1:, 0], (-1, 1))
            self.a = torch.reshape(self.a[1:, 0], (-1, 1))
            if isinstance(self.rhs, int) or isinstance(self.rhs, float):
                self.f = torch.reshape(self.f[1:, 0], (-1, 1))
            self.a[0, 0] = self.lBoundDir/self.dl
        self.effective_nele = self.A.size(dim=0)
        # for numerical stability
        #self.u = self.u * self.dl
        #self.a = self.a * self.dl
        if self.rhs is None:
            self.systemRhs = self.u + self.a
        elif isinstance(self.rhs, int) or isinstance(self.rhs, float):
            self.systemRhs = self.u + self.a - self.f




    def calcResKernel(self, x, y): # x is scalar and y is 1D or 2D vector
        x = torch.exp(x)  ### Form of Cs(x) = exp(x)
        y = torch.reshape(y, (-1, 1))
        if self.rhs is None:
            b = self.u + x * self.a - x * torch.matmul(self.A, y)
        elif isinstance(self.rhs, int) or isinstance(self.rhs, float):
            b = self.u + x * self.a - x * torch.matmul(self.A, y) - self.f
        return b
