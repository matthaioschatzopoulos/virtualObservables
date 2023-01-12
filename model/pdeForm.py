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
import fenics as df
import logging
from torch.distributions import constraints

smoke_test = ('CI' in os.environ)
from torch.distributions import constraints
from pyro.infer import Predictive
import time
from textwrap import wrap
from utils.times_and_tags import times_and_tags

class pdeFenics:
    def __init__(self, nele, mean_px, sigma_px, Nx_samp, lBoundDir=None, rBoundDir=None, lBoundNeu=None, rBoundNeu=None,
                 rhs=None):
        self.nele = nele
        self.mean_px = mean_px
        self.sigma_px = sigma_px
        self.Nx_samp = Nx_samp
        self.A = torch.zeros((self.nele + 1, self.nele + 1))
        self.u = torch.zeros((self.nele + 1, 1))
        self.a = torch.zeros((self.nele + 1, 1))
        self.lBoundDir = lBoundDir
        self.rBoundDir = rBoundDir
        self.lBoundNeu = lBoundNeu
        self.rBoundNeu = rBoundNeu
        self.rhs = rhs
        self.effective_nele = None
        self.f = None
        self.dl = 1 / self.nele
        self.s = torch.linspace(0, 1, nele + 1)
        self.systemRhs = None

        # %% General setup
        # Create mesh and define function space
        mesh = df.IntervalMesh(100, 0.0, 1.0)
        # displacement FunctionSpace
        V = df.FunctionSpace(mesh, "CG", 1)
        # material function space
        Vc = df.FunctionSpace(mesh, "DG", 0)

        # %% Boundary conditions
        # Define boundary condition
        u_D_left = df.Expression(str(lBoundDir), degree=0)  # x is domain coordinates
        u_D_right = df.Expression(str(rBoundDir), degree=0)  # x is domain coordinates

        tol = 1e-6
        def onboundary(x, on_boundary):
            return on_boundary
        def boundary_left(x, on_boundary):
            return on_boundary and (df.near(x[0], 0, tol))

        def boundary_right(x, on_boundary):
            return on_boundary and (df.near(x[0], 1, tol))

        bc1 = df.DirichletBC(V, u_D_left, boundary_left)
        bc2 = df.DirichletBC(V, u_D_right, boundary_right)
        bc = [bc1, bc2]
        bcZeroDir0 = df.DirichletBC(V, df.Expression('0', degree=0), onboundary)
        bcZeroDir1 = df.DirichletBC(V, df.Expression('1', degree=0), onboundary)


        # %% FE stuff
        # Define variational problem
        w = df.TrialFunction(V)
        y = df.TestFunction(V)

        # a random constant
        force = df.Constant(-rhs)

        # weak formulation
        a = df.dot(df.grad(w), df.grad(y)) * df.dx
        L = force * y * df.dx

        # Compute solution
        y = df.Function(V)

        # I want it to give it my custom vector
        y_numpy = np.random.rand(11)
        # y_fun = df.Function(V)
        # y_fun.vector().set_local(y_numpy)
        timer = times_and_tags()
        timer.add("Assembly1")
        A, b = df.assemble_system(a, L, bc)
        timer.add("Assembly2")
        A, b = df.assemble_system(a, L, bc)




        Adir0, f = df.assemble_system(a, L, bcZeroDir0)
        Adir1, bdir1 = df.assemble_system(a, L, bcZeroDir1)
        f = torch.from_numpy(f.get_local())
        A = torch.from_numpy(A.array())
        b = torch.from_numpy(b.get_local())
        timer.add("matmul")
        resulttt = np.matmul(A, b) - b
        timer.print()
        a = b - f
        # u = ...
        print(A)
        print(b)
        print(a)
        print(f)

        aa = np.array([[5], [0], [0], [5]])
        f = -1 * np.array([[0.2], [0.2], [0.2], [0.2]])
        b = 2. * aa - f
        print(b)

        def testAssembly(self):
            print('Comparing A, a, u, f with the actual assembly matrix from fenics')



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
