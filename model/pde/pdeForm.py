### Importing Libraries ###
import numpy as np

### Import Pyro/Torch Libraries ###
import torch

import os
import fenics as df

smoke_test = ('CI' in os.environ)
import time
from utils.times_and_tags import times_and_tags
import model.pde.shapeFunctions.hatFunctions as hatFuncs


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
    def __init__(self, nele, mean_px, sigma_px, sigma_r, Nx_samp, lBoundDir=None, rBoundDir=None, lBoundNeu=None, rBoundNeu=None, rhs=None):
        self.nele = nele
        self.mean_px = mean_px
        self.sigma_px = sigma_px
        self.sigma_r = sigma_r
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
        self.intPoints = 10001
        self.s_integration = torch.linspace(0, 1, self.intPoints)
        self.us = hatFuncs.hat(self.s_integration, self.s, self.dl)
        self.dus = hatFuncs.hatGrad(self.s_integration, self.s, self.dl)
        self.int_us = torch.trapezoid(self.us, self.s_integration, dim=1)
        self.int_dusdus = torch.trapezoid(torch.einsum('kz,jz->kjz', self.dus, self.dus), self.s_integration, dim=2)


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

    def cWeightFunc(self, phi, us):
        """
        :param phi: The coefficients of the weighting functions
        :param us: A multidimensional tensor repsresenting the shape function, the 1st dim is always the one
        to be multiplied with the dim of phi
        :return: The weight function w(s)
        """
        return torch.matmul(torch.reshape(phi, [1, -1]), us)
    def cTrialFunc(self, y, us):
        """
        :param y: The coefficients of the Trial/Solution Function
        :param us: A multidimensional tensor repsresenting the shape function, the 1st dim is always the one
        to be multiplied with the dim of y
        :return: The Trial/Solution function y(s)
        """
        return torch.matmul(torch.reshape(y, [1, -1]), us)
    def cParInputFunc(self, x, us):
        """
        :param x: The coefficients of the Parametric Input function c(x,s), that exists in the strong form.
        :param us: A multidimensional tensor repsresenting the shape function, the 1st dim is always the one
        to be multiplied with the dim of x
        :return: The Parametric Input function c(x,s)
        """
        return torch.matmul(torch.reshape(x, [1, -1]), us)
    def calcSingleRes(self, x, y, phi):
        x = torch.exp(x)  ### Form of Cs(x) = exp(x)
        phi = torch.cat((torch.tensor([[0]]), phi, torch.tensor([[0]])), dim=1)
        y = torch.cat((torch.tensor([[0]]), y, torch.tensor([[0]])), dim=1)
        res = - x * torch.matmul(torch.matmul(phi, self.int_dusdus), torch.reshape(y, [-1, 1])) - self.rhs * torch.matmul(phi, self.int_us)
        res = torch.squeeze(res, dim=1)
        res = torch.squeeze(res, dim=0)
        return res

    def calcSingleResGeneral(self, x, y, phi):
        x = torch.exp(x)  ### Form of Cs(x) = exp(x)
        phi = torch.cat((torch.tensor([[0]]), phi, torch.tensor([[0]])), dim=1)
        y = torch.cat((torch.tensor([[0]]), y, torch.tensor([[0]])), dim=1)
        res = torch.trapezoid(- x * torch.mul(self.cWeightFunc(phi, self.dus), self.cTrialFunc(y, self.dus)) \
              - self.rhs * self.cWeightFunc(phi, self.us), self.s_integration, dim=1)
        return torch.squeeze(res, dim=0)

    def calcResKernel(self, x, y): # x is scalar and y is 1D or 2D vector
        x = torch.exp(x)  ### Form of Cs(x) = exp(x)
        start = time.time()
        y = torch.reshape(y, (-1, 1))
        start = time.time()
        if self.rhs is None:
            b = self.u + x * self.a - x * torch.matmul(self.A, y)
        elif isinstance(self.rhs, int) or isinstance(self.rhs, float):
            b = self.u + x * self.a - x * torch.matmul(self.A, y) - self.f

        return b

    def calcResKernelSingleInput(self, x): # x is scalar and y is 1D or 2D vector
        x = torch.exp(x[:,0])  ### Form of Cs(x) = exp(x)
        y = torch.reshape(x[1:,:], (-1, 1))
        if self.rhs is None:
            b = self.u + x * self.a - x * torch.matmul(self.A, y)
        elif isinstance(self.rhs, int) or isinstance(self.rhs, float):
            b = self.u + x * self.a - x * torch.matmul(self.A, y) - self.f
        return b
