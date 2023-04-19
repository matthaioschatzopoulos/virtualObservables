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
from model.pde.numIntegrators.trapezoid import trapzInt2D
import matplotlib.pyplot as plt


class pdeForm:
    def __init__(self, nele, mean_px, sigma_px, sigma_r, Nx_samp, bnd_list, rhs=None):
        self.nele = nele
        self.grid_x, self.grid_y = torch.meshgrid(torch.linspace(0, 1, nele+1), torch.linspace(0, 1, nele+1), indexing='ij')
        self.grid = torch.stack((self.grid_x, self.grid_y), dim=0)
        self.NofShFuncs = torch.reshape(self.grid_x, [-1]).size(dim=0)
        self.node_corrs = torch.reshape(self.grid, [2, -1])
        self.dl = 1 / self.nele
        self.intPoints = 51
        self.sgrid_x, self.sgrid_y = torch.meshgrid(torch.linspace(0, 1, self.intPoints), torch.linspace(0, 1, self.intPoints),
                                                  indexing='ij')
        self.sgrid = torch.stack((self.sgrid_x, self.sgrid_y), dim=0)
        self.shapeFunc = hatFuncs.rbfInterpolation(self.grid, self.sgrid, 1/self.dl**2)
        self.wDir = self.DirichletFilterLikeWANs()

        self.mean_px = mean_px
        self.sigma_px = sigma_px
        self.sigma_r = sigma_r
        self.Nx_samp = Nx_samp
        self.A = torch.zeros((self.nele+1, self.nele+1))
        self.u = torch.zeros((self.nele +1, 1))
        self.a = torch.zeros((self.nele +1, 1))
        self.bnd_low = bnd_list[0]
        self.bnd_up = bnd_list[1]
        self.bnd_right = bnd_list[2]
        self.bnd_left = bnd_list[3]
        self.rhs = rhs
        self.effective_nele = None
        self.f = None
        self.s = torch.linspace(0, 1, nele+1)
        self.systemRhs = None
        #self.plotShapeFunctions()

        """
        self.s_integration = torch.linspace(0, 1, self.intPoints)
        self.us = hatFuncs.hat(self.s_integration, self.s, self.dl)
        self.dus = hatFuncs.hatGrad(self.s_integration, self.s, self.dl)
        self.int_us = torch.trapezoid(self.us, self.s_integration, dim=1)
        self.int_dusdus = torch.trapezoid(torch.einsum('kz,jz->kjz', self.dus, self.dus), self.s_integration, dim=2)
        """

        """
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
        """
    def DirichletFilterLikeWANs(self):
        """
        :return: The distance matrix w(x), which is described in the WANs paper. This will be multiplied by the weighting
        funtion w(s) to make sure that w=0 at the boundaries, when Diriclet conditions are applied.
        """
        reshaped_grid = torch.reshape(self.sgrid, [2, -1])
        boundary_points = []
        for i in range(0, torch.reshape(self.sgrid, [2, -1]).size(dim=1)):
            if reshaped_grid[0, i] == 0. or reshaped_grid[0, i] == 1. or reshaped_grid[1, i] == 0. or reshaped_grid[1, i] == 1.:
                boundary_points.append(torch.tensor([reshaped_grid[0, i], reshaped_grid[1, i]]))
        boundary_points = torch.stack(boundary_points)
        distFromBndPoints = []
        for i in range(0, boundary_points.size(dim=0)):
            distFromBndPoints.append(torch.sqrt((reshaped_grid[0, :] - boundary_points[i, 0])**2 +
                                                (reshaped_grid[1, :] - boundary_points[i, 1])**2))
        distFromBndPoints = torch.stack(distFromBndPoints)
        return torch.reshape(torch.min(distFromBndPoints, dim=0)[0], [self.intPoints, -1])

    def plotWeightFunc(self, phi):
        plt.pcolormesh(self.sgrid[0, :, :], self.sgrid[1, :, :], self.shapeFunc.cWeighFunc(phi), cmap='coolwarm', shading='auto')
        plt.colorbar()
        plt.show()

    def plotSolution(self, y):
        t1 = self.shapeFunc.cTrialSolution(y)
        plt.pcolormesh(self.sgrid[0, :, :], self.sgrid[1, :, :], self.shapeFunc.cTrialSolution(y), cmap='coolwarm',
                       shading='auto')
        plt.colorbar()
        plt.show()


        def plotShapeFunctions(self):
            plt.pcolormesh(self.sgrid[0, :, :], self.sgrid[1, :, :], self.shapeFunc.shapeFunc[3, :, :], cmap='coolwarm',
                           shading='auto')
            plt.colorbar()
            plt.show()

    def plotTrueSolution(self):
        csv_data = np.loadtxt('./model/pde/trueSol51_x=0.csv', delimiter=',')

        # Convert the numpy array to a torch tensor
        y = torch.reshape(torch.from_numpy(csv_data), [self.intPoints, -1])
        plt.pcolormesh(self.sgrid[0, :, :], self.sgrid[1, :, :], y, cmap='coolwarm',
                       shading='auto')
        plt.colorbar()
        plt.show()

    def plotAnalSolution(self):

        plt.pcolormesh(self.sgrid[0, :, :], self.sgrid[1, :, :],
                       -0.25*(self.sgrid[0, :, :]*(1-self.sgrid[0, :, :]) + self.sgrid[1, :, :]*(1-self.sgrid[1, :, :]))
                       , cmap='coolwarm', shading='auto')
        plt.colorbar()
        plt.show()

    def plotError(self, y):
        csv_data = np.stack((np.loadtxt('./model/pde/trueSol51_x=-1.csv', delimiter=','),
                            np.loadtxt('./model/pde/trueSol51_x=0.csv', delimiter=','),
                            np.loadtxt('./model/pde/trueSol51_x=1.csv', delimiter=',')), axis=0)
        csv_data = torch.from_numpy(csv_data)
        yTrue = torch.reshape(csv_data, [3, self.intPoints, -1])


        fig, axs = plt.subplots(1, 3, figsize=(9, 4))

        for i in range(3):
                t0 = y[i,:]
                t1 = torch.div(self.shapeFunc.cTrialSolution(y[i,:])-yTrue[i, :, :], (yTrue[i, :, :]+10**(-6)))
                torch.set_printoptions(profile='full')
                print(t1)
                axs[i].pcolormesh(self.sgrid[0, :, :], self.sgrid[1, :, :],
                                  torch.abs(torch.div(self.shapeFunc.cTrialSolution(y[i,:])-yTrue[i, :, :], (yTrue[i, :, :]+10**(-6)))),
                                  cmap='coolwarm', shading='auto')
                axs[i].set_title("Input x= "+str(i-1))
                axs[i].set_aspect('equal')
        cbar_ax = fig.add_axes([0.15, 0.10, 0.7, 0.03])
        fig.colorbar(axs[0].collections[0], cax=cbar_ax, orientation='horizontal', pad=0.)
        fig.subplots_adjust(bottom=0.15)

        plt.tight_layout()
        plt.show()

    def plotErrorMetrics(self, y):
        csv_data = np.stack((np.loadtxt('./model/pde/trueSol51_x=-1.csv', delimiter=','),
                             np.loadtxt('./model/pde/trueSol51_x=0.csv', delimiter=','),
                             np.loadtxt('./model/pde/trueSol51_x=1.csv', delimiter=',')), axis=0)
        csv_data = torch.from_numpy(csv_data)
        yTrue = torch.reshape(csv_data, [3, self.intPoints, -1])
        err = []
        meanPred = []
        meanTrue = []
        for i in range(0, y.size(dim=0)):
            err.append(torch.mean(torch.abs((self.shapeFunc.cTrialSolution(y[i, :]) -
                                             yTrue[i, :, :]))/(yTrue[i, :, :]+10**(-6))))
            meanTrue.append(torch.mean(yTrue[i, :, :]))
            meanPred.append(torch.mean(self.shapeFunc.cTrialSolution(y[i, :])))
        err = torch.stack(err)
        meanTrue = torch.stack(meanTrue)
        meanPred = torch.stack(meanPred)
        plt.plot([-1, 0, 1], meanPred, 'b')
        plt.plot([-1, 0, 1], meanTrue, 'r')
        plt.grid(True)
        plt.show()

        return err, meanTrue, meanPred


    def NodeToElementsMapping(self): # Not Used Currently
        self.totalNodes = (self.nele + 1)**2
        self.totalElem = (self.nele) ** 2
        self.elemList = torch.zeros(self.totalElem, 4)
        el_index = 0
        for i in range(0, self.nele):
            for j in range(0, self.nele):
                self.elemList[el_index, :] = torch.tensor([el_index+0+i, el_index+1+i,
                                                           el_index+(self.nele+1)+i, el_index+(self.nele+1)+1+i])
                el_index += 1#

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

    def calcSingleRes(self, x, y, phi):
        x = torch.exp(x)  ### Form of Cs(x) = exp(x)
        phi = torch.cat((torch.tensor([[0]]), phi, torch.tensor([[0]])), dim=1)
        y = torch.cat((torch.tensor([[0]]), y, torch.tensor([[0]])), dim=1)
        res = - x * torch.matmul(torch.matmul(phi, self.int_dusdus), torch.reshape(y, [-1, 1])) - self.rhs * torch.matmul(phi, self.int_us)
        res = torch.squeeze(res, dim=1)
        res = torch.squeeze(res, dim=0)
        return res

    def calcSingleResGeneral(self, x, y, phi):
        """
        :param x: Coefficients of the expansion for evaluating the input function (1D tensor).
        :param y: Coefficients of the expansion for evaluating the trial/solution function (1D tensor).
        :param phi: Coefficients of the expansion for evaluating the weighting function (1D tensor).
        :return: Single residual from the weak form.
        Important Theoritical Question: I have implemented dw/ds*dy/ds as a dot product between the grads, is it correct?
        """
        x = torch.exp(x)  ### Form of Cs(x) = exp(x)

        res = trapzInt2D((- x * torch.einsum('ijk,ijk->jk', self.shapeFunc.cdWeighFunc(phi), self.shapeFunc.cdWeighFunc(y)) \
              + self.rhs * self.shapeFunc.cWeighFunc(phi)))
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
