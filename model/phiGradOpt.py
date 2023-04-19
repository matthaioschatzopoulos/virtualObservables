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
    def __init__(self, pde, poly_pow=None, eigRelax=None, Iter_grad=1, gradLr=0.1, validationMode=False,
                 powIterTol=None, Nx_samp_phi=None, runTests=False):
        self.pde = pde
        self.model = None
        self.validationMode = validationMode
        self.eigRelax = eigRelax
        self.Nx_samp_phiOpt = Nx_samp_phi # 100 is usually a good value
        self.gradLr = gradLr
        self.epoch = Iter_grad
        self.powIterTol = powIterTol
        self.nele = pde.nele
        self.mean_px = pde.mean_px
        self.sigma_px = pde.sigma_px
        self.Nx_samp = pde.Nx_samp
        self.Nx_samp_phi = self.Nx_samp
        self.x = torch.zeros(self.Nx_samp_phi, 1)
        self.y = torch.zeros(self.Nx_samp_phi, self.nele)
        self.Nx_samp = self.Nx_samp
        self.Nx_counter = 0
        self.phi_max = torch.rand(self.pde.NofShFuncs)*0.01+torch.ones(self.pde.NofShFuncs)
        self.phi_max = self.phi_max/torch.linalg.norm(self.phi_max)
        self.poly_pow = poly_pow
        self.temp_res = torch.rand(self.Nx_samp).tolist()
        self.full_res_temp = torch.rand(self.Nx_samp).tolist()
        for i in range(0, len(self.full_res_temp)):
            self.full_res_temp[i] = torch.rand(self.nele)*0.1
        # First entry is True if the test has been passed the second is true if the test has been conducted
        self.testEigOpt = [True, runTests, 2] # 1-->Passed Maximization test, 2--> runTests, 3-->Passed MC test
        self.relImp = []
        self.N = self.nele + 2
        self.M = 5
        self.F_getPhi = self.assemblyF(self.N, self.M)
        self.F_getAlpha = self.assemblyF(self.M, self.N)


    def rbf(self, s, s0, tau=10):
        return torch.exp(-tau*torch.linalg.norm(s-s0))

    def assemblyF(self, N, M):
        """
        :param N: Dimension of high dimensional vector (Equal to the number of all the actual nodes)
        :param M: Dimension of low dimensional vector (E.G. M=3 means imaginary nodes at s=0, s=0.5, s=1.0)
        :return: Matrix F which if multiply by left the low dimensional vector gives the high dimensional vector
        """
        sN = torch.linspace(0, 1, N)
        sN = sN[1:N-1]
        sM = torch.linspace(0, 1, M)
        sM = sM[1:M-1]
        F = torch.zeros((sN.size(dim=0), sM.size(dim=0)))
        for i in range(0, F.size(dim=0)):
            for j in range(0, F.size(dim=1)):
                F[i, j] = self.rbf(sN[i], sM[j], tau=1)
        return F
    def calcC(self, Nx):
        C = torch.tensor(0)
        if self.model.surgtType == 'Delta' and self.validationMode is False:
            x, y = self.model.sampleResExpDelta(Nx)
        elif self.model.surgtType == 'Delta' or self.model.surgtType == 'DeltaNoiseless'\
                and self.validationMode is True:
            x = self.model.data_x
            y = self.model.sampleResExpValidation()
        elif self.model.surgtType == 'Mvn' and self.validationMode is False:
            x, y = self.model.sampleResExpDelta(Nx)
        for jj in range(0, x.size(dim=0)):
            b = self.pde.calcResKernel(x[jj], y[jj, :])
            C = C + torch.matmul(torch.reshape(b, (-1, 1)), torch.reshape(b, (1, -1)))
        return C / x.size(dim=0)

    def calcCGeneral(self, Nx, phi):
        res = 0.
        if self.model.surgtType == 'Delta' and self.validationMode is False:
            x, y = self.model.sampleResExpDelta(Nx)
        elif self.model.surgtType == 'Delta' or self.model.surgtType == 'DeltaNoiseless'\
                and self.validationMode is True:
            x = self.model.data_x
            y = self.model.sampleResExpValidation()
        elif self.model.surgtType == 'Mvn' and self.validationMode is False:
            x, y = self.model.sampleResExpDelta(Nx)

        for jj in range(0, x.size(dim=0)):
            resid = self.pde.calcSingleResGeneral(x[jj], y[jj, :], phi)
            res += resid**2

        return res / x.size(dim=0)
    def calcSqRes(self, phi, C):
        sqRes = torch.matmul(torch.matmul(torch.reshape(phi, (1, -1)), C),
                     torch.reshape(phi, (-1, 1)))
        return sqRes
    def phiGradOptTest(self, loss, phiOld):
        ### MCtest ###
        """
        if self.testEigOpt[2] == 2:
            res = torch.zeros(3)
            for i in range(0, 3):
                res[i] = self.calcSqRes(phiOld, self.calcC(int(10**(i+2)))).item()

            if abs(res[2]-res[1])/abs(res[2]) < 0.1:
                self.testEigOpt[2] = True
        """
        ### Maximazation of SqResidual test ###
        sqResOld = loss[0]
        #sqRes1 = self.calcSqRes(phiOld / torch.linalg.norm(phiOld), self.calcC(self.Nx_samp_phiOpt))
        #sqRes2 = self.calcSqRes(phiOld / torch.linalg.norm(phiOld), self.calcC(self.Nx_samp_phiOpt))
        #sqRes3 = self.calcSqRes(phiOld / torch.linalg.norm(phiOld), self.calcC(self.Nx_samp_phiOpt))
        sqResNew = loss[-1]
        self.relImp.append(((sqResNew - sqResOld)/sqResOld).item())
        #Old test: if sqResNew < sqResOld or sqResUpd < sqResOld:
        if (sqResNew - sqResOld)/sqResOld < -0.01:
            self.testEigOpt[0] = False
        return
    def manualGrad(self, phi, C):
        f = torch.reshape(phi, (-1, 1))
        t0 = torch.linalg.norm(phi)
        t1 = 1/t0**2
        t2 = torch.matmul(torch.transpose(C, 0, 1), f)
        grad = -(t1 * torch.matmul(C, f) + t1*t2 - 2/t0 ** 4 * torch.matmul(torch.transpose(f, 0, 1), t2) * f)
        return grad
    def getPhiGrad(self):
        #phi_max = torch.reshape(self.phi_max, (-1, 1))
        phi_max_old = self.phi_max
        phi_max_leaf = self.phi_max.clone().detach().requires_grad_(True)
        model_phi = [phi_max_leaf]
        optimizer = torch.optim.Adam(model_phi, lr=self.gradLr, maximize=True)
        optimizer.zero_grad()
        sqRes = self.calcCGeneral(self.Nx_samp_phiOpt, model_phi[0] / torch.linalg.norm(model_phi[0]))
        loss = sqRes
        loss.backward(retain_graph=True)
        grad = -1/(2*self.pde.sigma_r**2)*model_phi[0].grad
        return grad

    def phiGradOpt(self):
        #phi_max = torch.reshape(self.phi_max, (-1, 1))
        phi_max_old = self.phi_max
        phi_max_leaf = self.phi_max.clone().detach().requires_grad_(True)
        model_phi = [phi_max_leaf]
        optimizer = torch.optim.Adam(model_phi, lr=self.gradLr, maximize=True)
        loss_history = []

        ### Gradient optimization loops
        for epoch in range(0, self.epoch):
            optimizer.zero_grad()
            """ Old Method for 1D problem
            sqRes = self.calcSqRes(model_phi[0] / torch.linalg.norm(model_phi[0]), self.calcC(self.Nx_samp_phiOpt))
            """
            sqRes = self.calcCGeneral(self.Nx_samp_phiOpt, model_phi[0] / torch.linalg.norm(model_phi[0]))
            # sqRes1 = self.calcSqRes(model_phi[0] / torch.linalg.norm(model_phi[0]), self.calcCValidation())
            # sqRes2 = self.calcSqRes(model_phi[0] / torch.linalg.norm(model_phi[0]), self.calcCValidation())
            # sqRes3 = self.calcSqRes(model_phi[0], self.calcCValidation())
            # sqRes3 = self.calcSqRes(model_phi[0] / torch.linalg.norm(model_phi[0]), self.calcC(self.Nx_samp_phiOpt))
            # sqResTest = self.calcSqRes(model_phi[0]/torch.linalg.norm(model_phi[0]), self.calcC(10000))
            # sqRes = self.calcSqRes(model_phi[0]/torch.linalg.norm(model_phi[0]), self.calcCValidation())

            # manDer = -2*resid2/torch.linalg.norm(model_phi[0])*b # manual derivative for Nx = 1 for comparison reasons
            # manDer = -1/2/torch.sqrt(torch.matmul(torch.transpose(model_phi[0], 0, 1),model_phi[0]))*2*model_phi[0]
            ### Manual derivative implementation for comparison ###
            # manDer = -1/2*(torch.sum(model_phi[0]**2))**(-3/2)*2*model_phi[0]*torch.matmul(torch.transpose(model_phi[0],0,1),b)
            # manDer = manDer + 1/torch.linalg.norm(model_phi[0])*b
            # manDer = -manDer * 2 * torch.sqrt(resid2)
            ### Manual derivative implementation for comparison ###
            # if resid2 > 2:
            #    print("Warning: Residual= ","{:2f}".format(float(resid2))," > 1 in phiGradOptDelta()")
            # loss = criterion(resid2, torch.tensor([[0]]).to(torch.float32))
            # loss = - resid2
            # loss = -torch.linalg.norm(model_phi[0])
            loss = sqRes
            loss.backward(retain_graph=True)
            # validationGrad = self.manualGrad(phi_max_old, self.calcC(self.Nx_samp_phiOpt))
            # gradd = torch.autograd.grad(loss, model_phi_nml, retain_graph=True)
            optimizer.step()
            # print("Is model_phi[0] leaf tensor:", model_phi[0].is_leaf)
            # print("Is loss leaf tensor:", loss.is_leaf)
            # print("resid2: ","{:8f}".format(float(resid2)))
            # print("loss: ","{:8f}".format(float(loss)))
            # print("grad: ",model_phi[0].grad) ## Attention! This is dloss/dphi
            # print("manual grad:", manDer)
            # print("phi_max: ",model_phi[0])
            # print("loss: ",loss)
            loss_history.append(loss)
        loss_history.append(
            self.calcCGeneral(self.Nx_samp_phiOpt, model_phi[0] / torch.linalg.norm(model_phi[0])))
        phi_max =  model_phi[0] / torch.linalg.norm(model_phi[0])
        # self.phiEigOpt()
        #phi_max = torch.squeeze(phi_max, 1)
        self.phiGradOptTest(loss_history, phi_max_old)
        self.phi_max = phi_max
        return phi_max.clone().detach()


    def phiGradOptR(self):
        phi_max = torch.reshape(self.phi_max, (-1, 1))
        """
        #### Experimental part of code | Reducing phi_max dimensions for the optimization ###
        phi_max = torch.from_numpy(np.interp(np.linspace(0, self.nele, 5), np.linspace(0, self.nele, self.nele),
                          torch.squeeze(phi_max, dim=1).numpy()))
        phi_max = torch.reshape(phi_max, (-1, 1))
        #### Experimental part of code | Reducing phi_max dimensions for the optimization ###
        """
        phi_max_old = phi_max
        phi_max = torch.matmul(self.F_getAlpha, phi_max)
        phi_max = phi_max / torch.linalg.norm(phi_max)
        #phi_max = torch.tensor([[25.], [50.], [100.], [150.], [120.]])
        #phi_max = phi_max / torch.linalg.norm(phi_max)
        tess = torch.matmul(self.F_getPhi, phi_max)
        tess = tess / torch.linalg.norm(tess)
        phi_max_leaf = phi_max.clone().detach().requires_grad_(True)
        model_phi = [phi_max_leaf]
        optimizer = torch.optim.Adam(model_phi, lr=self.gradLr, maximize=True)
        loss_history = []

        ### Gradient optimization loops
        for epoch in range(0, self.epoch):
            optimizer.zero_grad()
            tess = torch.matmul(self.F_getPhi, model_phi[0] / torch.linalg.norm(model_phi[0]))
            tess = tess
            sqRes = self.calcSqRes(tess, self.calcC(self.Nx_samp_phiOpt))
            #sqRes = self.calcSqRes(model_phi[0] / torch.linalg.norm(model_phi[0]), self.calcC(self.Nx_samp_phiOpt))
            #sqRes1 = self.calcSqRes(model_phi[0] / torch.linalg.norm(model_phi[0]), self.calcCValidation())
            #sqRes2 = self.calcSqRes(model_phi[0] / torch.linalg.norm(model_phi[0]), self.calcCValidation())
            #sqRes3 = self.calcSqRes(model_phi[0], self.calcCValidation())
            #sqRes3 = self.calcSqRes(model_phi[0] / torch.linalg.norm(model_phi[0]), self.calcC(self.Nx_samp_phiOpt))
            #sqResTest = self.calcSqRes(model_phi[0]/torch.linalg.norm(model_phi[0]), self.calcC(10000))
            #sqRes = self.calcSqRes(model_phi[0]/torch.linalg.norm(model_phi[0]), self.calcCValidation())

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
            #loss = -torch.linalg.norm(model_phi[0])
            loss = sqRes
            loss.backward(retain_graph=True)
            #validationGrad = self.manualGrad(phi_max_old, self.calcC(self.Nx_samp_phiOpt))
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
        tess = torch.matmul(self.F_getPhi, model_phi[0] / torch.linalg.norm(model_phi[0]))
        tess = tess / torch.linalg.norm(tess)
        loss_history.append(self.calcSqRes(tess, self.calcC(self.Nx_samp_phiOpt)))
        phi_max = tess
        #self.phiEigOpt()
        phi_max = torch.squeeze(phi_max, 1)
        self.phiGradOptTest(loss_history, phi_max_old)
        self.phi_max = phi_max
        return phi_max.clone().detach()


    def phiEigOptTest(self, phiOld, phiNew, phiUpd, C):
        ### MCtest ###
        if self.testEigOpt[2] == 2:
            res = torch.zeros(3)
            for i in range(0, 3):
                res[i] = self.calcSqRes(phiOld, self.calcC(int(10**(i+2)))).item()

            if abs(res[2]-res[1])/abs(res[2]) < 0.1:
                self.testEigOpt[2] = True
        ### Maximazation of SqResidual test ###
        sqResOld = self.calcSqRes(phiOld, C).item()
        sqResNew = self.calcSqRes(phiNew, C).item()
        sqResUpd = self.calcSqRes(phiUpd, C).item()

        #Old test: if sqResNew < sqResOld or sqResUpd < sqResOld:
        if (sqResOld - sqResNew)/sqResOld > 0.05:   ### Simpler is to check sqResNew < sqResOld
            self.testEigOpt[0] = False
        return

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


        ### For Relaxation ###
        if len(self.phi_max.size()) > 1:
            phi_max_old = torch.reshape(self.phi_max, (-1, )).clone().detach() ### If you don't detach it everything will become slower
            phi_max_old = phi_max_old/torch.linalg.norm(phi_max_old)
        else:
            phi_max_old = self.phi_max.clone().detach()
            phi_max_old = phi_max_old / torch.linalg.norm(phi_max_old)

        ### For Relaxation ###
        phi_max = torch.reshape(self.phi_max, (-1, 1))

        ### Calculation of Matrix C ###
        C = self.calcC(self.Nx_samp_phiOpt)

        ### Solving Eigenvector Problem ###
        res = powerIteration(C, self.powIterTol)
        phi_max = res[0]
        phi_max = torch.squeeze(phi_max, 1)

        ### Relaxation ###
        if self.eigRelax is not None:
            phi_max_return = phi_max_old + self.eigRelax*(phi_max - phi_max_old)
            #phi_max_return = torch.add(phi_max_old, torch.mul(torch.add(phi_max, torch.mul(phi_max_old,-1)), eigRelax))
            phi_max_return = phi_max_return / torch.linalg.norm(phi_max_return)
        else:
            phi_max_return = phi_max

        ### Performing Tests ###
        if self.testEigOpt[1] == True:
            self.phiEigOptTest(phi_max_old.clone().detach(), phi_max.clone().detach(),
                               phi_max_return.clone().detach(), C.clone().detach())

        self.phi_max = phi_max_return
        return phi_max_return


