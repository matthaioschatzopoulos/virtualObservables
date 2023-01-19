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
#from numba import jit
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
import fenics as df
import os
import logging
from torch.distributions import constraints

smoke_test = ('CI' in os.environ)
from torch.distributions import constraints
from pyro.infer import Predictive
import time
import os
from textwrap import wrap

### Importing Manual Modules ###
from utils.powerIteration import powerIteration
from model.phiGradOpt import phiOptimizer
from utils.plotBasic import plotSimplePsi_Phi, plotPhiVsSpace, plotSimplePsi_Phi_Pol
from model.modelClass import modelMvn, modelDelta, modelMvnDeb, modelMvnPolynomial, modelDeltaAllres
from model.modelClass import modelDeltaPolynomial, modelDeltaNn
from utils.plotApproxVsTrueSol import plotApproxVsSol, plotApproxVsTrueSol
from model.pdeForm import pdeForm, pdeFenics
from input import *

################ Testing weighting residuals ################
#pyro.set_rng_seed(1)
#torch.set_printoptions(precision=5)
np.set_printoptions(formatter={'float': '{: 0.14f}'.format})
t = time.time()

### Device Selection ###


if device == 'cpu':
    use_cuda=False
else:
    use_cuda=True
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda:0" if use_cuda else "cpu")
### Device Selection ###

### Storing specific input in the ./results/
print(device)

os.system('cp ./input.py ./results/inputParams.txt')

### Definition and Form of the PDE ###
pde = pdeForm(nele, mean_px, sigma_px, Nx_samp, lBoundDir=lBoundDir, rBoundDir=rBoundDir,
              lBoundNeu=lBoundNeu, rBoundNeu=rBoundNeu, rhs=rhs)
A = pde.A
u = pde.u
nele = pde.effective_nele
real_nele = pde.nele
condition_number = torch.linalg.cond(A)

solll = torch.matmul(torch.linalg.inv(A), pde.systemRhs)
print(A)
print(pde.systemRhs)
analsol = torch.linspace(0, 1, nele+1) **2/2
analsol = torch.linspace(0, 1, nele+1)**2/2 + torch.linspace(0, 1, nele+1)/2+1
analsol2 = torch.linspace(0, 1, real_nele+1)**2/2 + torch.linspace(0, 1, real_nele+1)/2+1
analsol2 = torch.linspace(0, 1, real_nele+1)**2/2 - torch.linspace(0, 1, real_nele+1)/2 +1
bbb = torch.matmul(A, torch.reshape(analsol[1:], (-1, 1)))
####### Initialization of Parameters #######

for iii in range(0, 1):


    tt = time.time()


    sigma_history = np.zeros((nele, 1))
    psi_history = torch.reshape(torch.zeros(nele, poly_pow + 1), (-1,1))
    phi_max_history = np.zeros((nele, 1))
    phi_max = torch.rand((nele, 1), requires_grad=True)
    phi_max = phi_max / torch.linalg.norm(phi_max)


    progress_perc = 0
    grads = []
    gradsNorm = []

    phiOptim = phiOptimizer(pde, poly_pow=poly_pow, eigRelax=eigRelax, powIterTol=powerIterTol,
                            Nx_samp_phi=Nx_samp_phiOpt, runTests=runTests)
    residual_history = []


    samples = modelMvnPolynomial(pde, phi_max=phi_max, poly_pow=poly_pow, allRes=allRes)
    phiOptim.model = samples
    model = samples.executeModel
    guide = samples.executeGuide
    plot_phispace = plotPhiVsSpace(torch.squeeze(phi_max, 1), nele, Iter_total, display_plots, row=7, col=5)
    kkk = 0



    for kk in range(0, Iter_outer):
        """
        if progress_perc > phase1:
            Nx_samp = 10
            pde.Nx_samp = Nx_samp
            samples.Nx_samp_phi = Nx_samp
            samples.x = torch.zeros(samples.Nx_samp_phi, 1)
            samples.y = torch.zeros(samples.Nx_samp_phi, samples.nele)
            phiOptim.Nx_samp = pde.Nx_samp
            phiOptim.Nx_samp_phi = phiOptim.Nx_samp
            phiOptim.x = torch.zeros(phiOptim.Nx_samp_phi, 1)
            phiOptim.y = torch.zeros(phiOptim.Nx_samp_phi, phiOptim.nele)
            phiOptim.Nx_samp = phiOptim.Nx_samp
            phiOptim.temp_res = torch.rand(phiOptim.Nx_samp).tolist()
            phiOptim.full_res_temp = torch.rand(phiOptim.Nx_samp).tolist()

            if progress_perc > phase2:
                Nx_samp = 50
                pde.Nx_samp = Nx_samp
                samples.Nx_samp_phi = Nx_samp
                samples.x = torch.zeros(samples.Nx_samp_phi, 1)
                samples.y = torch.zeros(samples.Nx_samp_phi, samples.nele)
                phiOptim.Nx_samp = pde.Nx_samp
                phiOptim.Nx_samp_phi = phiOptim.Nx_samp
                phiOptim.x = torch.zeros(phiOptim.Nx_samp_phi, 1)
                phiOptim.y = torch.zeros(phiOptim.Nx_samp_phi, phiOptim.nele)
                phiOptim.Nx_samp = phiOptim.Nx_samp
                phiOptim.temp_res = torch.rand(phiOptim.Nx_samp).tolist()
                phiOptim.full_res_temp = torch.rand(phiOptim.Nx_samp).tolist()
        """
        hist_elbo = []
        pyro.clear_param_store()

        svi = SVI(model,
                  guide,
                  optim.Adam({"lr": lr}),
                  loss=Trace_ELBO(num_particles=Nx_samp, retain_graph=True)) ### JitTrace makes it faster


        num_iters = Iter_svi if not smoke_test else 2


        ### Phi Gradient Update ###

        ### Below is the currently corrent grad optimizer ###
        #phi_max = phiGradOptMvnNx(phi_max, lr_for_phi, nele, mean_px, sigma_px, Nx_samp, psi, A, u,
        #                            samples.x, samples.y, Iter_grad, residualcalc)

        #samples.sample(phi_max, torch.tensor(sigma_r), torch.tensor(sigma_w))
        """
        xxxx, yyyy = samples.sampleResExp(100)
        bbbb = 0.0
        for jj in range(0, xxxx.size(dim=0)):
            bbbbb = pde.calcResKernel(xxxx[jj], yyyy[jj, :])
            bbbb = bbbb + bbbbb/ xxxx.size(dim=0)
        current_residual = torch.linalg.norm(bbbb)
        """

        phi_max = phiOptim.phiEigOpt()
        #phi_max = phiOptim.phiGradOptMvnNx()
        #phi_max = torch.ones(nele)
        svi_time = 0
        rest_time = 0
        tsum = 0




        for i in range(num_iters):
            samples.removeSamples()
            kkk = kkk+1

            ### SVI step ###
            temp = time.time()
            elbo, current_grad = svi.stepGrad(phi_max,
                                              torch.tensor(sigma_r), torch.tensor(sigma_w))
            svi_time = svi_time + time.time() - temp
            temp = time.time()
            t1 = time.time()
            hist_elbo.append(elbo)
            t2 = time.time()
            val = []

            t3 = time.time()
            for name, value in pyro.get_param_store().items():
                #print(value.grad)
                val.append(value.clone().detach().cpu().numpy())
                assert value.requires_grad == True, "Not leaf tensor"
                #print(value.grad)
                #grads.append(value.clone.grad)
            t4 = time.time()
            for jjjj in range(0, 1):
                if len(val) == 2:
                    grad = current_grad[0].clone().detach()
                    grads.append(grad)
                    gradsNorm.append(torch.linalg.norm(grad))
                elif len(val) == 1:
                    grad = current_grad[jjjj].clone().detach()
                    grads.append(grad)
                    gradsNorm.append(torch.linalg.norm(grad))
            t5 = time.time()

            psi = torch.from_numpy(val[0]).to(device=device)
            phiOptim.full_res_temp = samples.full_temp_res
            t6 = time.time()

            NN = False
            if len(val) == 2:
                Sigmaa = torch.from_numpy(val[1])
                samples.psii = [psi, Sigmaa]
                sigma_history = np.concatenate((sigma_history, np.reshape(val[1], (-1, 1))), axis=1)

            elif len(val) == 1:
                Sigmaa = torch.zeros(nele)
                samples.psii = psi
            elif NN == True:
                Sigmaa = torch.zeros(nele)
                samples.psii = [psi, torch.from_numpy(val[1])]

            t7 = time.time()


            current_residual = 0
            for jk in range(0, Nx_samp):
                current_residual = current_residual + samples.temp_res[jk].item()
            current_residual = current_residual/Nx_samp

            t8 = time.time()
            residual_history.append(current_residual)
            t9 = time.time()


            #current_residual =torch.mean(torch.stack(samples.temp_res)).clone().detach().cpu().numpy()
            ### Updating parameters of the model/guide ###

            ### Recording History & Applying Multipliers to hyperparameters
            # psi_history = np.concatenate((psi_history, np.transpose(val[0])), axis=1) ## Simple case
            psi_history = np.concatenate((psi_history, np.reshape(np.transpose(val[0]), (-1, 1))),
                                         axis=1)  ## For Polynomials
            t10 = time.time()

            phi_max_history = np.concatenate((phi_max_history, torch.reshape(phi_max, (-1, 1)).detach().cpu().numpy()),
                                             axis=1)
            sigma_r = sigma_r * sigma_r_mult
            t11 = time.time()
            rest_time = rest_time + time.time() - temp
            tsum += np.array((t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6, t8 - t7, t9 - t8,
                             t10 - t9, t11 - t10))
            if (kkk + 1) % ((Iter_total) * 0.01) == 0:

                #plot_diffsigmar.add_curve(psi_history, kk, sigma_r, time.time()-tt)
                #print(residualcalc.residual.item())
                #residual_history.append(residualcalc.residual.item())

                ### Printing Progress ###

                progress_perc = progress_perc + 1


                #print("Iteration: ", kk,"/",Iter_outer, " ELBO", elbo, "sigma_r", sigma_r)
                print("Gradient:",current_grad)
                print("Parameters psi:",psi)
                print("Other stuff time analysis: ", tsum)
                print("Model stuff time analysis: ", samples.model_time)
                if len(val) == 2:
                    print("Standard deviation: ", Sigmaa)
                print("Progress: ", progress_perc, "/", 100, " ELBO", elbo, "Residual",
                      min(residual_history[-int(Iter_outer / 100):]), "sigma_r", sigma_r)
                plot_phispace.add_curve(phi_max, kkk)


    if iii == 0:
        plot_appSol = plotApproxVsSol(psi, poly_pow, pde, sigma_px, Iter_outer, display_plots, samples)
    plot_appSol.add_curve_pol(psi, torch.diag(Sigmaa), kkk)
    tt = time.time()
print("Program Finished.")
elapsed = time.time() - t
print("Total time: ", elapsed)
print("SVI time: ", svi_time)
print("Other stuff time: ", rest_time)
print("Model time: ", samples.model_time)
print("Guide time: ", samples.guide_time)
print("Sample time: ", samples.sample_time)

#plotProbabModel(model, guide, phi_max, A, u, sigma_r, sigma_w, psi, nele)

#plotModelDists(model, guide, phi_max, A, u, sigma_r, sigma_w, psi, nele)

#plotSimplePsi_Phi(Iter_total, nele, psi_history[0:nele,:], psi, phi_max_history, t,
#                  residual_history, grads, gradsNorm, Iter_svi, display_plots, sigma_history) # instead of grads it has only gradsNorm
plotSimplePsi_Phi(Iter_total, nele, poly_pow+1, psi_history, psi, phi_max_history, t,
                  residual_history, grads, gradsNorm, Iter_svi, display_plots, sigma_history) # instead
plot_phispace.show()


plot_appSol.show()



test = 1
#for i in range(0, len(grads)):
#    print("Iteration",i, " : ", grads[i])
print(psi)
if phiOptim.testEigOpt[1] == True:
    print("phiOptEigTest MaximizationTest: " + str(phiOptim.testEigOpt[0]))
    print("phiOptEigTest MCtest: " + str(phiOptim.testEigOpt[2]))

