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
import os
import logging
from torch.distributions import constraints

smoke_test = ('CI' in os.environ)
from torch.distributions import constraints
from pyro.infer import Predictive
import time
from textwrap import wrap

### Importing Manual Modules ###
from utils.powerIteration import powerIteration
from model.phiGradOpt import residualCalc, phiEigOpt, phiGradOptMvnNx
from utils.plotBasic import plotSimplePsi_Phi, plotPhiVsSpace, plotSimplePsi_Phi_Pol
from model.modelClass import modelMvn, modelDelta, modelMvnDeb, modelMvnPolynomial, modelDeltaAllres
from model.modelClass import modelDeltaPolynomial, modelDeltaNn
from utils.plotApproxVsTrueSol import plotApproxVsSol, plotApproxVsTrueSol
from model.pdeForm import pdeForm
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

print(device)


### Definition and Form of the PDE ###
pde = pdeForm(nele, mean_px, sigma_px, Nx_samp, lBoundDir=lBoundDir, rBoundDir=rBoundDir,
              lBoundNeu=lBoundNeu, rBoundNeu=rBoundNeu, rhs=rhs)
A = pde.A
u = pde.u
nele = pde.effective_nele
real_nele = pde.nele
condition_number = torch.linalg.cond(A)

solll = torch.matmul(torch.linalg.inv(A), pde.systemRhs)
analsol = torch.linspace(0, 1, nele+1) **2/2
analsol = torch.linspace(0, 1, nele+1)**2/2 + torch.linspace(0, 1, nele+1)/2+1
analsol2 = torch.linspace(0, 1, real_nele+1)**2/2 + torch.linspace(0, 1, real_nele+1)/2+1
analsol2 = torch.linspace(0, 1, real_nele+1)**2/2 - torch.linspace(0, 1, real_nele+1)/2 +1
bbb = torch.matmul(A, torch.reshape(analsol[1:], (-1, 1)))
####### Initialization of Parameters #######

for iii in range(0, 1):
    tt = time.time()


    sigma_history = np.zeros((nele, 1))
    phi_max_history = np.zeros((nele, 1))
    phi_max = torch.rand((nele, 1), requires_grad=True)
    phi_max = phi_max / torch.linalg.norm(phi_max)


    progress_perc = 0
    grads = []
    gradsNorm = []

    residualcalc = residualCalc(mode=mode, Nx_samp=Nx_samp, pde=pde, poly_pow=poly_pow)
    psi = residualcalc.psi_init
    psi_history = residualcalc.psi_history_init
    residual_history = []

    samples = modelDeltaPolynomial(pde, phi_max=phi_max, poly_pow=poly_pow, allRes=True)
    model = samples.executeModel
    guide = samples.executeGuide
    plot_phispace = plotPhiVsSpace(torch.squeeze(phi_max, 1), nele, Iter_total, display_plots, row=7, col=5)
    kkk = 0
    for kk in range(0, Iter_outer):
        hist_elbo = []
        pyro.clear_param_store()

        svi = SVI(model,
                  guide,
                  optim.Adam({"lr": lr}),
                  loss=Trace_ELBO(num_particles=Nx_samp, retain_graph=True)) ### JitTrace makes it faster


        num_iters = Iter_svi if not smoke_test else 2
        samples.sample(phi_max, torch.tensor(sigma_r), torch.tensor(sigma_w))

        ### Phi Gradient Update ###

        ### Below is the currently corrent grad optimizer ###
        #phi_max = phiGradOptMvnNx(phi_max, lr_for_phi, nele, mean_px, sigma_px, Nx_samp, psi, A, u,
        #                            samples.x, samples.y, Iter_grad, residualcalc)
        phi_max = phiEigOpt(phi_max, samples.x, samples.y, power_iter_tol, residualcalc, eigRelax)
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

            t6 = time.time()

            NN = False
            if len(val) == 2:
                Sigmaa = torch.from_numpy(val[1])
                samples.psii = [psi, Sigmaa]
            elif len(val) == 1:
                Sigmaa = torch.zeros(nele)
                samples.psii = psi
            elif NN == True:
                Sigmaa = torch.zeros(nele)
                samples.psii = [psi, torch.from_numpy(val[1])]

            t7 = time.time()
            current_residual = 0
            for i in range(0, Nx_samp):
                current_residual = current_residual + samples.temp_res[i].item()
                #current_residual = current_residual + 1
            current_residual = current_residual/Nx_samp
            #current_residual =torch.mean(torch.stack(samples.temp_res)).clone().detach().cpu().numpy()
            ### Updating parameters of the model/guide ###
            t8 = time.time()
            residual_history.append(current_residual)
            t9 = time.time()
            ### Recording History & Applying Multipliers to hyperparameters
            # psi_history = np.concatenate((psi_history, np.transpose(val[0])), axis=1) ## Simple case
            psi_history = np.concatenate((psi_history, np.reshape(np.transpose(val[0]), (-1, 1))),
                                         axis=1)  ## For Polynomials
            t10 = time.time()
            # sigma_history = np.concatenate((sigma_history, np.reshape(val[1],(-1, 1))), axis=1)
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
        plot_appSol = plotApproxVsSol(psi, poly_pow, pde, sigma_px, Iter_outer, display_plots)
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

plotSimplePsi_Phi(Iter_total, nele, psi_history[0:nele,:], psi, phi_max_history, t,
                  residual_history, grads, gradsNorm, Iter_svi, display_plots) # instead of grads it has only gradsNorm

plot_phispace.show()


plot_appSol.show()



test = 1
#for i in range(0, len(grads)):
#    print("Iteration",i, " : ", grads[i])
print(psi)


