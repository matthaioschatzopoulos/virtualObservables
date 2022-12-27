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

### Importing Manual Modules ###
from utils.powerIteration import powerIteration
from model.phiGradOpt import residualCalc, phiEigOpt, phiGradOptMvnNx
from utils.plotBasic import plotSimplePsi_Phi, plotPhiVsSpace, plotSimplePsi_Phi_Pol
from model.modelClass import modelMvn, modelDelta, modelMvnDeb, modelMvnPolynomial, modelDeltaAllres
from model.modelClass import modelDeltaPolynomial, modelDeltaNn
from utils.plotApproxVsTrueSol import plotApproxVsSol, plotApproxVsTrueSol
from model.pdeForm import pdeForm

################ Testing weighting residuals ################
torch.set_default_tensor_type(torch.DoubleTensor)
#torch.set_printoptions(precision=5)
np.set_printoptions(formatter={'float': '{: 0.14f}'.format})
t = time.time()
nele = 5
Nx_samp = 1  # 5 is good
mean_px = 0
sigma_px = 0.1

### Definition and Form of the PDE ###
pde = pdeForm(nele, mean_px, sigma_px, Nx_samp, lBoundDir=0, rBoundDir=0, lBoundNeu=None, rBoundNeu=None, rhs=-1)
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
sigma_r_list = [0.001, 0.005, 0.01, 0.1, 1, 10, 100, 1000]
sigma_r_list = [1, 5, 10, 20, 30, 50]
sigma_r_list = [0.05 * 0.001, 0.1 * 0.001, 0.5 * 0.001, 1 * 0.001, 5 * 0.001]
sigma_r_list = [0.001, 0.005, 0.01, 0.1, 1, 10, 100, 1000]
sigma_r_list = [1, 5, 10, 20]
for iii in range(0, 1):
    tt = time.time()


    #sig_mat = torch.rand(nele, nele)
    #sig_mat = torch.mm(sig_mat, sig_mat.t()) + torch.eye(nele, nele)
    # phi_max = np.random.rand(nele, 1)*0.1

    Iter_svi = 10
    Iter_grad = 1
    sigma_r = 10**(0) ### Very sensitivy to changes (For the multivariate is much different)
    sigma_r_mult = 1.00
    sigma_w = 10000000
    Iter_outer = 100
    poly_pow = 2
    sigma_r_f = round(sigma_r * sigma_r_mult ** Iter_outer, 2)
    mode="Polynomial" ### "TrueSol" or "Polynomial"
    sigma_history = np.zeros((nele, 1))
    phi_max_history = np.zeros((nele, 1))
    power_iter_tol = 10 ** (-5)
    phi_max = torch.rand((nele, 1), requires_grad=True)
    phi_max = phi_max / torch.linalg.norm(phi_max)

    ### Inputs ###
    lr = 0.0004 ### In the PolynomialMultivariate case reducing the learning rate can have a positive effect
    lr_for_phi = 0.005
    eigRelax = 0.05
    progress_perc = 0
    grads = []
    gradsNorm = []

    residualcalc = residualCalc(mode=mode, Nx_samp=Nx_samp, pde=pde, poly_pow=poly_pow)
    psi = residualcalc.psi_init
    psi_history = residualcalc.psi_history_init
    residual_history = []

    label_id = "Delta_Simple_" + "Nele=" + str(nele) + "_Iter_outer=" + str(Iter_outer)
    label_id = label_id + "_sigma_r=" + str(sigma_r) + "_lr=" + str(lr).replace(".", "d") + "_mean_px=" + str(mean_px)
    label_id = label_id + "_Nx_sample=" + str(Nx_samp) + "_sigma_w=" + str(sigma_w) + "_final_sigma_r="
    label_id = label_id + str(sigma_r_f).replace(".", "d")
    label_id = "_Iter_outer=" + str(Iter_outer) + "_sigma_r=" + str(sigma_r).replace(".", "d") + "_final_sigma_r=" \
               + str(sigma_r_f).replace(".", "d")

    label_id = "_Iter_outer=" + str(Iter_outer) + "_sigma_r=" + str(sigma_r).replace(".", "d")
    title_id = "Initial sigma_r = " + str(sigma_r) + ", Final sigma_r = "
    title_id = title_id + "{:.2f}".format(sigma_r_f) + ", Multiplier = " + str(sigma_r_mult)
    label_id = "_Iter_outer=" + str(Iter_outer) + "diff_sigma_r_lr_" + "{:.0e}".format(lr) + "_itersvi_" + str(Iter_svi)
    label_id = "_Iter_outer=" + str(Iter_outer) + "_sigma_r_" + str(sigma_r) + "_lr_" + "{:.0e}".format(
        lr) + "_diff_itersvi_"
    label_id = "_Iter_outer=" + str(Iter_outer) + "_sigma_r_" + "Nx_svi_iter_" + \
               "{:d}".format(Iter_svi)
    # title_id = "For $\sigma_{r}=$ "+str(sigma_r_list)
    title_id = "For Iter_svi = " + str(sigma_r_list)
    title_id = "For learning rate = " + str(sigma_r_list)
    # if iii == 0:
    # plot1 = plotDiffSigmar(np.zeros((nele, Iter_outer+1)), 4, nele, label_id, Iter_outer)

    # x = np.exp(np.random.normal(loc=mean_px, scale=sigma_px, size=Nx_samp))
    samples = modelDeltaPolynomial(pde, phi_max=phi_max, poly_pow=poly_pow, allRes=True)
    model = samples.executeModel
    guide = samples.executeGuide
    plot_phispace = plotPhiVsSpace(torch.squeeze(phi_max, 1), nele, label_id, Iter_outer, row=7, col=5)
    for kk in range(0, Iter_outer):

        hist_elbo = []
        pyro.clear_param_store()

        svi = SVI(model,
                  guide,
                  optim.Adam({"lr": lr}),
                  loss=Trace_ELBO(num_particles=Nx_samp, retain_graph=True))


        num_iters = Iter_svi if not smoke_test else 2
        samples.sample(phi_max, torch.tensor(sigma_r), torch.tensor(sigma_w))

        ### Phi Gradient Update ###

        ### Below is the currently corrent grad optimizer ###
        #phi_max = phiGradOptMvnNx(phi_max, lr_for_phi, nele, mean_px, sigma_px, Nx_samp, psi, A, u,
        #                            samples.x, samples.y, Iter_grad, residualcalc)
        phi_max = phiEigOpt(phi_max, samples.x, samples.y, power_iter_tol, residualcalc, eigRelax)
        #phi_max = torch.ones(nele)

        for i in range(num_iters):



            ### SVI step ###
            elbo, current_grad = svi.stepGrad(phi_max,
                                              torch.tensor(sigma_r), torch.tensor(sigma_w))
            hist_elbo.append(elbo)

            val = []



            """
            ### lr_svi rate regulator/controller ###
            grad_nrate = -(gradsNorm[kk] - gradsNorm[kk - 1]) / gradsNorm[kk - 1]
            if kk > 1000 and gradsNorm[kk] > 10**(-8):
                if grad_nrate < 1.06:
                    lr = lr * 1.001
                elif grad_nrate > 1.2:
                    lr = lr * 0.999
            else:
                lr = 0.000001
            ### lr_svi rate regulator/controller ###
            """

            for name, value in pyro.get_param_store().items():
                #print(value.grad)
                val.append(value.clone().detach().numpy())
                assert value.requires_grad == True, "Not leaf tensor"
                #print(value.grad)
                #grads.append(value.clone.grad)

            for jjjj in range(0, 1):
                if len(val) == 2:
                    grads.append(current_grad[0])
                    gradsNorm.append(torch.linalg.norm(current_grad[0]))
                elif len(val) == 1:
                    grads.append(current_grad[jjjj])
                    gradsNorm.append(torch.linalg.norm(current_grad[jjjj]))

            psi = torch.from_numpy(val[0])
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
            current_residual =torch.mean(torch.stack(samples.temp_res)).clone().detach().numpy()
            ### Updating parameters of the model/guide ###

            #samples.removeSamples()
            ### SVI step ###
            #if i == 10 and i < 12:
            #    psi10 = psi
            #    print('10')
            #if i == 99:
            #    psi100 = psi
            #    print('100')
            """
            if (i + 1) % ((Iter_svi) * 0.01) == 0:
                progress_perc = progress_perc + 1
                # print("Iteration: ", kk,"/",Iter_outer, " ELBO", elbo, "sigma_r", sigma_r)
                #print("Progress: ", progress_perc, "/", 100, " ELBO", elbo, "Residual",
                #      min(residual_history[-int(Iter_svi / 100):]), "sigma_r", sigma_r)
                print("Gradient:", current_grad)
                print("Parameters psi:", samples.psii)
                print("Residuals : ", samples.temp_res)
                if len(val) == 2:
                    print("Standard deviation: ", Sigmaa)
            """

        residual_history.append(current_residual)
        ### Recording History & Applying Multipliers to hyperparameters
        #psi_history = np.concatenate((psi_history, np.transpose(val[0])), axis=1) ## Simple case
        psi_history = np.concatenate((psi_history, np.reshape(np.transpose(val[0]), (-1, 1))), axis=1) ## For Polynomials
        #sigma_history = np.concatenate((sigma_history, np.reshape(val[1],(-1, 1))), axis=1)
        phi_max_history = np.concatenate((phi_max_history, torch.reshape(phi_max, (-1, 1)).detach().numpy()), axis=1)
        sigma_r = sigma_r * sigma_r_mult
        #plot_diffsigmar.add_curve(psi_history, kk, sigma_r, time.time()-tt)
        #print(residualcalc.residual.item())
        #residual_history.append(residualcalc.residual.item())

        ### Printing Progress ###
        if (kk+1) % ((Iter_outer)*0.01) == 0:
            progress_perc = progress_perc + 1
            #print("Iteration: ", kk,"/",Iter_outer, " ELBO", elbo, "sigma_r", sigma_r)
            print("Progress: ", progress_perc, "/", 100, " ELBO", elbo,"Residual",
                  min(residual_history[-int(Iter_outer/100):]) , "sigma_r", sigma_r)
            print("Gradient:",current_grad)
            print("Parameters psi:",psi)
            if len(val) == 2:
                print("Standard deviation: ", Sigmaa)

        plot_phispace.add_curve(phi_max, kk)
    if iii == 0:
        plot_appSol = plotApproxVsSol(psi, poly_pow, pde, sigma_px, label_id, Iter_outer)
    plot_appSol.add_curve_pol(psi, torch.diag(Sigmaa), kk)
    tt = time.time()
print("end")
print("end2")

#plotProbabModel(model, guide, phi_max, A, u, sigma_r, sigma_w, psi, nele)

#plotModelDists(model, guide, phi_max, A, u, sigma_r, sigma_w, psi, nele)

plotSimplePsi_Phi(Iter_outer, nele, psi_history[0:nele,:], title_id, label_id, phi_max_history, t,
                  residual_history, grads, gradsNorm, Iter_svi) # instead of grads it has only gradsNorm

plot_phispace.show()


plot_appSol.show(title_id=title_id)


elapsed = time.time() - t
print("Total time: ", elapsed)
test = 1
#for i in range(0, len(grads)):
#    print("Iteration",i, " : ", grads[i])
print(psi)
tess = torch.squeeze(psi, 0)


### Plot of parameter profile VS Taylor parameter profile ###
plt.plot(torch.linspace(0, 1, nele), torch.squeeze(psi, 0))
s = torch.linspace(0, 1, 100)
plt.plot(s, (-s**2/2+s/2), "--b")
plt.plot(s, -(-s**2/2+s/2), "--r")
plt.plot(s, 1/2*(-s**2/2+s/2), "--g")
plt.title("0th, 1st, 2nd order parameters $\psi$ VS the respective taylor parameters")
plt.xlabel("Space s")
plt.ylabel("Value of $\psi_i$")
plt.grid()
if not os.path.exists('./results/taylorParam/'):
    os.makedirs('./results/taylorParam/')
plt.savefig("./results/taylorParam/taylorParam.png", dpi=300, bbox_inches='tight')
plt.show()
