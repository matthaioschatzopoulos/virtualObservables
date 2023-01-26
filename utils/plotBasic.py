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
import time
from textwrap import wrap

def plotGradDeg(grad, deg, smoothing=10, display_plots=False):
    grad = torch.stack(grad).cpu()
    grad = grad[:,:, deg]
    plt.figure(60+deg)
    p0=plt.plot(np.linspace(0, len(grad), len(grad)), grad[:,0], '-m', label="$dF/ d \psi_{ji}$")
    p1 = plt.plot(np.linspace(0, len(grad), len(grad)), grad[:, 1:], '-m')
    norm = np.linalg.norm(grad.clone().detach().numpy(),axis =1)
    mean = np.mean(grad.clone().detach().numpy(),axis =1)
    p2=plt.plot(np.linspace(0, len(grad), len(grad)), smooth(mean,smoothing), '-c',
                label="Moving Average Mean $dF/ d \psi_{"+str(deg)+"i}$")
    p3=plt.plot(np.linspace(0, len(grad), len(grad)), smooth(norm, smoothing), '-k', label ="Moving Average Norm $dF/ d \psi_{"+str(deg)+"i}$")
    plt.grid(True)
    plt.yscale('symlog', linthresh=10**(-10))
    plt.title("Convergence of $dF/ d \psi_{"+str(deg)+"i}$")
    plt.xlabel("Number of external iterations")
    plt.ylabel("$dF/ d \psi_{"+str(deg)+"i}$")
    plt.legend(loc = 'lower right')
    plt.tight_layout()
    #plt.yticks(np.linspace(-torch.max(grad), torch.max(grad), 7))
    yticks = np.concatenate((-np.logspace(torch.log10(torch.max(grad)), -8, 5),
                             np.logspace(-8, torch.log10(torch.max(grad)), 5)), axis=None)
    #plt.yticks(yticks)
    if not os.path.exists('./results/gradsOrder/'):
        os.makedirs('./results/gradsOrder/')
    plt.savefig("./results/gradsOrder/der_"+str(deg)+".png", dpi =300)
    if display_plots:
        plt.show()
        plt.close(60+deg)



def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotSimplePsi_Phi(Iter_outer, nele, poly_pow, psi_history, psi, phi_max_history, t, residual, residuals,
                      elbo, relImp, grads,
                      gradsNorm, Iter_svi, display_plots, sigma_history):

    plt.figure(1)
    for i in range(0, poly_pow):
        #plt.plot(np.linspace(1, Iter_outer, Iter_outer+1), psi_history[i, :], '-r') # correct
        plt.plot(np.linspace(1, Iter_outer, Iter_outer+1), psi_history[i, :], '-r') # for nele=2
    plt.grid(True)
    #plt.title("Convergence \n" + "\n".join(wrap(title_id)))
    plt.title("Convergence of parameters $\psi_{0i}$")
    plt.xlabel("Number of external iterations")
    plt.ylabel("$\psi_0$ for each node i")
    plt.legend(["Convergence of $\psi_i$"+" Time:"+"{:.2f}".format((time.time()-t)/60) +" min",])
    if not os.path.exists('./results/order0Psi/'):
        os.makedirs('./results/order0Psi/')
    plt.savefig("./results/order0Psi/psi", dpi=300, bbox_inches='tight')
    #plt.savefig("./psi" + label_id, dpi=300, bbox_inches='tight')
    if display_plots:
        plt.show()

    plt.figure(2)
    for i in range(0, nele):
        plt.plot(np.linspace(1, Iter_outer, Iter_outer+1), phi_max_history[i, :], '-b')
    plt.grid(True)
    #plt.title("Parameters psi convergence  \n" + "\n".join(wrap(title_id)))
    plt.title("Convergence of parameters $\phi_{i}$")
    plt.xlabel("Number of external iterations")
    plt.ylabel("$\phi$ for each node i")
    if not os.path.exists('./results/order0Phi/'):
        os.makedirs('./results/order0Phi/')
    plt.savefig("./results/order0Phi/phi", dpi=300, bbox_inches='tight')
    #plt.savefig("./phi" + label_id, dpi=300, bbox_inches='tight')
    if display_plots:
        plt.show()



    plt.figure(3)
    plt.plot(np.linspace(1, Iter_outer, Iter_outer), np.asarray(residual), '-g')
    plt.plot(np.linspace(1, Iter_outer, Iter_outer), smooth(np.asarray(residual), 10), '-r')
    plt.plot(np.linspace(1, Iter_outer, Iter_outer), np.asarray(residuals), '-b')
    plt.plot(np.linspace(1, Iter_outer, Iter_outer), smooth(np.asarray(residuals), 10), '-c')
    plt.grid(True)
    plt.yscale('log')
    #plt.title("Residual convergence  \n" + "\n".join(wrap(title_id)))
    plt.title("Norm of the residual")
    plt.xlabel("Number of external iterations")
    plt.ylabel("$||Res||^2_2$")
    plt.legend(["Residual Norm", "Average Residual Norm"])
    if not os.path.exists('./results/residuals/'):
        os.makedirs('./results/residuals/')
    plt.savefig("./results/residuals/res", dpi=300, bbox_inches='tight')
    # plt.savefig("./phi" + label_id, dpi=300, bbox_inches='tight')
    if display_plots:
        plt.show()

    plt.figure(311)
    plt.plot(np.linspace(1, Iter_outer, Iter_outer), np.asarray(elbo), '-m')
    #plt.plot(np.linspace(1, Iter_outer, Iter_outer), smooth(np.asarray(residual), 10), '-r')
    plt.grid(True)
    plt.yscale('log')
    # plt.title("Residual convergence  \n" + "\n".join(wrap(title_id)))
    plt.title("ELBO")
    plt.xlabel("Number of iterations")
    plt.ylabel("ELBO")
    plt.legend(["ELBO"])
    if not os.path.exists('./results/elbo/'):
        os.makedirs('./results/elbo/')
    plt.savefig("./results/elbo/elbo", dpi=300, bbox_inches='tight')
    # plt.savefig("./phi" + label_id, dpi=300, bbox_inches='tight')
    if display_plots:
        plt.show()

    plt.figure(3111)
    plt.plot(np.linspace(1, Iter_outer, Iter_outer), np.asarray(relImp), '-m')
    #plt.plot(np.linspace(1, Iter_outer, Iter_outer), smooth(np.asarray(residual), 10), '-r')
    plt.grid(True)
    #plt.yscale('log')
    # plt.title("Residual convergence  \n" + "\n".join(wrap(title_id)))
    plt.title("Relative Improvement to sqRes when GradOpt is applied for phi")
    plt.xlabel("Number of iterations")
    plt.ylabel("relImp")
    plt.legend(["relImp"])
    if not os.path.exists('./results/relImp/'):
        os.makedirs('./results/relImp/')
    plt.savefig("./results/relImp/relImp", dpi=300, bbox_inches='tight')
    # plt.savefig("./phi" + label_id, dpi=300, bbox_inches='tight')
    if display_plots:
        plt.show()


    plt.figure(4)
    gradval=np.zeros((len(grads),nele))
    gradvalNorm = np.zeros((len(grads), 1))
    for i in range(0, len(gradvalNorm)):
        gradvalNorm[i,0]=gradsNorm[i].cpu()
    """
    for j in range(0, nele):
        for i in range(0, len(grads)):
            gradval[i,j]=(grads[i][0,j])
    
    for i in range(0, nele):
        plt.plot(np.linspace(1, Iter_outer, Iter_outer*Iter_svi), abs(gradval[:,i]), '-m')
    """
    plt.plot(np.linspace(1, Iter_outer, Iter_outer), abs(gradvalNorm), '-k')
    #plt.plot(np.linspace(1, Iter_outer, Iter_outer), smooth(np.asarray(residual), 100), '-r')
    plt.grid(True)
    plt.yscale('log')
    #plt.title("Gradients convergence  \n" + "\n".join(wrap(title_id)))
    plt.title("Norm of the derivatives $\partial log(L)/ \partial \psi$")
    plt.xlabel("Number of external iterations")
    plt.ylabel("$\partial log(L)/ \partial \psi$")
    if not os.path.exists('./results/grads/'):
        os.makedirs('./results/grads/')
    plt.savefig("./results/grads/grad", dpi=300, bbox_inches='tight')
    # plt.savefig("./phi" + label_id, dpi=300, bbox_inches='tight')
    if display_plots:
        plt.show()
    #for i in range(0, len(grads[0][0,:])):
    #    plotGradDeg(grads, i, smoothing=1000, display_plots=display_plots)

    ### Plot of parameter profile VS Taylor parameter profile ###
    psi = psi.cpu()
    plt.plot(torch.linspace(0, 1, nele).cpu(), psi)
    s = torch.linspace(0, 1, 100).cpu()
    plt.plot(s, (-s ** 2 / 2 + s / 2), "--b")
    plt.plot(s, -(-s ** 2 / 2 + s / 2), "--r")
    plt.plot(s, 1 / 2 * (-s ** 2 / 2 + s / 2), "--g")
    plt.title("0th, 1st, 2nd order parameters $\psi$ VS the respective taylor parameters")
    plt.xlabel("Space s")
    plt.ylabel("Value of $\psi_i$")
    plt.grid()
    if not os.path.exists('./results/taylorParam/'):
        os.makedirs('./results/taylorParam/')
    plt.savefig("./results/taylorParam/taylorParam.png", dpi=300, bbox_inches='tight')
    if display_plots:
        plt.show()

    plt.figure(7)
    for i in range(0, nele):
        plt.plot(np.linspace(1, Iter_outer, Iter_outer + 1), sigma_history[i, :], '-r')
    plt.grid(True)
    # plt.title("Convergence \n" + "\n".join(wrap(title_id)))
    plt.title("Convergence of parameters $\sigma_{0i}$")
    plt.xlabel("Number of external iterations")
    plt.ylabel("$\sigma_0$ for each node i")
    plt.yscale('log')
    plt.legend(["Convergence of $\sigma_i$" + " Time:" + "{:.2f}".format((time.time() - t) / 60) + " min", ])
    if not os.path.exists('./results/conv_sigma/'):
        os.makedirs('./results/conv_sigma/')
    plt.savefig("./results/conv_sigma/sigma", dpi=300, bbox_inches='tight')
    # plt.savefig("./psi" + label_id, dpi=300, bbox_inches='tight')
    if display_plots:
        plt.show()
def plotSimplePsi_Phi_Pol(Iter_outer, nele, psi_history, label_id, phi_max_history, t, residual,grads,
                      gradsNorm, Iter_svi):

    plt.figure(1)
    for i in range(0, nele):
        plt.plot(np.linspace(1, Iter_outer, Iter_outer + 1), psi_history[i, :], '-r')
    plt.grid(True)
    plt.title("Parameters $\psi_{0i}$ convergence")
    plt.xlabel("Number of external iterations")
    plt.ylabel("Value of Psi for each node in location [0.2, 0.4, 0.6, 0.8, 1.0]")
    plt.legend(["Convergence of $\psi_i$"+" Time:"+"{:.2f}".format((time.time()-t)/60) +" min",])
    plt.savefig("./figs_psi/psi", dpi=300, bbox_inches='tight')
    #plt.savefig("./psi" + label_id, dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(2)
    for i in range(0, nele):
        plt.plot(np.linspace(1, Iter_outer, Iter_outer + 1), phi_max_history[i, :], '-b')
    plt.grid(True)
    plt.title("Parameters $\phi_{0i} convergence")
    plt.xlabel("Number of external iterations")
    plt.ylabel("Value of Psi for each node in location [0.2, 0.4, 0.6, 0.8, 1.0]")
    plt.savefig("./figs_phi_max/phi", dpi=300, bbox_inches='tight')
    #plt.savefig("./phi" + label_id, dpi=300, bbox_inches='tight')
    plt.show()

    def smooth(y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    plt.figure(3)
    plt.plot(np.linspace(1, Iter_outer, Iter_outer), np.asarray(residual), '-g')
    plt.plot(np.linspace(1, Iter_outer, Iter_outer), smooth(np.asarray(residual), 100), '-r')
    plt.grid(True)
    plt.yscale('log')
    plt.title("Residuals convergence")
    plt.xlabel("Number of external iterations")
    plt.ylabel("$<r^2_{w \phi max}>_q$")
    plt.legend(["Residual", "Average Residual"])
    plt.savefig("./res/res", dpi=300, bbox_inches='tight')
    # plt.savefig("./phi" + label_id, dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(4)
    gradval=np.zeros((len(grads),nele))
    gradvalNorm = np.zeros((len(grads), 1))
#    for i in range(0, len(gradvalNorm)):
#        gradvalNorm[i,0]=gradsNorm[i]
    for j in range(0, nele):
        for i in range(0, len(grads)):
            gradval[i,j]=(grads[i][j])
    #for i in range(0, nele):
    #    plt.plot(np.linspace(1, Iter_outer, Iter_outer*Iter_svi), abs(gradval[:,i]), '-m')
    plt.plot(np.linspace(1, Iter_outer, Iter_outer * Iter_svi), abs(gradval[:, 0]), '-r')
    plt.plot(np.linspace(1, Iter_outer, Iter_outer * Iter_svi), abs(gradvalNorm), '-k')
    #plt.plot(np.linspace(1, Iter_outer, Iter_outer), smooth(np.asarray(residual), 100), '-r')
    plt.grid(True)
    plt.yscale('log')
    plt.title("Gradients convergence")
    plt.xlabel("Number of external iterations")
    plt.ylabel("$\partial log(L)/ \partial \psi$")
    plt.savefig("./res/grads", dpi=300, bbox_inches='tight')
    # plt.savefig("./phi" + label_id, dpi=300, bbox_inches='tight')
    plt.show()



class plotPhiVsSpace:
    def __init__(self, phi, nele, Iter_outer, display_plots, row, col):
        self.display_plots = display_plots
        self.phi = torch.cat((torch.zeros(1), phi))
        self.phi = self.phi.detach().cpu().numpy()
        self.nele = nele
        self.s = torch.linspace(0,1,nele+1)
        self.s = self.s.cpu()
        self.fig = plt.figure(2)
        self.grid_num = int(col)  # num of columns
        self.rows = int(row)
        self.plot_num_tot = self.grid_num * self.rows
        self.mod_iter = np.ceil(Iter_outer / (self.plot_num_tot-1))
        self.fig, self.ax = plt.subplots(self.plot_num_tot // self.grid_num, self.grid_num,num=14)
        #self.ax = self.fig.add_subplot(3,3,1)
        self.fig.set_figheight(15)
        self.fig.set_figwidth(15)
        self.fig.subplots_adjust(hspace=0.4, wspace=0.3)
        self.counter = 0
        self.leg_tuple = ("True Solution",)
        self.ax[0,0].plot(self.s, self.phi)
        self.ax[0,0].grid(True)
        self.ax[0,0].set_title("Eigenvector $phi$ during Initialization")
        self.ax[0,0].set_xlabel("s")
        self.ax[0,0].set_ylabel("$\phi$")
        self.ax[0,0].set_ylim([-1., 1.])
        self.counter = self.counter+1
        #self.ax.set_xlim()
        #self.ax.set_ylim([min(0.2*(counter+1)*np.exp(-self.xp)), max(0.2*(counter+1)*np.exp(-self.xp))])
        #self.ax[j, k].set_ylim([0, 1])

    def add_curve(self, phi, iterat):
        if iterat % self.mod_iter == 0:
            phi = torch.cat((torch.zeros(1), phi)).detach().cpu().numpy()
            i = self.counter//self.grid_num
            j = self.counter % self.grid_num
            if j>=0:
                self.ax[i,j].plot(self.s, phi)
            else:
                self.ax[i,j].plot(self.s, phi)
            self.ax[i, j].set_title("Eigenvector $phi$ for iter = " + str(iterat))
            self.ax[i, j].set_xlabel("s")
            self.ax[i, j].set_ylabel("$\phi$")
            self.ax[i, j].grid(True)
            self.ax[i, j].set_ylim([-1., 1.])
            self.counter += 1

    def show(self):
        #plot_title = "Eigenvector $\phi$ in each node ("+ str(self.nele)+" in total) \n"\
        #             + "\n".join(wrap(self.label_id))
        plot_title = "Eigenvector $\phi$ in each node ("+ str(self.nele)+" in total) \n"
        self.fig.suptitle(plot_title, fontsize=16)
        if not os.path.exists('./results/phiVsNodes/'):
            os.makedirs('./results/phiVsNodes/')
        self.fig.savefig("./results/phiVsNodes/psiVsNodes", dpi=300, bbox_inches='tight')
        #self.fig.tight_layout()
        if self.display_plots:
            self.fig.show()