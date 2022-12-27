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
torch.set_default_tensor_type(torch.DoubleTensor)

def plotGradDeg(grad, deg):
    grad = torch.stack(grad)
    grad = grad[:,:, deg]
    plt.figure(60+deg)
    p0=plt.plot(np.linspace(0, len(grad), len(grad)), grad[:,0], '-m', label="$dF/ d \psi_{ji}$")
    p1 = plt.plot(np.linspace(0, len(grad), len(grad)), grad[:, 1:], '-m')
    norm = np.linalg.norm(grad.clone().detach().numpy(),axis =1)
    mean = np.mean(grad.clone().detach().numpy(),axis =1)
    p2=plt.plot(np.linspace(0, len(grad), len(grad)), smooth(mean,1000), '-c',
                label="Moving Average Mean $dF/ d \psi_{"+str(deg)+"i}$")
    p3=plt.plot(np.linspace(0, len(grad), len(grad)), smooth(norm, 1000), '-k', label ="Moving Average Norm $dF/ d \psi_{"+str(deg)+"i}$")
    plt.grid(True)
    plt.yscale('symlog', linthresh=10**(-6))
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
    plt.show()
    plt.close(60+deg)
def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotSimplePsi_Phi(Iter_outer, nele, psi_history, title_id, label_id, phi_max_history, t, residual,grads,
                      gradsNorm, Iter_svi):

    plt.figure(1)
    for i in range(0, nele):
        plt.plot(np.linspace(1, Iter_outer, Iter_outer + 1), psi_history[i, :], '-r')
    plt.grid(True)
    #plt.title("Convergence \n" + "\n".join(wrap(title_id)))
    plt.title("Convergence of parameters $\psi_{0i}$")
    plt.xlabel("Number of external iterations")
    plt.ylabel("$\psi_0$ for each node i")
    plt.legend(["Convergence of $\psi_i$"+" Time:"+"{:.2f}".format((time.time()-t)/60) +" min",])
    if not os.path.exists('./results/order0Psi/'):
        os.makedirs('./results/order0Psi/')
    plt.savefig("./results/order0Psi/psi" + label_id, dpi=300, bbox_inches='tight')
    #plt.savefig("./psi" + label_id, dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(2)
    for i in range(0, nele):
        plt.plot(np.linspace(1, Iter_outer, Iter_outer + 1), phi_max_history[i, :], '-b')
    plt.grid(True)
    #plt.title("Parameters psi convergence  \n" + "\n".join(wrap(title_id)))
    plt.title("Convergence of parameters $\phi_{i}$")
    plt.xlabel("Number of external iterations")
    plt.ylabel("$\phi$ for each node i")
    if not os.path.exists('./results/order0Phi/'):
        os.makedirs('./results/order0Phi/')
    plt.savefig("./results/order0Phi/phi" + label_id, dpi=300, bbox_inches='tight')
    #plt.savefig("./phi" + label_id, dpi=300, bbox_inches='tight')
    plt.show()



    plt.figure(3)
    plt.plot(np.linspace(1, Iter_outer, Iter_outer), np.asarray(residual), '-g')
    plt.plot(np.linspace(1, Iter_outer, Iter_outer), smooth(np.asarray(residual), 100), '-r')
    plt.grid(True)
    plt.yscale('log')
    #plt.title("Residual convergence  \n" + "\n".join(wrap(title_id)))
    plt.title("Norm of the residual")
    plt.xlabel("Number of external iterations")
    plt.ylabel("$||Res||^2_2$")
    plt.legend(["Residual Norm", "Average Residual Norm"])
    if not os.path.exists('./results/residuals/'):
        os.makedirs('./results/residuals/')
    plt.savefig("./results/residuals/res" + label_id, dpi=300, bbox_inches='tight')
    # plt.savefig("./phi" + label_id, dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(4)
    gradval=np.zeros((len(grads),nele))
    gradvalNorm = np.zeros((len(grads), 1))
    for i in range(0, len(gradvalNorm)):
        gradvalNorm[i,0]=gradsNorm[i]
    """
    for j in range(0, nele):
        for i in range(0, len(grads)):
            gradval[i,j]=(grads[i][0,j])
    
    for i in range(0, nele):
        plt.plot(np.linspace(1, Iter_outer, Iter_outer*Iter_svi), abs(gradval[:,i]), '-m')
    """
    plt.plot(np.linspace(1, Iter_outer, Iter_outer * Iter_svi), abs(gradvalNorm), '-k')
    #plt.plot(np.linspace(1, Iter_outer, Iter_outer), smooth(np.asarray(residual), 100), '-r')
    plt.grid(True)
    plt.yscale('log')
    #plt.title("Gradients convergence  \n" + "\n".join(wrap(title_id)))
    plt.title("Norm of the derivatives $\partial log(L)/ \partial \psi$")
    plt.xlabel("Number of external iterations")
    plt.ylabel("$\partial log(L)/ \partial \psi$")
    if not os.path.exists('./results/grads/'):
        os.makedirs('./results/grads/')
    plt.savefig("./results/grads/grad" + label_id, dpi=300, bbox_inches='tight')
    # plt.savefig("./phi" + label_id, dpi=300, bbox_inches='tight')
    plt.show()
    for i in range(0, len(grads[0][0,:])):
        plotGradDeg(grads, i)
def plotSimplePsi_Phi_Pol(Iter_outer, nele, psi_history, title_id, label_id, phi_max_history, t, residual,grads,
                      gradsNorm, Iter_svi):

    plt.figure(1)
    for i in range(0, nele):
        plt.plot(np.linspace(1, Iter_outer, Iter_outer + 1), psi_history[i, :], '-r')
    plt.grid(True)
    plt.title("Parameters psi convergence \n" + "\n".join(wrap(title_id)))
    plt.xlabel("Number of external iterations")
    plt.ylabel("Value of Psi for each node in location [0.2, 0.4, 0.6, 0.8, 1.0]")
    plt.legend(["Convergence of $\psi_i$"+" Time:"+"{:.2f}".format((time.time()-t)/60) +" min",])
    plt.savefig("./figs_psi/psi" + label_id, dpi=300, bbox_inches='tight')
    #plt.savefig("./psi" + label_id, dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(2)
    for i in range(0, nele):
        plt.plot(np.linspace(1, Iter_outer, Iter_outer + 1), phi_max_history[i, :], '-b')
    plt.grid(True)
    plt.title("Parameters psi convergence  \n" + "\n".join(wrap(title_id)))
    plt.xlabel("Number of external iterations")
    plt.ylabel("Value of Psi for each node in location [0.2, 0.4, 0.6, 0.8, 1.0]")
    plt.savefig("./figs_phi_max/phi" + label_id, dpi=300, bbox_inches='tight')
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
    plt.title("Residual convergence  \n" + "\n".join(wrap(title_id)))
    plt.xlabel("Number of external iterations")
    plt.ylabel("$<r^2_{w \phi max}>_q$")
    plt.legend(["Residual", "Average Residual"])
    plt.savefig("./res/res" + label_id, dpi=300, bbox_inches='tight')
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
    plt.title("Gradients convergence  \n" + "\n".join(wrap(title_id)))
    plt.xlabel("Number of external iterations")
    plt.ylabel("$\partial log(L)/ \partial \psi$")
    plt.savefig("./res/grads" + label_id, dpi=300, bbox_inches='tight')
    # plt.savefig("./phi" + label_id, dpi=300, bbox_inches='tight')
    plt.show()



class plotPhiVsSpace:
    def __init__(self, phi, nele, label_id, Iter_outer, row, col):
        self.phi = torch.cat((torch.zeros(1), phi))
        self.phi = self.phi.detach().numpy()
        self.nele = nele
        self.s = torch.linspace(0,1,nele+1)
        self.label_id = label_id
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
            phi = torch.cat((torch.zeros(1), phi)).detach().numpy()
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
        plot_title = "Eigenvector $\phi$ in each node ("+ str(self.nele)+" in total) \n"\
                     + "\n".join(wrap(self.label_id))
        self.fig.suptitle(plot_title, fontsize=16)
        if not os.path.exists('./results/phiVsNodes/'):
            os.makedirs('./results/phiVsNodes/')
        self.fig.savefig("./results/phiVsNodes/psiVsNodes" + self.label_id, dpi=300, bbox_inches='tight')
        #self.fig.tight_layout()
        self.fig.show()