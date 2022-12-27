### Ploting y(x) vs the true solution exp(-x)
import numpy as np
import math
import matplotlib.pyplot as plt
from textwrap import wrap
import os

import torch

"""
def plotApproxVsTrueSol(psi, poly_pow, nele, label_id):
    fig = plt.figure()

    fig.set_figheight(10)
    fig.set_figwidth(15)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    xp = np.linspace(-3, 3, 101)
    xvec = np.zeros((poly_pow+1, 1))
    for j in range(0, nele):
        ax = fig.add_subplot(nele // math.ceil(np.sqrt(nele)) + 1, math.ceil(np.sqrt(nele)), j + 1)
        ax.plot(xp, np.exp(-xp))
        yp = np.zeros((101, 1))
        for i in range(0, 101):
            for k in range(0, poly_pow+1):
                xvec[k, 0] = xp[i]**k
            psi_node = np.reshape(psi[j, :], (1, -1))
            yp[i, 0] = np.matmul(psi_node, xvec)
        ax.plot(xp, yp)
        plt.grid(True)
        plt.title("Solution for node " + str(j+1))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(["True Solution", "Test Solution"])
    plot_title = "Plot of Approximate Vs True Solution \n" + "\n".join(wrap(label_id))
    fig.suptitle(plot_title, fontsize=16)
    plt.savefig("./figs/"+label_id, dpi=300, bbox_inches='tight')
    fig.tight_layout()
    plt.show()
"""

class plotApproxVsTrueSol:
    def __init__(self, psi, poly_pow, nele, sigma_px, label_id):
        self.psi = psi
        self.poly_pow = poly_pow
        self.nele = nele
        self.label_id = label_id
        nelsq = np.sqrt(self.nele)
        self.fig, self.ax = plt.subplots(self.nele // math.ceil(nelsq) + 1, math.ceil(nelsq))
        self.fig.set_figheight(10)
        self.fig.set_figwidth(15)
        self.fig.subplots_adjust(hspace=0.4, wspace=0.3)
        #self.xp = np.linspace(-1, 1, 101)
        self.xp = np.linspace(-3*sigma_px, 3*sigma_px, 101)
        counter =0
        self.leg_tuple = ("True Solution",)
        for j in range(0, self.nele // math.ceil(nelsq) + 1):
            for k in range(0, math.ceil(nelsq)):
                self.ax[j, k].plot(self.xp, 0.2*(counter+1)*np.exp(-self.xp), linewidth = 5)
                self.ax[j, k].grid(True)
                self.ax[j, k].set_title("Solution for node " + str(counter+1))
                self.ax[j, k].set_xlabel("x")
                self.ax[j, k].set_ylabel("y")
                self.ax[j, k].set_xlim([min(self.xp), max(self.xp)])
                self.ax[j, k].set_ylim([min(0.2*(counter+1)*np.exp(-self.xp)), max(0.2*(counter+1)*np.exp(-self.xp))])
                #self.ax[j, k].set_ylim([0, 1])
                counter = counter + 1
                if counter == self.nele:
                    break

    def add_curve(self, psi, iterat):
        nelsq = np.sqrt(self.nele)
        j = 0
        self.leg_tuple = self.leg_tuple + ("Approx. Solution, iter = " + str(iterat),)
        for jj in range(0, self.nele // math.ceil(nelsq) + 1):
            for kk in range(0, math.ceil(nelsq)):
                yp = np.zeros((101, 1))
                xvec = np.zeros((self.poly_pow + 1, 1))
                for i in range(0, 101):
                    for k in range(0, self.poly_pow + 1):
                        xvec[k, 0] = self.xp[i] ** k
                    psi_node = np.reshape(psi[j, :], (1, -1))
                    yp[i, 0] = np.matmul(psi_node, xvec)
                #nelsq = np.sqrt(self.nele)
                #self.ax = self.fig.add_subplot(self.nele // math.ceil(nelsq) + 1, math.ceil(nelsq), j + 1)
                self.ax[jj, kk].plot(self.xp, yp)
                #self.ax[jj, kk].legend(("Test Solution"+str(j),))
                self.ax[jj, kk].legend(self.leg_tuple)
                j = j + 1
                if j == self.nele:
                    break
            if j == self.nele:
                break

    def show(self, title_id):
        plot_title = "Plot of Approximate Vs True Solution \n" + "\n".join(wrap(self.label_id))
        self.fig.suptitle(plot_title, fontsize=16)
        self.fig.savefig("./" + self.label_id, dpi=300, bbox_inches='tight')
        #self.fig.tight_layout()
        self.fig.show()


class plotApproxVsSol:
    def __init__(self, psi, poly_pow, pde, sigma_px, label_id, iterat):
        self.iterat = iterat
        self.pde = pde
        self.psi = psi
        self.poly_pow = poly_pow
        self.nele = pde.effective_nele
        self.tot_nele = pde.nele
        self.label_id = label_id
        self.conf_inter = 2
        self.conf_inter_px = 2
        nelsq = np.sqrt(self.nele)
        self.fig, self.ax = plt.subplots(self.nele // math.ceil(nelsq) + 1, math.ceil(nelsq),num=18)
        #self.fig.set_figheight(30)
        #self.fig.set_figwidth(50)
        #self.fig.subplots_adjust(hspace=0.5, wspace=0.35)
        self.fig.set_figheight(8)
        self.fig.set_figwidth(10)
        self.fig.subplots_adjust(hspace=0.3, wspace=0.2)
        def analsol(s,x):
            y = (-s**2/2+s/2)*np.exp(-x)
            return y
        #self.xp = np.linspace(-1, 1, 101)
        self.xp = np.linspace(-self.conf_inter_px*sigma_px, self.conf_inter_px*sigma_px, 101)
        counter =0
        self.leg_tuple = ("True solution coeff.",)
        for j in range(0, self.nele // math.ceil(nelsq) + 1):
            if counter == self.nele:
                break
            for k in range(0, math.ceil(nelsq)):
                s = 1/self.tot_nele*(counter+1)
                self.ax[j, k].plot(self.xp, analsol(s,self.xp), linewidth = 5)
                self.ax[j, k].grid(True)
                self.ax[j, k].set_title("Solution for node " + str(counter+1))
                self.ax[j, k].set_xlabel("Number of Iterations")
                self.ax[j, k].set_ylabel("$\psi_{i}$ for node "+str(counter+1), )
                self.ax[j, k].set_xlim([min(self.xp), max(self.xp)])
                self.ax[j, k].set_ylim([analsol(s,self.conf_inter_px*sigma_px),
                                        analsol(s,-self.conf_inter_px*sigma_px)])
                counter = counter + 1
                if counter == self.nele:
                    break

    def add_curve(self, psi, iterat, sigma_r):
        psi = torch.squeeze(psi, 0)
        nelsq = np.sqrt(self.nele)
        j = 0
        self.leg_tuple = self.leg_tuple + ("Approx. Solution coeff., Nx_samp = " + str(sigma_r)+" Time:"
                                           +"{:.2f}".format(1/60) +" min",)
        for jj in range(0, self.nele // math.ceil(nelsq) + 1):
            if j== self.nele:
                break
            for kk in range(0, math.ceil(nelsq)):
                self.ax[jj, kk].plot(self.xp, psi[j]*np.exp(-self.xp))
                self.ax[jj, kk].legend(self.leg_tuple)
                j = j + 1
                if j == self.nele:
                    break

    def add_curve_pol(self, psi, var, iterat):
        #psi = torch.transpose(psi, 0, 1)
        psi = psi.cpu()
        var = torch.diag(var)
        var = torch.reshape(var, (-1, 1))
        var = var.detach().cpu().numpy()
        nelsq = np.sqrt(self.nele)
        j = 0
        self.leg_tuple = self.leg_tuple + ("Approx. Solution, iter = " + str(iterat),)
        for jj in range(0, self.nele // math.ceil(nelsq) + 1):
            for kk in range(0, math.ceil(nelsq)):
                yp = np.zeros((101, 1))
                xvec = np.zeros((self.poly_pow + 1, 1))
                for i in range(0, 101):
                    for k in range(0, self.poly_pow + 1):
                        xvec[k, 0] = self.xp[i] ** k
                    psi_node = np.reshape(psi[j, :], (1, -1))

                    yp[i, 0] = np.matmul(psi_node, xvec)
                varr = var[j, :].squeeze(0)
                # nelsq = np.sqrt(self.nele)
                # self.ax = self.fig.add_subplot(self.nele // math.ceil(nelsq) + 1, math.ceil(nelsq), j + 1)
                yp = yp.squeeze(1)
                #yp = yp * np.exp(-self.xp)
                sigma = np.sqrt(varr)
                self.ax[jj, kk].plot(self.xp, yp)
                self.ax[jj, kk].plot(self.xp, yp + self.conf_inter * sigma, '--k')
                self.ax[jj, kk].plot(self.xp, yp - self.conf_inter * sigma, '--k')
                # self.ax[jj, kk].legend(("Test Solution"+str(j),))
                self.ax[jj, kk].legend(self.leg_tuple)
                j = j + 1
                if j == self.nele:
                    break
            if j == self.nele:
                break

    def show(self, title_id):
        #plot_title = "Plot of Approximate Vs True Solution \n" + "\n".join(wrap(title_id))
        plot_title = "Plot of Approximate Vs True Solution for different x"
        self.fig.suptitle(plot_title, fontsize=16)
        if not os.path.exists('./results/approxVsTrueSol/'):
            os.makedirs('./results/approxVsTrueSol/')
        self.fig.savefig("./results/approxVsTrueSol/" + self.label_id, dpi=300, bbox_inches='tight')
        #self.fig.tight_layout()
        self.fig.show()