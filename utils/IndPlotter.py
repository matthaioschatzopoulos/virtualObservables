### Ploting y(x) vs the true solution exp(-x)
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from textwrap import wrap
import os
import matplotlib.animation as animation
import torch

Strue, Xtrue = torch.meshgrid(torch.linspace(0, 1, 101), torch.linspace(-1, 1, 101), indexing='ij')
yTrueSurf = torch.zeros((Strue.size(dim=0), Xtrue.size(dim=1)))

figsurf, surf = plt.subplots(2)
figsurf.set_figheight(7)
figsurf.set_figwidth(7)
real_nele=21
S = torch.linspace(0, 1, real_nele + 1)
X = torch.linspace(-1, 1, 101)
S, X = torch.meshgrid(S, X, indexing='ij')
#text = self.surf.text(0.75, 0.85, 0.85, 'test')
# psi = torch.transpose(psi, 0, 1)


def polynomial(x):
    y = torch.zeros(4 + 1, 1)
    for k in range(0, 4 + 1):
        y[k, 0] = x ** k
    return y

def calcSurf(psi, var, iterat):
    real_nele = 21
    conf_inter = 1
    S = torch.linspace(0, 1, real_nele + 1)
    X = torch.linspace(-1, 1, 101)
    # self.S, self.X = torch.meshgrid(S, X, indexing='ij')
    yMean = torch.zeros((real_nele + 1, X.size(dim=0)))
    yHigh = torch.zeros((real_nele + 1, X.size(dim=0)))
    yLow = torch.zeros((real_nele + 1, X.size(dim=0)))
    # psi = torch.transpose(psi, 0, 1)
    psi = psi.cpu()
    #var = torch.diag(var)
    var = torch.reshape(var, (-1, 1))
    var = var.detach().cpu().numpy()

    for kkk in range(1, real_nele):
        for iii in range(0, X.size(dim=0)):
            polvecx = polynomial(X[iii])
            tess = torch.matmul(psi, polvecx)
            yMean[1:real_nele, iii] = torch.squeeze(torch.matmul(psi, polvecx), dim=1)
            yHigh[1:real_nele, iii] = yMean[1:real_nele, iii] + conf_inter * torch.sqrt(
                torch.from_numpy(var[kkk - 1, :]))
            yLow[1:real_nele, iii] = yMean[1:real_nele, iii] - conf_inter * torch.sqrt(
                torch.from_numpy(var[kkk - 1, :]))
    yMean[0, :] = torch.ones(X.size(dim=0)) * 0
    yHigh[0, :] = torch.ones(X.size(dim=0)) * 0
    yLow[0, :] = torch.ones(X.size(dim=0)) * 0
    yMean[-1, :] = torch.ones(X.size(dim=0)) * 0
    yHigh[-1, :] = torch.ones(X.size(dim=0)) * 0
    yLow[-1, :] = torch.ones(X.size(dim=0)) * 0
    return yMean, yHigh

psi = torch.tensor([[ 0.0066,  0.0039, -0.0020,  0.0041,  0.0032],
        [ 0.0086,  0.0063,  0.0042,  0.0027,  0.0042],
        [ 0.0128,  0.0064,  0.0109,  0.0069,  0.0101],
        [ 0.0090,  0.0017,  0.0065,  0.0075,  0.0080],
        [ 0.0111,  0.0043,  0.0141,  0.0065,  0.0096],
        [ 0.0095,  0.0082,  0.0106,  0.0084,  0.0108],
        [ 0.0074,  0.0109,  0.0096,  0.0048,  0.0095],
        [ 0.0083,  0.0056,  0.0049,  0.0090,  0.0070],
        [ 0.0063,  0.0082,  0.0057,  0.0017,  0.0067],
        [ 0.0072,  0.0079,  0.0069,  0.0094,  0.0039],
        [ 0.0038,  0.0061,  0.0085,  0.0110,  0.0033],
        [ 0.0059,  0.0063,  0.0122,  0.0064,  0.0052],
        [ 0.0074,  0.0036,  0.0045,  0.0098,  0.0080],
        [ 0.0098,  0.0075,  0.0093,  0.0029,  0.0074],
        [ 0.0117,  0.0051,  0.0080,  0.0072,  0.0100],
        [ 0.0121,  0.0054,  0.0064,  0.0039,  0.0036],
        [ 0.0164,  0.0036,  0.0076,  0.0095,  0.0047],
        [ 0.0113, -0.0024,  0.0060,  0.0053, -0.0009],
        [ 0.0093,  0.0011,  0.0020,  0.0007,  0.0016],
        [ 0.0061, -0.0043,  0.0021,  0.0033, -0.0008]])
var = torch.tensor([1.5422e-05, 1.4528e-05, 1.4787e-05, 1.4834e-05, 1.5193e-05, 1.5241e-05,
        1.5314e-05, 1.5287e-05, 1.4747e-05, 1.4848e-05, 1.4967e-05, 1.5183e-05,
        1.5473e-05, 1.5604e-05, 1.5596e-05, 1.5098e-05, 1.4681e-05, 1.4560e-05,
        1.4380e-05, 1.5530e-05])
iterat = 1000
yMean, yHigh = calcSurf(psi, var, iterat)
sigm = torch.diag(yHigh - yMean)




print("end")


class plotApproxVsSol:
    def __init__(self, psi, poly_pow, pde, sigma_px, iterat, display_plots, model):
        self.iterat = iterat
        self.data_x = model.data_x.clone().detach().numpy()
        self.pde = pde
        self.psi = psi
        self.display_plots = display_plots
        self.poly_pow = poly_pow
        self.nele = pde.effective_nele
        self.tot_nele = pde.nele
        self.conf_inter = 2
        self.conf_inter_px = 1
        nelsq = np.sqrt(self.nele)
        self.fig, self.ax = plt.subplots(self.nele // math.ceil(nelsq) + 1, math.ceil(nelsq)+1,num=18) ### +1 is added for dimx=1
        #self.fig.set_figheight(30)
        #self.fig.set_figwidth(50)
        #self.fig.subplots_adjust(hspace=0.5, wspace=0.35)
        self.fig.set_figheight(8)
        self.fig.set_figwidth(10)
        self.fig.subplots_adjust(hspace=0.3, wspace=0.2)
        self.Strue, self.Xtrue = torch.meshgrid(torch.linspace(0, 1, 101), torch.linspace(-1, 1, 101), indexing='ij')
        self.yTrueSurf = torch.zeros((self.Strue.size(dim=0), self.Xtrue.size(dim=1)))
        self.yHighHist = []
        self.yMeanHist = []
        self.yLowHist = []
        def analsol(s,x):
            y = (-s**2/2+s/2)*np.exp(-x)
            return y
        for i1 in range(0, self.Strue.size(dim=0)):
            for i2 in range(0, self.Xtrue.size(dim=1)):
                self.yTrueSurf[i1, i2] = analsol(self.Strue[i1, i2], self.Xtrue[i1, i2])
        #self.yTrueSurf = analsol(self.Strue[)
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
                y = analsol(s, self.data_x)
                self.ax[j, k].scatter(self.data_x, analsol(s, self.data_x),c='r', marker='o',s=100)
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
                #self.ax[jj, kk].set_yscale('log')
                j = j + 1
                if j == self.nele:
                    break
            if j == self.nele:
                break

    def polynomial(self, x):
        y = torch.zeros(self.poly_pow + 1, 1)
        for k in range(0, self.poly_pow + 1):
            y[k, 0] = x ** k
        return y

    def calcSurf(self, psi, var, iterat):
        real_nele = self.pde.nele
        S = torch.linspace(0, 1, real_nele + 1)
        X = torch.linspace(-1, 1, 101)
        #self.S, self.X = torch.meshgrid(S, X, indexing='ij')
        yMean = torch.zeros((real_nele + 1, X.size(dim=0)))
        yHigh = torch.zeros((real_nele + 1, X.size(dim=0)))
        yLow = torch.zeros((real_nele + 1, X.size(dim=0)))
        # psi = torch.transpose(psi, 0, 1)
        psi = psi.cpu()
        var = torch.diag(var)
        var = torch.reshape(var, (-1, 1))
        var = var.detach().cpu().numpy()

        self.leg_tuple = self.leg_tuple + ("Approx. Solution, iter = " + str(iterat),)
        for kkk in range(1, real_nele):
            for iii in range(0, X.size(dim=0)):
                polvecx = self.polynomial(X[iii])
                tess = torch.matmul(psi, polvecx)
                yMean[1:real_nele, iii] = torch.squeeze(torch.matmul(psi, polvecx), dim=1)
                yHigh[1:real_nele, iii] = yMean[1:real_nele, iii] + self.conf_inter * torch.sqrt(
                    torch.from_numpy(var[kkk - 1, :]))
                yLow[1:real_nele, iii] = yMean[1:real_nele, iii] - self.conf_inter * torch.sqrt(
                    torch.from_numpy(var[kkk - 1, :]))
        yMean[0, :] = torch.ones(X.size(dim=0)) * self.pde.lBoundDir
        yHigh[0, :] = torch.ones(X.size(dim=0)) * self.pde.lBoundDir
        yLow[0, :] = torch.ones(X.size(dim=0)) * self.pde.lBoundDir
        yMean[-1, :] = torch.ones(X.size(dim=0)) * self.pde.rBoundDir
        yHigh[-1, :] = torch.ones(X.size(dim=0)) * self.pde.rBoundDir
        yLow[-1, :] = torch.ones(X.size(dim=0)) * self.pde.rBoundDir

        self.yHighHist.append(yHigh)
        self.yMeanHist.append(yMean)
        self.yLowHist.append(yLow)

    def make3dAnimation(self):
        self.figsurf, self.surf = plt.subplots(subplot_kw={"projection": "3d"})
        real_nele = self.pde.nele
        S = torch.linspace(0, 1, real_nele + 1)
        X = torch.linspace(-1, 1, 101)
        self.S, self.X = torch.meshgrid(S, X, indexing='ij')
        # psi = torch.transpose(psi, 0, 1)
        testtt = self.yHighHist[0]
        surfH = self.surf.plot_wireframe(self.S, self.X, self.yHighHist[0], color='gray', alpha=0.3,
                                 linewidth=0.01, antialiased=False, label='Upper Confidence Interval')
        surfL = self.surf.plot_wireframe(self.S, self.X, self.yLowHist[0], color='gray', alpha=0.3,
                                 linewidth=0.01, antialiased=False, label='Lower Confidence Interval')
        surfM = self.surf.plot_surface(self.S, self.X, self.yMeanHist[0], cmap='Reds', alpha=0.5, linewidth=0.05, antialiased=False,
                               label='Approximate Solution')
        surfT = self.surf.plot_surface(self.Strue, self.Xtrue, self.yTrueSurf, cmap='Greens', alpha=0.5,
                               linewidth=0.01, antialiased=False, label='True Solution')
        ConfPatch = mpatches.Patch(color='gray', label='Posterior Solution $\pm 2\sigma$')
        ApproxPatch = mpatches.Patch(color='red', label='Posterior Solution')
        TruePatch = mpatches.Patch(color='green', label='True Solution')
        self.surf.legend(handles=[ConfPatch, ApproxPatch, TruePatch])
        self.surf.set_xlabel('Space: s', fontsize=10)
        self.surf.set_ylabel('Uncertain Input: x', fontsize=10)
        self.surf.set_zlabel('Solution Value: y', fontsize=10)
        plot_title = "Plot of Approximate Vs True Solution for different x"
        self.figsurf.suptitle(plot_title, fontsize=14)
        def animate(i):
            self.surf.clear()
            #text = self.surf.set_text(0.75, 0.85, 0.85, 'Iteration: %d' % i)
            self.surf.set_zlim(0, torch.max(self.yTrueSurf))
            self.surf.legend(handles=[ConfPatch, ApproxPatch, TruePatch])
            self.surf.set_xlabel('Space: s', fontsize=10)
            self.surf.set_ylabel('Uncertain Input: x', fontsize=10)
            self.surf.set_zlabel('Solution Value: y', fontsize=10)
            text = self.surf.text(0.75, 0.85, '')
            surfM = self.surf.plot_surface(self.S, self.X, self.yMeanHist[i], cmap='Reds', alpha=0.5, linewidth=0.05, antialiased=False,
                               label='Approximate Solution')
            surfT = self.surf.plot_surface(self.Strue, self.Xtrue, self.yTrueSurf, cmap='Greens', alpha=0.5,
                                           linewidth=0.01, antialiased=False, label='True Solution')
            surfH = self.surf.plot_wireframe(self.S, self.X, self.yHighHist[i], color='gray', alpha=0.9,
                                             linewidth=0.1, antialiased=False, label='Upper Confidence Interval')
            surfL = self.surf.plot_wireframe(self.S, self.X, self.yLowHist[i], color='gray', alpha=0.9,
                                             linewidth=0.1, antialiased=False, label='Lower Confidence Interval')
            text.set_text("Iteration = %d" % i)
            return surfM, surfT, surfH, surfL

        an1 = animation.FuncAnimation(self.figsurf, animate, interval=20, blit=True, save_count=(len(self.yMeanHist)-1))
        an1.save("./results//approxVsTrueSol/ani.mp4", dpi=300)
        plt.show()

    def make3dAnimationHeatmap(self):
        self.figsurf, self.surf = plt.subplots(2)
        self.figsurf.set_figheight(7)
        self.figsurf.set_figwidth(7)
        real_nele = self.pde.nele
        S = torch.linspace(0, 1, real_nele + 1)
        X = torch.linspace(-1, 1, 101)
        self.S, self.X = torch.meshgrid(S, X, indexing='ij')
        #text = self.surf.text(0.75, 0.85, 0.85, 'test')
        # psi = torch.transpose(psi, 0, 1)
        testtt = self.yHighHist[0]
        """
        surfH = self.surf.imshow(self.S, self.X, self.yHighHist[0], color='gray', alpha=0.3,
                                 linewidth=0.01, antialiased=False, label='Upper Confidence Interval')
        surfL = self.surf.imshow(self.S, self.X, self.yLowHist[0], color='gray', alpha=0.3,
                                 linewidth=0.01, antialiased=False, label='Lower Confidence Interval')
        """
        ### for interpolation add shading='gouraud'
        surfM = self.surf[0].pcolormesh(self.S, self.X, self.yMeanHist[0], cmap='jet', vmax=torch.max(self.yTrueSurf),
                                        vmin=torch.min(self.yTrueSurf), antialiased=False,
                               label='Approximate Solution')
        surfT = self.surf[1].pcolormesh(self.Strue, self.Xtrue, self.yTrueSurf, cmap='jet',vmax=torch.max(self.yTrueSurf),
                                        vmin=torch.min(self.yTrueSurf), antialiased=False,
                                        label='True Solution')
        ConfPatch = mpatches.Patch(color='gray', label='Posterior Solution $\pm 2\sigma$')
        ApproxPatch = mpatches.Patch(color='red', label='Posterior Solution')
        TruePatch = mpatches.Patch(color='green', label='True Solution')
        #self.surf[1].legend(handles=[ConfPatch, ApproxPatch, TruePatch])
        text = self.figsurf.text(0.4, 0.9, '', fontsize=12)
        self.surf[0].set_xlabel('Space: s', fontsize=10)
        self.surf[0].set_ylabel('Uncertain Input: x', fontsize=10)
        self.surf[1].set_xlabel('Space: s', fontsize=10)
        self.surf[1].set_ylabel('Uncertain Input: x', fontsize=10)
        self.figsurf.colorbar(surfT)
        self.figsurf.colorbar(surfM)
        #self.surf.set_zlabel('Solution Value: y', fontsize=10)
        plot_title = "Plot of Approximate Vs True Solution for different x"
        self.figsurf.suptitle(plot_title, fontsize=14)
        def animate(i):
            self.surf[0].clear()
            #text = self.surf.set_text(0.75, 0.85, 0.85, 'Iteration: %d' % i)
            #self.surf.set_zlim(0, torch.max(self.yTrueSurf))
            #self.surf.legend(handles=[ConfPatch, ApproxPatch, TruePatch])
            text.set_text("Iteration = %d" % i)
            self.surf[0].set_xlabel('Space: s', fontsize=10)
            self.surf[0].set_ylabel('Uncertain Input: x', fontsize=10)
            #self.surf.set_zlabel('Solution Value: y', fontsize=10)
            surfM = self.surf[0].pcolormesh(self.S, self.X, self.yMeanHist[i], cmap='jet',vmax=torch.max(self.yTrueSurf),
                                        vmin=torch.min(self.yTrueSurf), antialiased=False,
                               label='Approximate Solution')
            """
            surfH = self.surf.imshow(self.S, self.X, self.yHighHist[i], color='gray', alpha=0.9,
                                             linewidth=0.1, antialiased=False, label='Upper Confidence Interval')
            surfL = self.surf.imshow(self.S, self.X, self.yLowHist[i], color='gray', alpha=0.9,
                                             linewidth=0.1, antialiased=False, label='Lower Confidence Interval')
            """
            #text.set_text("Iteration = %d" % i)
            return surfM, surfT

        an1 = animation.FuncAnimation(self.figsurf, animate, interval=20, blit=True, save_count=(len(self.yMeanHist)-1))
        an1.save("./results/approxVsTrueSol/ani.mp4", dpi=300)
        plt.show()
    def add_surface(self, psi, var, iterat):
        real_nele = self.pde.nele
        S = torch.linspace(0, 1, real_nele + 1)
        X = torch.linspace(-1, 1, 101)
        self.S, self.X = torch.meshgrid(S, X, indexing='ij')
        yMean = torch.zeros((real_nele+1, X.size(dim=0)))
        yHigh = torch.zeros((real_nele+1, X.size(dim=0)))
        yLow = torch.zeros((real_nele+1, X.size(dim=0)))
        # psi = torch.transpose(psi, 0, 1)
        psi = psi.cpu()
        var = torch.diag(var)
        var = torch.reshape(var, (-1, 1))
        var = var.detach().cpu().numpy()

        self.leg_tuple = self.leg_tuple + ("Approx. Solution, iter = " + str(iterat),)
        for kkk in range(1, real_nele):
            for iii in range(0, X.size(dim=0)):
                polvecx = self.polynomial(X[iii])
                tess = torch.matmul(psi, polvecx)
                yMean[1:real_nele, iii] = torch.squeeze(torch.matmul(psi, polvecx), dim=1)
                yHigh[1:real_nele, iii] = yMean[1:real_nele, iii] + self. conf_inter * torch.sqrt(torch.from_numpy(var[kkk-1, :]))
                yLow[1:real_nele, iii] = yMean[1:real_nele, iii] - self.conf_inter * torch.sqrt(torch.from_numpy(var[kkk-1, :]))
        yMean[0, :] = torch.ones(X.size(dim=0)) * self.pde.lBoundDir
        yHigh[0, :] = torch.ones(X.size(dim=0)) * self.pde.lBoundDir
        yLow[0, :] = torch.ones(X.size(dim=0)) * self.pde.lBoundDir
        yMean[-1, :] = torch.ones(X.size(dim=0)) * self.pde.rBoundDir
        yHigh[-1, :] = torch.ones(X.size(dim=0)) * self.pde.rBoundDir
        yLow[-1, :] = torch.ones(X.size(dim=0)) * self.pde.rBoundDir


        self.surf.plot_wireframe(self.S, self.X, yHigh, color='gray', alpha=0.3,
                                            linewidth=0.01, antialiased=False, label='Upper Confidence Interval')
        self.surf.plot_wireframe(self.S, self.X, yLow, color='gray', alpha=0.3,
                                           linewidth=0.01, antialiased=False, label='Lower Confidence Interval')
        self.surf.plot_surface(self.S, self.X, yMean, cmap='Reds', alpha=0.5,                                            linewidth=0.05, antialiased=False, label='Approximate Solution')
        self.surf.plot_surface(self.Strue, self.Xtrue, self.yTrueSurf, cmap='Greens', alpha=0.5,
                                          linewidth=0.01, antialiased=False, label='True Solution')
        ConfPatch = mpatches.Patch(color='gray', label='Posterior Solution $\pm 2\sigma$')
        ApproxPatch = mpatches.Patch(color='red', label='Posterior Solution')
        TruePatch = mpatches.Patch(color='green', label='True Solution')
        self.surf.legend(handles=[ConfPatch, ApproxPatch, TruePatch])
        self.surf.set_xlabel('Space: s', fontsize=10)
        self.surf.set_ylabel('Uncertain Input: x', fontsize=10)
        self.surf.set_zlabel('Solution Value: y', fontsize=10)




    def show(self, iter=0):
        #plot_title = "Plot of Approximate Vs True Solution \n" + "\n".join(wrap(title_id))
        plot_title = "Plot of Approximate Vs True Solution for different x"
        self.fig.suptitle(plot_title, fontsize=16)
        self.figsurf.suptitle(plot_title, fontsize=16)
        if not os.path.exists('./results/approxVsTrueSol/'):
            os.makedirs('./results/approxVsTrueSol/')
        if not os.path.exists('./results/approxVsTrueSol/surfPlots%d/' % iter):
            os.makedirs('./results/approxVsTrueSol/surfPlots%d/' % iter)
        self.fig.savefig("./results/approxVsTrueSol/approxVsTrueSol", dpi=300, bbox_inches='tight')
        if False:
            for ii in range(0, 360, 30):
                self.surf.view_init(elev=10., azim=ii)
                self.figsurf.savefig("./results/approxVsTrueSol/surfPlots%d/surf%d.png" % (iter, ii), dpi=300, bbox_inches='tight')
        #self.fig.tight_layout()
        if self.display_plots:
            self.fig.show()
            self.figsurf.show()