import torch
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
def hat(x, x_node, h):
    # Reshape x and x_node to enable broadcasting
    x = x.unsqueeze(-1)
    x_node = x_node.view(-1, 1, 1)

    # Compute boolean masks
    mask1 = (x >= x_node - h) & (x <= x_node)
    mask2 = (x < x_node - h) | (x > x_node + h)

    # Compute intermediate tensor
    out1 = torch.where(mask1, (x - (x_node - h)) / h, 1 - (x - x_node) / h)

    # Apply final mask and reshape to get output matrix
    return torch.reshape(torch.where(mask2, torch.zeros_like(out1), out1), [x_node.size(dim=0), -1])


def hatGrad(x, x_node, h):
    # Reshape x and x_node to enable broadcasting
    x = x.unsqueeze(-1)
    x_node = x_node.view(-1, 1, 1)

    # Compute boolean masks
    mask1 = (x >= x_node - h) & (x <= x_node)
    mask2 = (x < x_node - h) | (x > x_node + h)

    # Compute intermediate tensor
    out1 = torch.where(mask1, 1 / h, -1 / h)

    # Apply final mask and reshape to get output matrix
    return torch.reshape(torch.where(mask2, torch.zeros_like(out1), out1), [x_node.size(dim=0), -1])

class Rectangular2D:
    def N1(self, node_corrs, grid): # low left corner
        return (grid[0, :, :] - node_corrs[0, 1, 0])*(grid[1, :, :] - node_corrs[1, 0, 1])
    def N2(self, node_corrs, grid): # low right corner
        return -(grid[0, :, :] - node_corrs[0, 0, 0])*(grid[1, :, :] - node_corrs[1, 0, 1])
    def N3(self, node_corrs, grid): # up left corner
        return -(grid[0, :, :] - node_corrs[0, 1, 0])*(grid[1, :, :] - node_corrs[1, 0, 0])
    def N4(self, node_corrs, grid): # up right corner
        return (grid[0, :, :] - node_corrs[0, 0, 0])*(grid[1, :, :] - node_corrs[1, 0, 0])

    def VectorN(self, sNodes, s, dl):
        return torch.stack((self.N1(sNodes, s), self.N2(sNodes, s), self.N3(sNodes, s), self.N4(sNodes, s)), dim=0)/dl**2
    def cWeightFunc(self, wAtLocalNodes, dl, sNodes, s):
        """
        :param wAtLocalNodes: The values of w for this (local) node
        :param dl: The length/width of each element (equal for x and y coordinates here)
        :param sNodes: The coordinates of the nodes for this element
        :param s: The coordinates of the grid for this local element (space s)
        :return: The contribution to the value of the weightingfunction/field/solution (w(s)/u(s)/y(s))
        """
        tess = self.N1(sNodes, s)/dl**2
        NShapeFuncs = self.VectorN(sNodes, s, dl)
        wAtLocalNodes = torch.reshape(wAtLocalNodes, [-1])
        return torch.einsum('i,ijk->jk', wAtLocalNodes, NShapeFuncs)

    def cGlobalWeightFunc(self, w, sgrid_x, grid, sgrid, dl):
        """
        :param w: Global nodal points of the weighting function
        :param sgrid_x: Global integration x-grid (by keeping the first tensor from the output of the meshgrid command)
        :param grid: The Global grid (The original, not the one for the integration)
        :param sgrid: The Global expanded grid (The one used for the integration)
        :param dl: The distance between 2 nodes (assumed equal for all the grid)
        :return: The global weighting function, ready for integration in space.
        Example: res = self.shapeFunc.cGlobalWeightFunc(self.wtest, self.sgrid_x, self.grid, self.sgrid, self.dl)
        """
        z = torch.zeros_like(sgrid_x)
        nele = int(1/dl)
        for i in range(0, nele):
            for j in range(0, nele):
                z[(j * 10):((j + 1) * 10 + 1), (i * 10):((i + 1) * 10 + 1)] = self.cWeightFunc(
                    w[i:(i + 2), j:(j + 2)], dl, grid[:, i:(i + 2), j:(j + 2)],
                    sgrid[:, (i * 10):((i + 1) * 10 + 1), (j * 10):((j + 1) * 10 + 1)])
        return z

class rbfInterpolation:
    def __init__(self, grid, sgrid, tau):
        """
        :param grid: Center points (in the form of a grid) of each RBF shape function
        :param sgrid: Integration Grid for performing the numerical integration
        :param tau: Scaling parameters for each shape function. Currently this is the same for every shape function.
        :param dl:
        """
        self.grid = grid
        self.sgrid = sgrid
        self.tau = tau
        self.c = 1.
        self.dirFunc = self.ByConstrFuncForDirichlet()
        self.dirFuncGradx()
        self.dirFuncGrady()
        self.shapeFunc = torch.vmap(self.rbf)(torch.reshape(self.grid[0, :, :], [-1]), torch.reshape(self.grid[1, :, :], [-1]))
        self.shapeFuncDx = torch.vmap(self.rbfGradx)(torch.reshape(self.grid[0, :, :], [-1]),
                                  torch.reshape(self.grid[1, :, :], [-1]))
        self.shapeFuncDy = torch.vmap(self.rbfGrady)(torch.reshape(self.grid[0, :, :], [-1]),
                                  torch.reshape(self.grid[1, :, :], [-1]))

        self.shapeFunc = torch.vmap(self.filtRbf)(torch.reshape(self.grid[0, :, :], [-1]),
                                              torch.reshape(self.grid[1, :, :], [-1]))
        self.shapeFuncDx = torch.vmap(self.filtRbfGradx)(torch.reshape(self.grid[0, :, :], [-1]),
                                                     torch.reshape(self.grid[1, :, :], [-1]))
        self.shapeFuncDy = torch.vmap(self.filtRbfGrady)(torch.reshape(self.grid[0, :, :], [-1]),
                                                     torch.reshape(self.grid[1, :, :], [-1]))


        if False:
            torch.set_printoptions(profile='full')
            print("Here lies the shape function R.I.P")

            #print(self.shapeFunc[3, :, :])
            print(self.shapeFuncDx[10, :, 4])

            self.plotShapeFunctions()
            self.plotShapeFunctionsGrad(x=True)
            #self.plotShapeFunctionsGrad(x=False)
            test = 1



    def ByConstrFuncForDirichlet(self):
        """
        It Provides the function with which the weighting function will be multiplied in order to make sure that w(s)=0
        on the boundaries when diriclet conditions are used.
        ATTENTION!: The normalization constant (0.0625) could have a great stabilization effect on the convergence of
        the algorithm, so use it with caution.
        :return:
        """
        return (1-self.sgrid[0, :, :])*self.sgrid[0, :, :]*(1-self.sgrid[1, :, :])*self.sgrid[1, :, :]/self.c
    def dirFuncGradx(self):
        return -self.sgrid[0, :, :]*(1-self.sgrid[1, :, :])*self.sgrid[1, :, :]/self.c + \
               (1-self.sgrid[0, :, :])*(1-self.sgrid[1, :, :])*self.sgrid[1, :, :]/self.c
    def dirFuncGrady(self):
        return -(1-self.sgrid[0, :, :])*self.sgrid[0, :, :]*self.sgrid[1, :, :]/self.c +\
               (1-self.sgrid[0, :, :])*self.sgrid[0, :, :]*(1-self.sgrid[1, :, :])/self.c
    def rbf(self, xnode, ynode):
        return torch.exp(-self.tau*((xnode-self.sgrid[0, :, :])**2+(ynode-self.sgrid[1, :, :])**2))
    def rbfGradx(self, xnode, ynode):
        return -2 * self.tau * (self.sgrid[0, :, :] - xnode) * \
             torch.exp(-self.tau * ((xnode - self.sgrid[0, :, :]) ** 2 + (ynode - self.sgrid[1, :, :]) ** 2))
    def rbfGrady(self, xnode, ynode):
        return -2 * self.tau * (self.sgrid[1, :, :] - ynode) *\
             torch.exp(-self.tau*((xnode-self.sgrid[0, :, :])**2+(ynode-self.sgrid[1, :, :])**2))

    def filtRbf(self, xnode, ynode):
        return torch.mul(self.rbf(xnode, ynode), self.dirFunc)
    def filtRbfGradx(self, xnode, ynode):
        return torch.mul(self.rbfGradx(xnode, ynode), self.dirFunc) +\
               torch.mul(self.rbf(xnode, ynode), self.dirFuncGradx())
    def filtRbfGrady(self, xnode, ynode):
        return torch.mul(self.rbfGrady(xnode, ynode), self.dirFunc) +\
               torch.mul(self.rbf(xnode, ynode), self.dirFuncGrady())


    def cWeighFunc(self, phi):
        """
        :param phi: Coefficients of the expansion (NOT RELATED to the grid and WITHOUT any physical meaning). This should
        be a 1D tensor consisting of N entries.
        :return: The weighting function: w(s) = \sum_0^N phi_i * u_i(s)
        """
        return torch.einsum('i,ijk->jk', phi, self.shapeFunc)

    def cTrialSolution(self, phi):
        """
        :param phi: Coefficients of the expansion (NOT RELATED to the grid and WITHOUT any physical meaning). This should
        be a 1D tensor consisting of N entries.
        :return: The weighting function: w(s) = \sum_0^N phi_i * u_i(s)
        """
        return torch.einsum('i,ijk->jk', phi, self.shapeFunc)
    def cWeighFuncOrig(self, phi):
        """
        :param phi: Coefficients of the expansion (NOT RELATED to the grid and WITHOUT any physical meaning). This should
        be a 1D tensor consisting of N entries.
        :return: The weighting function: w(s) = \sum_0^N phi_i * u_i(s)
        """
        return torch.einsum('i,ijk->jk', phi, self.shapeFunc)
    def cdWeighFunc(self, phi):
        """
        :param phi: Coefficients of the expansion (NOT RELATED to the grid and WITHOUT any physical meaning). This should
        be a 1D tensor consisting of N entries.
        :return: The weighting function: w(s) = \sum_0^N phi_i * u_i(s)
        """
        dx = torch.einsum('i,ijk->jk', phi, self.shapeFuncDx)
        dy = torch.einsum('i,ijk->jk', phi, self.shapeFuncDy)
        return torch.stack((dx, dy), dim=0)

    def rbfScipyInterp(self, w_coeffs): ### Not used for now
        t1 = torch.stack((self.grid[0, :, :].ravel(), self.grid[1, :, :].ravel()), dim=1)
        t2 = w_coeffs.ravel()
        interp = RBFInterpolator(torch.stack((self.grid[0, :, :].ravel(),
                                              self.grid[1, :, :].ravel()), dim=1), w_coeffs.ravel())
        return torch.reshape(torch.from_numpy(interp(torch.stack((self.sgrid[0, :, :].ravel(),
                                                 self.sgrid[1, :, :].ravel()), dim=1))), [self.sgrid.size(dim=1), -1]).T

    def plotShapeFunctions(self):
        numOfShFuncs = self.grid.size(dim=1)
        gridSize = self.sgrid.size(dim=1)
        fig, axs = plt.subplots(numOfShFuncs, numOfShFuncs, figsize=(12, 12))
        temp = self.shapeFunc.view(numOfShFuncs, numOfShFuncs, gridSize, gridSize)
        for i in range(numOfShFuncs):
            for j in range(numOfShFuncs):
                axs[i, j].pcolormesh(self.sgrid[0, :, :], self.sgrid[1, :, :], temp[i, j, :, :].T,
                                  cmap='coolwarm', shading='auto')
                axs[i, j].set_title("Shape Function: " + str(i*4+j))
                axs[i, j].set_aspect('equal')

        #cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.3])
        #fig.colorbar(axs[0, 0].collections[0], ax=cbar_ax, orientation='horizontal', pad=0.)
        #fig.subplots_adjust(bottom=0.05)
        #plt.tight_layout()
        plt.show()


    def plotShapeFunctionsGrad(self, x=True):
        numOfShFuncs = self.grid.size(dim=1)
        gridSize = self.sgrid.size(dim=1)
        fig, axs = plt.subplots(numOfShFuncs, numOfShFuncs, figsize=(12, 12))
        if x == True:
            temp = self.shapeFuncDx.view(numOfShFuncs, numOfShFuncs, gridSize, gridSize)
        else:
            temp = self.shapeFuncDy.view(numOfShFuncs, numOfShFuncs, gridSize, gridSize)
        for i in range(numOfShFuncs):
            for j in range(numOfShFuncs):
                axs[i, j].pcolormesh(self.sgrid[0, :, :], self.sgrid[1, :, :], temp[i, j, :, :].T,
                                  cmap='coolwarm', shading='auto')
                axs[i, j].set_title("Shape Function: " + str(i*4+j))
                axs[i, j].set_aspect('equal')

        #cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.3])
        #fig.colorbar(axs[0, 0].collections[0], ax=cbar_ax, orientation='horizontal', pad=0.)
        #fig.subplots_adjust(bottom=0.05)
        #plt.tight_layout()
        plt.show()
