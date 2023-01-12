"""
The parameters of the program are defined here.
"""
device='cpu'   # 'cpu' or 'cuda'
nele = 5
Nx_samp = 1 # 5 is good
phase1 = 101 # Number of iteration out of 100 e.g 75 out of 100
phase2 = 100 # Number of iteration out of 100 e.g 90 out of 100
mean_px = 0
sigma_px = 0.1
Iter_svi = 50
Iter_outer = 50
lr = 0.001  ### In the PolynomialMultivariate case reducing the learning rate can have a positive effect
lr_for_phi = 0.005
eigRelax = 0.05
Iter_grad = 1
sigma_r = 10 ** (-1.5)  ### Very sensitivy to changes (For the multivariate is much different)
sigma_r_mult = 1.00
sigma_w = 10000000
Iter_total = Iter_outer * Iter_svi
poly_pow = 2
sigma_r_f = round(sigma_r * sigma_r_mult ** Iter_outer, 2)
mode = "Polynomial"  ### "TrueSol" or "Polynomial"
powerIterTol = 10 ** (-5)
display_plots = True
allRes = False

#Boundary Conditions
lBoundDir = 1
rBoundDir = 1
lBoundNeu = None
rBoundNeu = None
rhs = -1