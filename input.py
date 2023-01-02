"""
The parameters of the program are defined here.
"""
device='cpu'   # 'cpu' or 'cuda'
nele = 5
Nx_samp = 10 # 5 is good
mean_px = 0
sigma_px = 0.5
Iter_svi = 50
Iter_outer = 200
lr = 0.001  ### In the PolynomialMultivariate case reducing the learning rate can have a positive effect
lr_for_phi = 0.005
eigRelax = 0.05
Iter_grad = 1
sigma_r = 10 ** (-3)  ### Very sensitivy to changes (For the multivariate is much different)
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
lBoundDir = 0
rBoundDir = 0
lBoundNeu = None
rBoundNeu = None
rhs = -1