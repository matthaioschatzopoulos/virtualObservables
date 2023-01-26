"""
The parameters of the program are defined here.
"""
device='cpu'   # 'cpu' or 'cuda'
nele = 3
Nx_samp = 10 # 5 is good
Nx_samp_phiOpt = 100 # Checked manually and 100 seem ok
phase1 = 101 # Number of iteration out of 100 e.g 75 out of 100
phase2 = 100 # Number of iteration out of 100 e.g 90 out of 100
mean_px = 0
sigma_px = 1
Iter_svi = 4000
Iter_outer = 1

### Phi Optimization ###
Iter_grad = 1 #Epochs number for phiGradOpt method
lr = 0.001  ### In the PolynomialMultivariate case reducing the learning rate can have a positive effect
gradLr = 0.01
eigRelax = None
sigma_r = 10 ** (-3)  ### Very sensitivy to changes (For the multivariate is much different)
### If sigma_r scheduler is used ###
sigma_r_init = 10**(-0)
sigma_r_final = 10**(-1.5)
sigma_r_mult = 1.00
sigma_w = 10000000
Iter_total = Iter_outer * Iter_svi
poly_pow = 2
sigma_r_f = round(sigma_r * sigma_r_mult ** Iter_outer, 2)
mode = "Polynomial"  ### "TrueSol" or "Polynomial"
powerIterTol = 10 ** (-5)
display_plots = True
allRes = False
runTests = True
surgtType = 'Delta'
phiValidMode = False

#Boundary Conditions
lBoundDir = 0
rBoundDir = 0
lBoundNeu = None
rBoundNeu = None
rhs = -1