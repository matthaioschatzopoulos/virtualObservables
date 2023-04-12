"""
The parameters of the program are defined here.
"""
device='cpu'   # 'cpu' or 'cuda'
nele = 10
Nx_samp = 11 # 5 is good
Nx_samp_phiOpt = 1000 # Checked manually and 100 seem ok
phase1 = 101 # Number of iteration out of 100 e.g 75 out of 100
phase2 = 100 # Number of iteration out of 100 e.g 90 out of 100
mean_px = 0
sigma_px = 1
IterSvi = 10
Iter_svi = IterSvi*20
Iter_outer = 1
#Me grammiki paremboli gia nele=19 ekana 5000 iter_outer for approx 550, therefore I need 5 times that (assuming linear
#extrapolation is correct
### Phi Optimization ###
lr = 0.00001  ### In the PolynomialMultivariate case reducing the learning rate can have a positive effect
gradLr = 0.00025 ### It is probably good to be the same as the SVI lr for stability reasons
Iter_grad = 10 #Epochs number for phiGradOpt method
eigRelax = None
sigma_r = 10 ** (-2.5)  ### Very sensitivy to changes (For the multivariate is much different)
### If sigma_r scheduler is used ###
sigma_r_init = 10**(-0)
sigma_r_final = 10**(-1.5)
sigma_r_mult = 1.00
sigma_w = 10**8
Iter_total = Iter_outer * Iter_svi
poly_pow = 4
sigma_r_f = round(sigma_r * sigma_r_mult ** Iter_outer, 2)
mode = "Polynomial"  ### "TrueSol" or "Polynomial"
powerIterTol = 10 ** (-5)
display_plots = True
allRes = False
runTests = False
surgtType = 'DeltaNoiseless'
phiValidMode = True

#Boundary Conditions
lBoundDir = 0
rBoundDir = 0
lBoundNeu = None
rBoundNeu = None
rhs = -1