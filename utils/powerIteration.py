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
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal
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
torch.set_default_tensor_type(torch.DoubleTensor)

#torch.set_printoptions(precision=8)

def powerIteration(A, etol, max_iter=100):
    # Converting numpy arrays to torch
    if isinstance(A, np.ndarray):
        a = torch.from_numpy(A)
    else:
        a = A

    # Initialization
    ndim = a.size()[0]
    b = torch.ones((ndim, 1))
    b = b + torch.reshape(torch.rand(ndim), (-1, 1))*0.01
    b = b/torch.linalg.vector_norm(b)
    b = b.to(dtype=torch.float64)

    # Main Loop
    counter = 0
    bold = 10*b

    for i in range(0, max_iter):
        bold = b
        b = torch.matmul(a, b)
        b = b/torch.linalg.vector_norm(b)
        counter = counter + 1
        diff = torch.linalg.norm(bold - b)/torch.linalg.norm(bold)
        if diff < etol:
            break
    return b, counter

