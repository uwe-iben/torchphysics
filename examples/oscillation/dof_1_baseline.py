# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 14:35:18 2021

@author: KRD2RNG
"""

import numpy as np
import pandas as pd
from scipy import signal as sg
import matplotlib.pyplot as plt
import matplotlib


import torch
import numpy as np
import pytorch_lightning as pl
import pytorch
from timeit import default_timer as timer

from neural_diff_eq.problem import Variable
from neural_diff_eq.setting import Setting
from neural_diff_eq.problem.domain import (Rectangle,
                                           Interval)
from neural_diff_eq.problem.condition import (DirichletCondition,NeumannCondition,
                                              DiffEqCondition,
                                              DataCondition)
from neural_diff_eq.models.fcn import SimpleFCN
from neural_diff_eq import PINNModule
from neural_diff_eq.utils import laplacian, gradient
from neural_diff_eq.utils.fdm import FDM, create_validation_data
from neural_diff_eq.utils.plot import Plotter
from neural_diff_eq.utils.evaluation import (get_min_max_inside,
                                             get_min_max_boundary)
from neural_diff_eq.setting import Setting

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # select GPUs to use

#pl.seed_everything(43) # set a global seed
torch.cuda.is_available()

matplotlib.style.use('default')

stiffness = 1e4
damping = 4
mass = 1
SampleFrequency = 1024
t_end = 5
t = np.linspace(0, t_end, t_end * SampleFrequency)
#%%
def time_evolution(A, B, C, D, u, x, t):
    """calculate time response"""
    y = np.zeros((len(D), len(t)))
    for k in range(0, len(t)-1):
        y[:, k] = C@x.ravel() + D@u[:, k]
        x = A@x.ravel() + B@u[:, k]
    return(pd.DataFrame(data=y.T, index=t, columns=["state_space"]))

def state_space_dof_1(u, x0):
    Ac = np.array([[0, 1], [-stiffness/mass, -damping/mass]])
    Bc = np.array([[0], [1/mass]])
    Cc = np.array([[1, 0]])
    Dc = np.array([[0]])

    # Discrete
    A, B, C, D, _ = sg.cont2discrete((Ac, Bc, Cc, Dc), dt=1/SampleFrequency)

    y = time_evolution(A, B, C, D, u, x0, t)
    return y

def analytical_dof_1(u, x0):
    delta = damping / (2 * mass)
    omega_0 = np.sqrt(stiffness / mass)
    omega_d = np.sqrt(omega_0**2 - delta**2)
    y = np.exp(-delta * t) * (((x0[1] + delta * x0[0]) / omega_d) * np.sin(omega_d * t) + x0[0] * np.cos(omega_d * t))
    return pd.DataFrame(data=y.T, index=t)

#%% Analytical solution
dirac = np.array([sg.unit_impulse(len(t), idx=0)])
x0 = np.array([[1], [0]])
y = state_space_dof_1(dirac, x0)
y["analytical"] = analytical_dof_1(dirac, x0)
y.plot()

#%% PINN approach
# u_tt + delta * u_t + omega**2 * u = f(t)

norm = torch.nn.MSELoss()

time = Variable(name='time',
              order=1,
              domain=Interval(low_bound=0,
                              up_bound=t_end),
              train_conditions={},
              val_conditions={})
c = Variable(name='stiffness',
              order=0,
              domain=Interval(low_bound=1e3,
                              up_bound=1e5),
              train_conditions={},
              val_conditions={})

# the same can be done to achieve an initial condition for the time axis:
def time_dirichlet_fun(input):
    return np.ones_like(input['time'])

def time_neumann_fun(input):
    return np.zeros_like(input['time'])

# to get only initial (and not end-) values, we can set boundary_sampling_strategy to sample
# only one bound of the interval
time.add_train_condition(DirichletCondition(dirichlet_fun=time_dirichlet_fun,
                                          name='dirichlet',
                                          norm=norm,
                                          dataset_size=150,
                                          boundary_sampling_strategy='lower_bound_only',
                                          data_plot_variables=True))

time.add_train_condition(NeumannCondition(neumann_fun=time_neumann_fun,
                                          name='neumann',
                                          norm=norm,
                                          dataset_size=150,
                                          boundary_sampling_strategy='lower_bound_only',
                                          data_plot_variables=True))

# a pde function handle takes the output and the input (as a dict again) of the network. We can use
# functions like 'laplacian' from the utils part to compute common differential operators.
def ode_oscillation(u, input):
    return laplacian(u, input['time']) + (damping / (2 * mass)) * gradient(u, input['time']) + pytorch.sqrt(input["stiffness"]/mass) * u

# a DiffEqCondition works similar to the boundary condiitions
train_cond = DiffEqCondition(pde=ode_oscillation,
                              name='ode_oscillation',
                              norm=norm,
                              sampling_strategy='random',
                              weight=1.0,
                              dataset_size=5000,
                              data_plot_variables=False)#)('time'))




    
    