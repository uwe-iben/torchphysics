# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 14:35:18 2021

@author: KRD2RNG
"""

import os
import numpy as np
import pandas as pd
from scipy import signal as sg

import torch
import pytorch_lightning as pl
from torchphysics.problem import Variable
from torchphysics.setting import Setting
from torchphysics.problem.domain import Interval
from torchphysics.problem.condition import (DirichletCondition,
                                            NeumannCondition, 
                                              DiffEqCondition)
from torchphysics.models.fcn import SimpleFCN
from torchphysics import PINNModule
from torchphysics.utils import laplacian, grad
from torchphysics.utils.plot import _plot


os.environ["CUDA_VISIBLE_DEVICES"] = "0" # select GPUs to use

#pl.seed_everything(43) # set a global seed
torch.cuda.is_available()
# matplotlib.style.use('default')

stiffness = 1e4
damping = 4
mass = 1
SampleFrequency = 1024
t_end = 5
t = np.linspace(0, t_end, t_end * SampleFrequency)
#%%
def calc_omega_0(c, m):
    return np.sqrt(c/m)
def calc_delta(b, m):
    return b/(2*m)
#%%
def time_evolution(A, B, C, D, excitation, x, t):
    """calculate time response"""
    y = np.zeros((len(D), len(t)))
    for k in range(0, len(t)-1):
        y[:, k] = C @ x.ravel() + D @ excitation[:, k]
        x = A @ x.ravel() + B @ excitation[:, k]
    return(pd.DataFrame(data=y.T, index=t, columns=["state_space"]))

def state_space_dof_1(excitation, x0):
    Ac = np.array([[0, 1], [-stiffness/mass, -damping/mass]])
    Bc = np.array([[0], [1/mass]])
    Cc = np.array([[1, 0]])
    Dc = np.array([[0]])

    # Discrete
    A, B, C, D, _ = sg.cont2discrete((Ac, Bc, Cc, Dc), dt=1/SampleFrequency)

    y = time_evolution(A, B, C, D, excitation, x0, t)
    return y

def analytical_dof_1(x0):
    delta = calc_delta(damping, mass)
    omega_0 = calc_omega_0(stiffness, mass)
    omega_d = np.sqrt(omega_0**2 - delta**2)
    y = np.exp(-delta * t) * (((x0[1] + delta * x0[0]) / omega_d) * np.sin(omega_d * t) + x0[0] * np.cos(omega_d * t))
    return pd.DataFrame(data=y.T, index=t)

#%% reference solution
dirac = np.array([sg.unit_impulse(len(t), idx=0)])
x0 = np.array([[1], [0]])
y = state_space_dof_1(dirac, x0)
y["analytical"] = analytical_dof_1(x0)
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
# c = Variable(name='stiffness',
#               order=0,
#               domain=Interval(low_bound=1e3,
#                               up_bound=1e5),
#               train_conditions={},
#               val_conditions={})

def time_dirichlet_fun(**input):
    return np.ones_like(input['time'])

def time_neumann_fun(**input):
    return np.zeros_like(input['time'])

time.add_train_condition(DirichletCondition(dirichlet_fun=time_dirichlet_fun,
                                          name='dirichlet',
                                          norm=norm,
                                          dataset_size=1,
                                          boundary_sampling_strategy='lower_bound_only',
                                          data_plot_variables=True))

time.add_train_condition(NeumannCondition(neumann_fun=time_neumann_fun,
                                          name='neumann',
                                          norm=norm,
                                          dataset_size=1,
                                          boundary_sampling_strategy='lower_bound_only',
                                          data_plot_variables=True))

def ode_oscillation(u, **input):
    # return laplacian(u, input['time']) + (damping / (2 * mass)) * grad(u, input['time']) + torch.sqrt(input["stiffness"]/mass) * u
    f = laplacian(u, input['time']) + calc_delta(damping, mass) * grad(u, input['time']) + calc_omega_0(stiffness, mass) * u
    return f
# a DiffEqCondition works similar to the boundary condiitions
train_cond = DiffEqCondition(pde=ode_oscillation,
                              name='ode_oscillation',
                              norm=norm,
                              sampling_strategy='random',
                              weight=1.0,
                              dataset_size=50000,
                              data_plot_variables='time')#)('time'))
#%%
setup = Setting(variables=time,
                train_conditions={'ode_oscillation': train_cond},
                val_conditions={},
                solution_dims={'u': 1},
                n_iterations=50)
#%%
solver = PINNModule(model=SimpleFCN(variable_dims=setup.variable_dims,
                                    solution_dims=setup.solution_dims,
                                    depth=4,
                                    width=25),
                    optimizer=torch.optim.Adam,
                    lr=1e-3,
                    # log_plotter=plotter
                    )
#%%
trainer = pl.Trainer(gpus='-1' if torch.cuda.is_available() else None,
                     logger=False,
                     num_sanity_val_steps=0,
                     benchmark=True,
                     check_val_every_n_epoch=50,
                     log_every_n_steps=10,
                     max_epochs=6,
                     checkpoint_callback=False
                     )
#%%
trainer.fit(solver, setup)
#%%

fig = _plot(model=solver.model, solution_name="u", plot_variables=time, points=1500,
            plot_type='line') 
fig.axes[0].set_box_aspect(1/2)

    
    