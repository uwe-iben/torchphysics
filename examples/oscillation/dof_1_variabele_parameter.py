# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:00:02 2021

@author: KRD2RNG
"""

import os
import numpy as np
import pandas as pd
from scipy import signal as sg
import matplotlib.pyplot as plt

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
from torchphysics.utils.plot import _plot, _create_domain


os.environ["CUDA_VISIBLE_DEVICES"] = "0" # select GPUs to use

#pl.seed_everything(43) # set a global seed
torch.cuda.is_available()
# matplotlib.style.use('default')
def calc_mass(c):
    return c / (2 * np.pi) **2

n_stiffness = 100
stiffness_array = np.linspace(1e3, 1e4, n_stiffness) #1e4
damping = 100
mass = calc_mass(stiffness_array) # 2e2

SampleFrequency = 1024
t_end = 1
t = np.linspace(0, t_end, t_end * SampleFrequency)

eval_points = 32
#%%

def calc_omega_0(c, m):
    return np.sqrt(c/m)
def calc_delta(b, m):
    return b/(2*m)
#%%
def analytical_dof_1(x0):
    delta = calc_delta(damping, mass)
    omega_0 = calc_omega_0(stiffness_array, mass)
    omega_d = np.sqrt(omega_0**2 - delta**2)
    y = np.exp(-delta[np.newaxis,:] * t[:,np.newaxis]) * \
        (((x0[1] + delta[np.newaxis,:] * x0[0]) / omega_d[np.newaxis,:]) * np.sin(omega_d[np.newaxis,:] * t[:,np.newaxis]) \
         + x0[0] * np.cos(omega_d[np.newaxis,:] * t[:,np.newaxis]))
    return pd.DataFrame(data=y, index=t, columns= np.ceil(stiffness_array))

#%% reference solution
dirac = np.array([sg.unit_impulse(len(t), idx=0)])
x0 = np.array([[1], [0]])
y = analytical_dof_1(x0)
y.plot(legend=False)
#%% PINN approach
# u_tt + 2*delta * u_t + omega**2 * u = f(t)

norm = torch.nn.MSELoss() #  #L1Loss

time = Variable(name='time',
              order=1,
              domain=Interval(low_bound=0,
                              up_bound=t_end),
              train_conditions={},
              val_conditions={})
stiffness = Variable(name='stiffness',
              order=1,
              domain=Interval(low_bound=1e3,
                              up_bound=1e4),
              train_conditions={},
              val_conditions={})

def time_dirichlet_fun(time): # (time)
    return np.ones_like(time)

def time_neumann_fun(time):
    return np.zeros_like(time)

time.add_train_condition(DirichletCondition(dirichlet_fun=time_dirichlet_fun,
                                          name='dirichlet',
                                          norm=norm,
                                          weight = 200,
                                          dataset_size=1,
                                          boundary_sampling_strategy='lower_bound_only',
                                          data_plot_variables=True))

time.add_train_condition(NeumannCondition(neumann_fun=time_neumann_fun,
                                          name='neumann',
                                          norm=norm,
                                          weight=50,
                                          dataset_size=1,
                                          boundary_sampling_strategy='lower_bound_only',
                                          data_plot_variables=True))

def ode_oscillation(u, time, stiffness):
    m = calc_mass(stiffness)
    f = laplacian(u, time) + (damping / m) * grad(u, time) + (stiffness / m) * u
    # f = laplacian(u, time) + 2* grad(u, time) + stiffness * u
    return f

train_cond = DiffEqCondition(pde=ode_oscillation,
                              name='ode_oscillation',
                              norm=norm,
                              sampling_strategy='grid',
                              weight=1,
                              dataset_size=eval_points,
                              data_plot_variables=("time"))#)('time'))True
#%%
setup = Setting(variables=(time,stiffness), 
                train_conditions={'ode_oscillation': train_cond},
                val_conditions={},
                solution_dims={'u': 1},
                n_iterations=50)
#%%
solver = PINNModule(model=SimpleFCN(variable_dims=setup.variable_dims,
                                    solution_dims=setup.solution_dims,
                                    depth=4,
                                    width=15,
                                    activation_func=torch.nn.Mish()),
                    optimizer=torch.optim.Adam, # Adam
                    lr=1e-3,
                    # log_plotter=plotter
                    )
#%%
trainer = pl.Trainer(gpus='-1' if torch.cuda.is_available() else None,
                      # logger=False,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      check_val_every_n_epoch=2,
                      log_every_n_steps=10,
                      max_epochs=12,
                      checkpoint_callback=False
                      )
#%%

from torchphysics.problem.datacreator import InnerDataCreator
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
dc = InnerDataCreator(variables={'x': x, 't': t, 'D': D},
                      dataset_size=[2000, 400, 1], 
                      # Here we use different number of points for each variable
                      # x has 100 points, t = 20 and D only 1 (constant) value.
                      # The number of total points are all different combinations:
                      # -> 2000 different points
                      sampling_strategy='grid')
input_dic = dc.get_data() # create the data
# Change the D (if you want):
input_dic['D'] = 10 * np.ones((len(input_dic['D']), 1))
# cast everything to tensors:
for name in input_dic:
    input_dic[name] = torch.FloatTensor(input_dic[name]).to(device)
# Evaluate
solver = solver.to(device)
start = time.time()
pred = solver.forward(input_dic) 
#%%
trainer.fit(solver, setup)
# #%%
device = "cpu"

#%%
stiff_value = 1000
point_num = 256
input_dic = {time.name: torch.tensor(time.domain.grid_for_plots(point_num), device=device),
             stiffness.name: stiff_value * torch.ones((point_num, 1), device=device),
             }  # for the stiffnes we want one fixed value, not a grid

torch.cat([v for v in input_dic.values()], dim=1)
#y["PINN"] = solver.model.forward(input_dic)["u"].detach().numpy()
#y.plot()

# Plot surface with one axis time the other stiffness:
_plot(solver.model, solution_name="u", plot_variables = [time, stiffness], points=256, angle=[30, 30],
      dic_for_other_variables=None, all_variables=None, device='cpu')

# Plot line over time, need to set the used stiffness in dic_for_other_variables: 
fig = _plot(model=solver.model, solution_name="u", plot_variables=time, points=256,
            plot_type='line', dic_for_other_variables={'stiffness': 3}) #<- here set your stiffness-value