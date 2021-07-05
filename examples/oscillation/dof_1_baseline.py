# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 14:35:18 2021

@author: KRD2RNG
"""

import numpy as np
import pandas as pd
from scipy import signal as sg

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
    return(pd.DataFrame(data=y.T, index=t))

def state_space_dof_1(u, x0):
    #  Generate some data

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
    y_stat = np.exp(-delta * t) * (((x0[1] + delta * x0[0]) / omega_d) * np.sin(omega_d * t) + x0[0] * np.cos(omega_d * t))
    return pd.DataFrame(data=y_stat.T, index=t)
# u = np.array([sg.unit_impulse(len(t), idx="mid")])
u = np.array([sg.unit_impulse(len(t), idx=0)])

x0 = np.array([[0], [0]])
y = state_space_dof_1(u, x0)
y.plot()
x0 = np.array([[1], [0]])
y_ana = analytical_dof_1(u, x0)
y_ana.plot()

    
    