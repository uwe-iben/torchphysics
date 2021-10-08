#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import Image


# In[2]:


import torch
import os
import numpy as np
import pytorch_lightning as pl
from timeit import default_timer as timer

from torchphysics.problem import Variable
from torchphysics.setting import Setting
from torchphysics.problem.domain import (Rectangle,
                                           Interval,
                                           Circle)
from torchphysics.problem.condition import (DirichletCondition,
                                              DiffEqCondition,
                                              DataCondition)
from torchphysics.models.fcn import SimpleFCN
from torchphysics import PINNModule
from torchphysics.utils import laplacian, jac, convective
from torchphysics.utils.fdm import FDM, create_validation_data
from torchphysics.utils.plot import Plotter
from torchphysics.utils.evaluation import (get_min_max_inside,
                                             get_min_max_boundary)
from torchphysics.setting import Setting

os.environ["CUDA_VISIBLE_DEVICES"] = "0" # select GPUs to use

#pl.seed_everything(43) # set a global seed
torch.cuda.is_available()


# In[3]:


w, h = 3, 1
t0, tend = 0, 25
Re = 50

u = 'u'


# In[ ]:


from torchphysics.problem.domain.domain_operations import Cut

R = Rectangle(corner_dl=[0, 0], corner_dr=[w, 0], corner_tl=[0, h])
#C1 = Circle([w/3,0], h/3)
C2 = Rectangle(corner_dl=[1, 0], corner_dr=[1.2, 0], corner_tl=[1, 0.4])
domain=Cut(R,C2)
#domain=Cut(domain,C2)


# Darstellung der Geometrie mit Punkten auf dem Rand (blau) und Punkte im Inneren (rot)
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 5*w, 5*h #Skalierung mit der Grösse
b = domain.sample_boundary(300, type='grid')
k = domain.sample_inside(400, type='grid')
plt.scatter(b[:, 0], b[:, 1])
plt.scatter(k[:, 0], k[:, 1])
plt.grid(True)
plt.savefig('geometrie.png', dpi=300)


x = Variable(name='x',
             order=2,
             domain=domain,
             train_conditions={},
             val_conditions={})
t = Variable(name='t',
             order=1,
             domain=Interval(low_bound=0,
                             up_bound=tend),
             train_conditions={},
             val_conditions={})


# In[ ]:


norm = torch.nn.MSELoss()
# at start: erverything 0
def t_dirichlet_fun(**input):
    return np.zeros_like(input['x'])
N_bc = 1000
t.add_train_condition(DirichletCondition(dirichlet_fun=t_dirichlet_fun,
                                         solution_name=u,
                                         whole_batch=True,
                                         name='dirichlet',
                                         norm=norm,
                                         dataset_size=N_bc,
                                         boundary_sampling_strategy='lower_bound_only',
                                         data_plot_variables=('x','t')))
# at boundary: flow dependent on time (left into the domain, right out of the domain)
# y component always zero
# at points where this function returns None, no boundary condition will be applied
def x_dirichlet_fun(x, t):
    out = np.zeros(2)
    if np.isclose(x[0], 0):
        out[0] = 5*x[1]*(h-x[1])*(1-np.exp(-t))
        return out
    if not np.isclose(x[0], w):
        return out

x.add_train_condition(DirichletCondition(dirichlet_fun=x_dirichlet_fun,
                                         solution_name=u,
                                         whole_batch=False, # this enables us to use point-wise defined dirichlet_fun
                                         name='dirichlet',
                                         sampling_strategy='grid',
                                         boundary_sampling_strategy='grid',
                                         norm=norm,
                                         weight=1.0,
                                         dataset_size={'x': 600, 't': 25},
                                         data_plot_variables=('x','t')))


# In[ ]:


from IPython.display import Image, Math, Latex
from IPython.core.display import HTML 
display(Math(r'u_t + (u \cdot \nabla)u = \frac{1}{Re} \Delta u \,, u=u(t,x,y) \quad (x,y) \in \Omega'))


# In[ ]:


def pde(u, x, t):
    jac_t = jac(u, t).squeeze(dim=2) # time derivative of first and second output
    conv = convective(u, x, u) # convection term
    l_1 = laplacian(u[:, 0], x)
    l_2 = laplacian(u[:, 1], x)
    laplace_vec = torch.cat((l_1, l_2), dim=1) # put laplace in one vector
    return jac_t + conv - 1/Re * laplace_vec

train_cond = DiffEqCondition(pde=pde,
                             name='pde',
                             norm=norm,
                             weight=1.0,
                             dataset_size= {'x': 2000, 't': 25},
                             sampling_strategy = 'grid',
                             data_plot_variables=('x','t'))


# In[ ]:


setup = Setting(variables=(x, t),
                train_conditions={'pde': train_cond},
                val_conditions={},
                solution_dims={'u': 2},
                n_iterations=800)


# In[ ]:


solver = PINNModule(model=SimpleFCN(variable_dims=setup.variable_dims,
                                    solution_dims=setup.solution_dims,
                                    normalization_dict=setup.normalization_dict,
                                    depth=4,
                                    width=20),
                    optimizer=torch.optim.Adam,
                    lr=1e-2,
                    #log_plotter=plotter
                    )


# In[ ]:


trainer = pl.Trainer(gpus='-1' if torch.cuda.is_available() else None,
                     logger=False,
                     num_sanity_val_steps=0,
                     benchmark=True,
                     check_val_every_n_epoch=50,
                     log_every_n_steps=10,
                     max_epochs=2,
                     checkpoint_callback=False
                     )

trainer.fit(solver, setup)


# In[ ]:


solver.optimizer = torch.optim.LBFGS
solver.lr = 0.1

trainer = pl.Trainer(gpus='-1' if torch.cuda.is_available() else None,
                     logger=False,
                     num_sanity_val_steps=0,
                     benchmark=True,
                     check_val_every_n_epoch=50,
                     log_every_n_steps=10,
                     max_epochs=2,
                     checkpoint_callback=False
                     )

trainer.fit(solver, setup)


# In[ ]:


from torchphysics.utils.plot import _plot
zeit = tend
fig = _plot(model=solver.model, solution_name=u, plot_variables=x, points=2000,
            dic_for_other_variables={'t' : zeit}, plot_type='contour_surface') # = 'quiver_2D' for vectors
fig.axes[0].set_box_aspect(h/w)
plt.savefig(f'u_{Re}.png')


# In[ ]:


#%load_ext autoreload
#%autoreload 2
from torchphysics.utils.animation import animation
fig, ani = animation(model=solver.model, solution_name=u, plot_variable=x, domain_points=400, 
                animation_variable=t, frame_number=100, ani_type='contour_surface') # = 'quiver_2D' for vectors
fig.axes[0].set_box_aspect(h/w)
ani.save(f'flow_{Re}.gif')


# In[ ]:


class out_abs(torch.nn.Module):
    def forward(self,input_dic):
        out = solver.model(input_dic)
        return {'u': torch.norm(out['u'],dim=1,keepdim=True)}
fig = _plot(model=out_abs(), solution_name=u, plot_variables=t, points=1000,
            dic_for_other_variables={'x' : [1.0,0.5]}, plot_output_entries = 0) # 0 erste Komponente; 1 zweite Komponente
plt.savefig(f'norm_u0_{Re}.png')

time_value = fig.axes[0].get_children()[0].get_xdata()
func_value = fig.axes[0].get_children()[0].get_ydata()
append_values = np.column_stack((time_value, func_value))
np.savetxt("data_for_x=[1.0,0.5]_{Re}.csv", append_values, delimiter=",")


# In[ ]:


class out_abs(torch.nn.Module):
    def forward(self,input_dic):
        out = solver.model(input_dic)
        return {'u': torch.norm(out['u'],dim=1,keepdim=True)}
fig = _plot(model=out_abs(), solution_name=u, plot_variables=t, points=1000,
            dic_for_other_variables={'x' : [2.0,0.2]}, plot_output_entries = 0) # 0 erste Komponente; 1 zweite Komponente
plt.savefig(f'norm_u1_{Re}.png')


# In[ ]:


time_value = fig.axes[0].get_children()[0].get_xdata()
func_value = fig.axes[0].get_children()[0].get_ydata()
append_values = np.column_stack((time_value, func_value))
np.savetxt("data_for_x=[2.0,0.2]_{Re}.csv", append_values, delimiter=",")


# In[ ]:





# ### 
