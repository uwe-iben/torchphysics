# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:23:17 2021

@author: inu2sh
"""
import pandas as pd
import numpy as np

data1 = pd.read_csv("data_for_x=[1.0,0.5].csv",header=None,index_col=0)
data1.index.name = 'Time'
data1.columns.name = 'u_PINN'
#data1.plot()

data2 = pd.read_csv("Burger_IAG/Re50_baseline_RecordPoints_1_05.csv")

data2 = data2.set_index("Time")
data1_inter = pd.DataFrame(np.interp(data2.index,data1.index,data1.to_numpy().flatten()),
                           index = data2.index,columns=data1.columns)
data = pd.concat((data1_inter,data2['u']),axis=1)
data.columns = ['PINN','DGM']
data.plot(grid=True)

