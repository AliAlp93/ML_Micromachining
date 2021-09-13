# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 15:08:25 2021

@author: alial
"""

import torch
import torch.nn.functional as ff
a = torch.arange(9, dtype= torch.float)
b = a.reshape((3, 3))
#torch.norm(a)
#torch.norm(b)

b_normalized=ff.normalize(b, p=2, dim=1, eps=1e-12, out=None)
#a_normalized=ff.normalize(a, p=1, dim=1, eps=1e-12, out=None)
a_normalized=torch.nn.BatchNorm1d
#%%
torch.norm(a, float('inf'))
torch.norm(b, float('inf'))
#%%
c = torch.tensor([[ 1, 2, 3],[-1, 1, 4]] , dtype= torch.float)
torch.norm(c, dim=0)
#%%
torch.norm(c, dim=1)
tensor([3.7417, 4.2426])
torch.norm(c, p=1, dim=1)
tensor([6., 6.])
d = torch.arange(8, dtype= torch.float).reshape(2,2,2)
torch.norm(d, dim=(1,2))
tensor([ 3.7417, 11.2250])
torch.norm(d[0, :, :]), torch.norm(d[1, :, :])
(tensor(3.7417), tensor(11.2250))