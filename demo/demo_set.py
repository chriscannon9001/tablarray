#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 13:28:00 2020

@author: chris
"""

from matplotlib import pyplot as plt
import numpy as np

import tablarray as ta


# demo writing physics lib code with added TablaSet intel
@ta.tablaset_args(setargs=('qvect', 'v'))
def qvect_decompose(qvect, v):
    """Given q return w, curv=1/R
    {where 1/q = 1/R - j * lambda / (pi*w^2)}"""
    q_scalar = qvect[0]/qvect[1]
    w = ta.sqrt((10/(np.pi*v)) * (-1/ta.imag(1/q_scalar)))
    curv = ta.real(1/q_scalar)
    return w, curv


# this is a paraxial round trip through a resonator
abcd = ta.TablArray([[[[1, 0], [-2/100, 1]]],  # curved end mirror
                     [[[1, 10], [0, 1]]],  # air
                     [[[1, 4], [0, 1]]],  # laser crystal
                     [[[1, 20], [0, 1]]],  # air
                     [[[1, 20], [0, 1]]],  # air
                     [[[1, 10], [0, 1]]],  # air
                     [[[1, 0], [0, 1]]],  # plano end mirror
                     [[[1, 10], [0, 1]]],  # air
                     [[[1, 20], [0, 1]]],  # air
                     [[[1, 20], [0, 1]]],  # air
                     [[[1, 4], [0, 1]]],  # laser crystal
                     [[[1, 10], [0, 1]]]], cdim=2)  # air

set1 = ta.TablaSet(abcd=abcd)

# cumulative abcd matrix
set1['cumabcd'] = ta.cummatmul(set1['abcd'], axis=0)
# get the last cumulative abcd matrix, then solve the eigens
eig_val, eig_vect = ta.linalg.eig(set1['cumabcd'].bcast[-1, :])
# pick one of the eigenvectors (the first one)
set1['qvect0'] = eig_vect.cell[:, 1]
set1['qvect'] = ta.matmul(set1['cumabcd'], set1['qvect0'])
set1['w'], set1['curv'] = qvect_decompose(set1, v=10000/1.064)

opl_i = set1['abcd'].cell[0, 1]
set1['opl'] = ta.cumsum(opl_i.table, axis=0).cell

plt.plot(set1['opl', :, 0].base, set1['w', :, 0].base)
plt.xlabel('OPL (mm)')
plt.ylabel('w_00 (mm)')
with ta.printoptions(precision=3, threshold=10):
    print(set1)
