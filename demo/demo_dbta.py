#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 13:28:00 2020

@author: chris
"""

from matplotlib import pyplot as plt
import numpy as np

import tablarray as tarr


abcd = tarr.TablArray([[[[1, 10], [0, 1]]],  # air
                       [[[1, 4], [0, 1]]],  # laser crystal
                       [[[1, 20], [0, 1]]],  # air
                       [[[1, 20], [0, 1]]],  # air
                       [[[1, 10], [0, 1]]],  # air
                       [[[1, 0], [0, 1]]],  # plano mirror
                       [[[1, 10], [0, 1]]],  # air
                       [[[1, 20], [0, 1]]],  # air
                       [[[1, 20], [0, 1]]],  # air
                       [[[1, 4], [0, 1]]],  # laser crystal
                       [[[1, 10], [0, 1]]],  # air
                       [[[1, 0], [-2/100, 1]]]], cdim=2)  # curved mirror

db1 = tarr.dbTA(abcd=abcd)

# cumulative abcd matrix
db1['cumabcd'] = tarr.cummatmul(db1['abcd'], axis=0)
# get the last cumulative abcd matrix, then solve the eigens
eig_val, eig_vect = tarr.linalg.eig(db1['cumabcd'].bcast[-1, :])
# pick one of the eigenvectors (the first one)
qv0 = eig_vect.cell[:, 1]
db1['qv0'] = qv0 / qv0[1]
qv = (tarr.matmul(db1['cumabcd'], db1['qv0'])).cell
db1['qv'] = qv / qv[1]
inv_q = 1/(db1['qv'].cell[0])
db1['curve'] = inv_q.real
db1['w'] = tarr.sqrt( (-0.001064/np.pi) / inv_q.imag)

opl_i = db1['abcd'].cell[0, 1]
db1['opl'] = tarr.cumsum(opl_i.table, axis=0).cell

plt.plot(db1['opl'].table[:,0].array, db1['w'].table[:,0].array)
plt.xlabel('OPL (mm)')
plt.ylabel('w_00 (mm)')
