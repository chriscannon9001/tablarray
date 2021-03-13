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
@ta.solver_spec('cumabcd(abcd){table} --> (cumabcd)')
def cumabcd(abcd):
    """given ABCD, accumulate cumabcd"""
    return ta.cummatmul(abcd, axis=0)


@ta.solver_spec('qvect0(cumabcd){bcast} --> (qvect0){table}')
def qvect0(cumabcd):
    """given cumabcd, return qvect0"""
    _, eig_vect = ta.linalg.eig(cumabcd[-1])
    qvect0 = eig_vect.cell[:, 1]
    return qvect0


@ta.solver_spec('qvect(cumabcd, qvect0){table} --> (qvect)')
def qvect(cumabcd, qvect0):
    return ta.matmul(cumabcd, qvect0)


@ta.solver_spec(inargs=['qvect', 'v'], inkwargs=[], outargs=['w', 'curv'],
                inview='cell')
def qvect_decompose(qvect, v):
    """Given q return w, curv=1/R
    {where 1/q = 1/R - j * lambda / (pi*w^2)}"""
    q_scalar = qvect[0]/qvect[1]
    w = ta.sqrt((10/(np.pi*v)) * (-1/ta.imag(1/q_scalar)))
    curv = ta.real(1/q_scalar)
    return w, curv


@ta.solver_spec(inargs=['abcd'], inkwargs=[], outargs=['opl'])
def opl(abcd):
    opl_i = abcd.cell[0, 1]
    opl = ta.cumsum(opl_i.table, axis=0).cell
    return opl


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

set1 = ta.TablaSet(abcd=abcd, v=1e4/1.06)

solve1 = ta.TablaSolver(set1, [cumabcd, qvect0, qvect, qvect_decompose, opl])
solve1(maxiter=2)

plt.plot(set1['opl', :, 0].base, set1['w', :, 0].base)
plt.xlabel('OPL (mm)')
plt.ylabel('w_00 (mm)')
with ta.printoptions(precision=3, threshold=10):
    print(set1)
