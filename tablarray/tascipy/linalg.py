#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TablArray compatible wrapping of scipy.linalg

Created on Sun Oct  1 18:08:34 2023

@author: chris
"""

import scipy as _sp

from ..wraps import tawrap_ax2scalar

norm = tawrap_ax2scalar(_sp.linalg.norm, default_view='cell')

from ..wraps import tawrap_mat1_r1

# func(matrix)->scalar; i.e. cdim=0
det = tawrap_mat1_r1(_sp.linalg.det, min_cdim=2, rval_cdim=0)
#cond = tawrap_mat1_r1(_sp.linalg.cond, min_cdim=2, rval_cdim=0)
# func(matrix)->vector; i.e. cdim=1
eigvals = tawrap_mat1_r1(_sp.linalg.eigvals, min_cdim=2, rval_cdim=1)
eigvalsh = tawrap_mat1_r1(_sp.linalg.eigvalsh, min_cdim=2, rval_cdim=1)
# func(matrix)->matrix; i.e. cdim=2
cholesky = tawrap_mat1_r1(_sp.linalg.cholesky, min_cdim=2, rval_cdim=2)
inv = tawrap_mat1_r1(_sp.linalg.inv, min_cdim=2, rval_cdim=2)
#matrix_power = tawrap_mat1_r1(_sp.linalg.matrix_power, min_cdim=2, rval_cdim=2)
# func(array)->scalar
#matrix_rank = tawrap_mat1_r1(_sp.linalg.matrix_rank, min_cdim=1, rval_cdim=0)

from ..wraps import tawrap_mat1_rN

eig = tawrap_mat1_rN(_sp.linalg.eig, 2, 1, 2)
eigh = tawrap_mat1_rN(_sp.linalg.eigh, 2, 1, 2)
#slogdet = tawrap_mat1_rN(_sp.linalg.slogdet, 2, 0, 1)
