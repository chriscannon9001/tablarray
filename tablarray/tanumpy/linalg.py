#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TablArray compatible wrapping of numpy.linalg

Created on Sun Oct  1 17:00:49 2023

@author: chris
"""

import numpy as _np

from ..wraps import tawrap_ax2scalar

norm = tawrap_ax2scalar(_np.linalg.norm, default_view='cell')

from ..wraps import tawrap_mat1_r1

# func(matrix)->scalar; i.e. cdim=0
det = tawrap_mat1_r1(_np.linalg.det, min_cdim=2, rval_cdim=0)
cond = tawrap_mat1_r1(_np.linalg.cond, min_cdim=2, rval_cdim=0)
# func(matrix)->vector; i.e. cdim=1
eigvals = tawrap_mat1_r1(_np.linalg.eigvals, min_cdim=2, rval_cdim=1)
eigvalsh = tawrap_mat1_r1(_np.linalg.eigvalsh, min_cdim=2, rval_cdim=1)
# func(matrix)->matrix; i.e. cdim=2
cholesky = tawrap_mat1_r1(_np.linalg.cholesky, min_cdim=2, rval_cdim=2)
inv = tawrap_mat1_r1(_np.linalg.inv, min_cdim=2, rval_cdim=2)
matrix_power = tawrap_mat1_r1(_np.linalg.matrix_power, min_cdim=2, rval_cdim=2)
# func(array)->scalar
matrix_rank = tawrap_mat1_r1(_np.linalg.matrix_rank, min_cdim=1, rval_cdim=0)

from ..wraps import tawrap_mat1_rN

eig = tawrap_mat1_rN(_np.linalg.eig, 2, 1, 2)
eigh = tawrap_mat1_rN(_np.linalg.eigh, 2, 1, 2)
slogdet = tawrap_mat1_rN(_np.linalg.slogdet, 2, 0, 1)
