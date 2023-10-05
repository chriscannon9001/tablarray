#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 11:26:17 2020

@author: chr
"""

import numpy as _np

from . import misc

def mmul_ta_signature(arg, mxdim):
    """Given a TablArray or np.ndarray, return a TablArray"""
    if misc.istablarray(arg):
        # arg is TablArray type
        array = arg
    # elif (type(arg) is tuple and len(arg) == 2
    #          and isinstance(arg[0], np.ndarray)):
    #        # arg is (array, cdim)
    #        cdim = arg[1]
    #        array = arg[0]
    elif isinstance(arg, _np.ndarray):
        from .ta import TablArray
        # assume either cdim = 2 or ndim
        cdim = max(mxdim, arg.ndim)
        array = TablArray(arg, cdim)
    elif _np.isscalar(arg):
        from .ta import TablArray
        array = TablArray(_np.array(arg), 0)
    else:
        raise ValueError('requires TablArray-like args')
    return array


def _matmul_MV(a, b):
    """matrix-vector multiplication a, b supporting tabular super-structure
    and/or broadcasting

    Parameters
    ----------
    a : array 2 dim or higher
        Where higher dim implies it is an array of 2 dim matrices. Matrix
        multiplication is always aligned to the last 2 dim.
    b : array 1 dim or higher
        Where higher dim implies it is an array of 1 dim vectors. Matrix
        multiplication is always aligned to the last 1 dim.
    """
    # matmul 2d matrix by vector
    subscripts = '...ij,...j->...i'
    # do einsum of a,b using the subscripts
    return _np.einsum(subscripts, a, b)


def _matmul_MM(a, b):
    """matrix-matrix multiplication a, b supporting tabular super-structure
    and/or broadcasting

    Parameters
    ----------
    a : array 2 dim or higher
        Where higher dim implies it is an array of 2 dim matrices. Matrix
        multiplication is always aligned to the last 2 dim.
    b : array 2 dim or higher
        Where higher dim implies it is an array of 2 dim vectors. Matrix
        multiplication is always aligned to the last 2 dim.
    """
    # matmul 2d matrix by 2d matrix
    subscripts = '...ij,...jk->...ik'
    # do einsum of a,b using the subscripts
    return _np.einsum(subscripts, a, b)


def matmul(a, b):
    """Fast matrix multiplication with TablArray compatibility.

    Signatures::

        c = matmul(a: TablArray, b: TablArray)
        c = matmul(a: ndarray, b: ndarray)
        c = matmul(a: ndarray, b: TablArray)

    Where allowed cdim are::

        (a: 1d, b: 1d)  # vector dot product
        (a: 2d, b: 1d)  # multiply matrix by vector
        (a: 2d, b: 2d)  # multiply matrix by matrix
    """
    a = mmul_ta_signature(a, mxdim=2)
    b = mmul_ta_signature(b, mxdim=2)
    rclass = a.__class__
    # setup the subscripts to achieve matmul
    if a.ts.cdim == 2 and b.ts.cdim == 1:
        # matmul 2d matrix by vector
        rarray = _matmul_MV(a.base, b.base)
        return rclass(rarray, 1, a.view)
    elif a.ts.cdim == 2 and b.ts.cdim == 2:
        # matmul 2d matrix by 2d matrix
        rarray = _matmul_MM(a.base, b.base)
        return rclass(rarray, 2, a.view)
    else:
        raise ValueError('matmul works on 1d and/or 2d cells (cdim)')


def dot(a, b):
    """Dot product of two TablArray-like parameters"""
    a = mmul_ta_signature(a, mxdim=1)
    b = mmul_ta_signature(b, mxdim=1)
    # print(a)
    # print(b)
    rclass = a.__class__
    if a.ts.cdim == 1 and b.ts.cdim == 1:
        # dot product of vectors
        rarray = _np.einsum('...i,...i->...', a.base, b.base)
        return rclass(rarray, 0, a.view)
    else:
        raise ValueError('dot works on 1d cells (cdim)')


def cross(a, b):
    """Cross product of two TablArray-like parameters"""
    a = mmul_ta_signature(a, mxdim=1)
    b = mmul_ta_signature(b, mxdim=1)
    rclass = a.__class__
    if a.ts.cdim == 1 and b.ts.cdim == 1:
        # cross product of vectors
        rarray = _np.cross(a.base, b.base)
        return rclass(rarray, 1, a.view)
    else:
        raise ValueError('cross works on 1d cells (cdim=1')


def cummatmul(a, axis):
    """Given an ATC object, return the cumulative matmul
    (matrix multiplication) of the cells along the given tabular axis
    Always implies cumulative axis is one of ts.tshape."""
    a = mmul_ta_signature(a, mxdim=2)
    # get a blank ATC for the result
    rarray = _np.zeros(a.base.shape, dtype=a.base.dtype)
    rclass = a.__class__
    new_atc = rclass(rarray, a.ts.cdim, a.view)
    # get a table view of the result ATC
    new_table = new_atc.table
    # table view of the atc input
    table = a.table
    # prep for loop
    N_iter = table.shape[axis]
    slice0 = [slice(None)] * table.ndim
    slice1 = [slice(None)] * table.ndim
    # loop over table across this axis
    slice0[axis] = 0
    new_table[slice0] = table[slice0]
    for i in range(1, N_iter):
        slice0[axis] = i - 1
        slice1[axis] = i
        new_table[slice1] = matmul(table[slice1], new_table[slice0])
    return new_table
