#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TODO:
    1. wrapper for linalg.lstsq
    2. "" for linalg.matrx_power

Created on Sat Sep 30 08:08:03 2023

@author: chris
"""

import functools as _functools

from .. import misc


def tawrap_mat1_r1(func, min_cdim, rval_cdim):
    """
    TablArray wrap for numpy-compatible functions (esp. linalg functions)
    of the form::

        func(matrix)->scalar
        func(matrix)->vector
        func(matrix)->matrix
        # generally:
        func(a, *args, **kwargs)->rarray

    Examples::

        from numpy import linalg
        det = tawrap_mat1_r1(linalg.det, min_cdim=2, rval_cdim=0)
        eigvals = tawrap_mat1_r1(linalg.eigvals, min_cdim=2, rval_cdim=1)
        inv = tawrap_mat1_r1(linalg.inv, min_cdim=2, rval_cdim=2)

    Parameters
    ----------
    func : callable
        func(a, *args, **kwargs) to be wrapped
    min_cdim :  int
        Minimum cdim required, will throw exception for a.ts.cdim < min_cdim
    rval_cdim : int
        rarray.cdim==rval_cdim
    """
    @_functools.wraps(func)
    def wrapped_mat1_r1_atc(a, *args, **kwargs):
        if misc.istablarray(a):
            if (a.ts.cdim < min_cdim):
                raise ValueError(
                        '%d-dimensional array given.' % a.ts.cdim
                        + 'Array must be at least %d-dimensional' % min_cdim)
            rarray = func(a.cell, *args, **kwargs)
            rclass = a.__class__
            return misc._rval_once_a_ta(rclass, rarray, rval_cdim, a.view)
            # return rclass(rarray, rval_cdim, view=a.view)
        else:
            return func(a, *args, **kwargs)
    return wrapped_mat1_r1_atc


def tawrap_mat1_rN(func, min_cdim, *rv_cdims):
    """
    TablArray wrap for numpy-compatible functions (esp. linalg functions)
    of the form::

        func(a, *args, **kwargs)->rarray1, [rarray2, .. rarrayN]

    Where cdim of every rarray may be specified in advance.

    Examples::

        from numpy import linalg
        eig = tawrap_mat1_rN(linalg.eig, 2, 1, 2)
        slogdet = tawrap_mat1_rN(linalg.slogdet, 2, 0, 1)

    Parameters
    ----------
    func : callable
        func(a, *args, **kwargs) to be wrapped
    min_cdim :  int
        Minimum cdim required, will throw exception for a.ts.cdim < min_cdim
    *rval_cdims : int, [.., int]
        Specify in advance cdim for each rarray
    """
    N_rval = len(rv_cdims)
    @_functools.wraps(func)
    def wrapped_mat1_rN_atc(a, *args, **kwargs):
        if misc.istablarray(a):
            if (a.ts.cdim < min_cdim):
                raise ValueError(
                        '%d-dimensional array given.' % a.ts.cdim
                        + 'Array must be at least %d-dimensional' % min_cdim)
            rvals = func(a.cell, *args, **kwargs)
            assert len(rvals) == N_rval, '%d rvals, expected %d' % (
                    len(rvals), N_rval)
            rclass = a.__class__
            rval2 = [None] * N_rval
            for i in range(N_rval):
                # print(rv_cdims[i])
                # rval2[i] = rclass(rvals[i], rv_cdims[i], view=a.view)
                rval2[i] = misc._rval_once_a_ta(
                    rclass, rvals[i], rv_cdims[i], a.view)
            return tuple(rval2)
        else:
            return func(a, *args, **kwargs)
    return wrapped_mat1_rN_atc
