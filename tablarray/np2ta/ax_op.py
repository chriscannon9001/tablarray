#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:10:44 2020

@author: chris
"""

import functools
import numpy as np

from .. import misc


def tawrap_ax2scalar(func):
    """
    TablArray wrap for numpy-compatible functions which have unary operands
    where one or more axes transform to a scalar (axis -> scalar)

    After wrap, the function will allow TablArray-like inputs including
    np.ndarray, or scalar.
    """
    @functools.wraps(func)
    def wrap_ax_bcast(a, axis=None, **kwargs):
        if misc.istablarray(a):
            if axis is None:
                axis = a._viewdims
                cdim = a._viewcdim
            else:
                if type(axis) is tuple:
                    axis = tuple(np.array(a._viewdims)[list(axis)])
                else:
                    axis = a._viewdims[axis]
                if a._cellular:
                    # if one of the cellular dims collapses to a scalar,
                    # then cdims will decrease
                    if np.isscalar(axis):
                        cdim = a.ts.cdim - 1
                    else:
                        cdim = a.ts.cdim - len(axis)
                else:
                    # if one of the tabular dims collapses to a scalar,
                    # the number of cdims is unchanged
                    cdim = a.ts.cdim
            rarray = func(a.base, axis=axis, **kwargs)
            rclass = a.__class__  # probably TablArray
            # once a TablArray, usually a TablArray
            return misc._rval_once_a_ta(rclass, rarray, cdim, a.view)
        else:
            # just passthrough
            return func(a, axis=axis, **kwargs)
    return wrap_ax_bcast


# these are also available as methods
all = tawrap_ax2scalar(np.all)
any = tawrap_ax2scalar(np.any)
argmax = tawrap_ax2scalar(np.argmax)
argmin = tawrap_ax2scalar(np.argmin)
max = tawrap_ax2scalar(np.max)
mean = tawrap_ax2scalar(np.mean)
min = tawrap_ax2scalar(np.min)
prod = tawrap_ax2scalar(np.prod)
std = tawrap_ax2scalar(np.std)
sum = tawrap_ax2scalar(np.sum)

# these are only available here - not as methods
amax = tawrap_ax2scalar(np.amax)
amin = tawrap_ax2scalar(np.amin)
median = tawrap_ax2scalar(np.median)
nanargmax = tawrap_ax2scalar(np.nanargmax)
nanargmin = tawrap_ax2scalar(np.nanargmin)
nanmax = tawrap_ax2scalar(np.nanmax)
nanmean = tawrap_ax2scalar(np.nanmean)
nanmedian = tawrap_ax2scalar(np.nanmedian)
nanmin = tawrap_ax2scalar(np.nanmin)
nanprod = tawrap_ax2scalar(np.nanprod)
nansum = tawrap_ax2scalar(np.nansum)
nanstd = tawrap_ax2scalar(np.nanstd)
nanvar = tawrap_ax2scalar(np.nanvar)
var = tawrap_ax2scalar(np.var)
