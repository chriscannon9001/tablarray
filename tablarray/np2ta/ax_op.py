#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:10:44 2020

@author: chris
"""

import functools
import numpy as np

from .. import misc


def tawrap_ax2scalar(func, default_view=None):
    """
    TablArray wrap for numpy-compatible functions which have unary operands
    where one or more axes transform to a scalar (axis -> scalar)

    After wrap, the function will allow TablArray-like inputs including
    np.ndarray, or scalar.
    """
    _doc_prepend = ("    **TablArray compatible** %s, where axis aligns w.r.t. view\n\n" % func.__name__
                    + "    view: 'cell', 'table', or None (default=%s)\n" % default_view
                    + "        overrides a.view if istablarray(a)\n"
                    + "    -----\n\n")
    @functools.wraps(func)
    def wrap_ax_bcast(a, axis=None, view=default_view, **kwargs):
        # what should happen if user throws keepdims?
        # or instead of looking for 'keepdims' in kwargs
        #   should we verify dimensions change in rval?
        if misc.istablarray(a):
            if type(view) is str:
                # get view of a (same as a.cell or a.table)
                # not a.setview which alters input parameter
                a = a.__getattribute__(view)
            if axis is None:
                axis = a._viewdims
                # cdim = a_.viewcdim
                delta_cdim = a.ts.cdim - a._viewcdim
            else:
                # _viewdims[axis] translates axis w.r.t. a.base
                if type(axis) is tuple:
                    axis = tuple(np.array(a._viewdims)[list(axis)])
                else:
                    axis = a._viewdims[axis]
                if a._cellular:
                    # if one of the cellular dims collapses to a scalar,
                    # then cdims will decrease
                    if np.isscalar(axis):
                        # cdim = a.ts.cdim - 1
                        delta_cdim = 1
                    else:
                        delta_cdim = len(axis)
                        # cdim = a.ts.cdim - len(axis)
                else:
                    # if one of the tabular dims collapses to a scalar,
                    # the number of cdims is unchanged, easy case
                    delta_cdim = 0
                    # cdim = a.ts.cdim
            rarray = func(a.base, axis=axis, **kwargs)
            rclass = a.__class__  # probably TablArray
            # there are cases where the ndim doesn't actually reduce (e.g. keepdims=True in kwargs)
            cdim = a.ts.cdim - delta_cdim if (np.ndim(rarray) < np.ndim(a.base)) else 0
            # once a TablArray, usually a TablArray
            return misc._rval_once_a_ta(rclass, rarray, cdim, a.view)
        else:
            # just passthrough
            return func(a, axis=axis, **kwargs)
    wrap_ax_bcast.__doc__ = (
        _doc_prepend + wrap_ax_bcast.__doc__)
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
