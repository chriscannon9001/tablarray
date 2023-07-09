#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:41:01 2020

@author: chris
"""

import functools
import numpy as np

from .. import misc


def _axial2_broadcast(func, default_view=None):
    """ACT compatibility for unary operands where number of dimensions
    does not change"""
    _doc_prepend = ("    **TablArray compatible** %s, where axis aligns w.r.t. view\n\n" % func.__name__
                    + "    view: 'cell', 'table', or None (default=%s)\n" % default_view
                    + "        overrides a.view if istablarray(a)\n"
                    + "    -----\n\n")
    @functools.wraps(func)
    def wrapped_ax2_bcast(a, axis=None, view=default_view, **kwargs):
        if misc.istablarray(a):
            if type(view) is str:
                # get view of a (same as a.cell or a.table)
                # not a.setview which alters input parameter
                a = a.__getattribute__(view)
            axis = a._viewdims[axis]
            rarray = func(a.base, axis=axis, **kwargs)
            rclass = a.__class__
            # once a TablArray, usually a TablArray
            return misc._rval_once_a_ta(rclass, rarray, a.ts.cdim, a.view)
        else:
            # pass through to numpy
            return func(a, axis=axis, **kwargs)
    wrapped_ax2_bcast.__doc__ = (
        _doc_prepend + wrapped_ax2_bcast.__doc__)
    return wrapped_ax2_bcast


cumprod = _axial2_broadcast(np.cumprod)
cumsum = _axial2_broadcast(np.cumsum)
nancumprod = _axial2_broadcast(np.nancumprod)
nancumsum = _axial2_broadcast(np.nancumsum)
