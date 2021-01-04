#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:41:01 2020

@author: chris
"""

import functools
import numpy as np

from .. import _io


def _axial2_broadcast(func):
    """ACT compatibility for unary operands where number of dimensions
    does not change"""
    @functools.wraps(func)
    def wrapped_cum_atc(a, axis=None, **kwargs):
        if _io.quackslike_Tablarray(a):
            axis = a._viewdims[axis]
            rarray = func(a.base, axis=axis, **kwargs)
            rclass = a.__class__
            # once a TablArray, usually a TablArray
            return _io.rval_once_a_ta(rclass, rarray, a.ts.cdim, a.view)
        else:
            # pass through to numpy
            return func(a, axis=axis, **kwargs)
    return wrapped_cum_atc


cumprod = _axial2_broadcast(np.cumprod)
cumsum = _axial2_broadcast(np.cumsum)
nancumprod = _axial2_broadcast(np.nancumprod)
nancumsum = _axial2_broadcast(np.nancumsum)
