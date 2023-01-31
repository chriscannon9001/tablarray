#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
My idea is to have a comparable lib to numpy.add, .sum, etc.
But it will work regardless of whether the x1, x2, ... are numpy or TablArray.

Created on Sun May 17 18:17:59 2020

@author: chris
"""

import functools
import numpy as np

from .. import misc
from ..tashape import taShape


def _cast_other_type(other, ta):
    """when a TablArray and other type are cast in a binary operator, make sure
    other is np.ndarray compatible, also maybe reorient for broadcasting
    if the TablArray is in a tabular view"""
    o_type = type(other)
    other = np.array(other) if (o_type is list or o_type is tuple) else other
    if ta._tabular and not np.isscalar(other):
        # if my view is tabular I need to promote to tabular shape
        o_shape2 = tuple(list(other.shape) + [1] * ta.ts.cdim)
        other = other.reshape(o_shape2)
    return other


def tawrap_binarybroadcast(func, dtype=None):
    """
    TablArray wrap for numpy-compatible functions which have binary input
    and need TablArray broadcasting adaptation.

    After wrap, the function will allow TablArray-like inputs including
    np.ndarray, or scalar.
    """
    @functools.wraps(func)
    def wrap_bin_bcast(a, b, *args, **kwargs):
        """depending on the types of a and b, find a suitable broadcasting"""
        if misc.istablarray(a) and misc.istablarray(b):
            # if both are TablArray, then use tablarray broadcasting
            cdim, bc = a.ts.combine(b.ts)
            rarray = bc.calc_function(func, a.base, b.base, *args,
                                      dtype=dtype, **kwargs)
            rclass = a.__class__
            view = a.view
        elif misc.istablarray(a):
            b = _cast_other_type(b, a)
            # if only one is TablArray, then use numpy array broadcast
            rarray = func(a.base, b, *args, **kwargs)
            rclass = a.__class__
            # assume the result has the same cdim as a.ts.cdim
            cdim = a.ts.cdim
            view = a.view
        elif misc.istablarray(b):
            a = _cast_other_type(a, b)
            rarray = func(a, b.base, *args, **kwargs)
            rclass = b.__class__
            cdim = b.ts.cdim
            view = b.view
        else:
            # if neither operand is TablArray, just fall back on numpy
            return func(a, b, *args, **kwargs)
        # once a TablArray, always a TablArray
        return misc._rval_once_a_ta(rclass, rarray, cdim, view)
    wrap_bin_bcast.__doc__ = (
        "**TablArray compatible** %s\n\n" % func.__name__
        + wrap_bin_bcast.__doc__)
    return wrap_bin_bcast


def tawrap_multiop_bcast(func, arg_ctl, dtype=None):
    '''
    TablArray wrap for numpy-compatible functions which have any number of
    input operands in need of TablArray broadcasting adaptation. This does
    require the wrapped function to have a single array-like return.

    After wrap, the function will allow TablArray-like inputs including
    np.ndarray, or scalar.

    Input
    -----
    arg_ctl : list of bool
        ags expected to be TablArray-like, e.g. [True, False, True]. TablArray
        args will only be considered if they correspond to a True flag.
    '''
    Narg0 = len(arg_ctl)
    @functools.wraps(func)
    def wrap_multi_bcast(*args, **kwargs):
        # get map of important arg types
        arg_is_ta = np.zeros(Narg0, dtype=bool)
        for i in range(min(len(args), Narg0)):
            # if the arg_ctl masked off the argument here, ignore it
            if arg_ctl[i]:
                arg_is_ta[i] = misc.istablarray(args[i])
        # find the broadcast shape first based only on TablArray args
        bc = taShape((), 0)
        idx_is_ta, = np.nonzero(arg_is_ta)
        for i in idx_is_ta:
            bc, _ = args[i].ts.combine(bc)
        # Important: imply that np.ndarray args share same cdim as max
        cdim = bc.cdim
        # substitute TablArray args
        args2 = list(args)
        for i in idx_is_ta:
            arg = args[i]
            cdim_i = arg.ts.cdim
            if cdim_i == cdim:
                # if the cdim is at max, just get the base array
                args2[i] = arg.base
            else:
                # if the cdim is less than max
                cshape = list(arg.ts.cshape)
                # pad dimensions that lie in between this cdim and the
                # broadcast shape
                cshape2 = tuple([1] * (cdim - cdim_i) + cshape)
                # use that for a reshape
                arg2 = (arg.cell.reshape(cshape2)).table
                # but use the base
                args2[i] = arg2.base
        rval = func(*tuple(args2), **kwargs)
        if np.sum(arg_is_ta) > 0:
            # if any TablArray were passed, consider returning TablArray
            arg = args[idx_is_ta[0]]
            rclass = arg.__class__
            view = arg.view
            return misc._rval_once_a_ta(rclass, rval, cdim, view)
        else:
            return rval
    wrap_multi_bcast.__doc__ = (
        "**TablArray multi op compatible wrapped** %s\n\n" % func.__name__
        + wrap_multi_bcast.__doc__)
    return wrap_multi_bcast


# binary functions from numpy wrapped for TablArray compatibility

# these are also available as methods
add = tawrap_binarybroadcast(np.add)
subtract = tawrap_binarybroadcast(np.subtract)
multiply = tawrap_binarybroadcast(np.multiply)
power = tawrap_binarybroadcast(np.power)
true_divide = tawrap_binarybroadcast(np.true_divide)
divmod = tawrap_binarybroadcast(np.divmod)
equal = tawrap_binarybroadcast(np.equal, dtype=bool)
greater_equal = tawrap_binarybroadcast(np.greater_equal, dtype=bool)
greater = tawrap_binarybroadcast(np.greater, dtype=bool)
less_equal = tawrap_binarybroadcast(np.less_equal, dtype=bool)
less = tawrap_binarybroadcast(np.less, dtype=bool)
logical_and = tawrap_binarybroadcast(np.logical_and)
logical_or = tawrap_binarybroadcast(np.logical_or)
logical_xor = tawrap_binarybroadcast(np.logical_xor)

# these are only available here - not as methods
# allclose = tawrap_binarybroadcast(np.allclose, dtype=bool)
arctan2 = tawrap_binarybroadcast(np.arctan2)
bitwise_and = tawrap_binarybroadcast(np.bitwise_and)
bitwise_or = tawrap_binarybroadcast(np.bitwise_or)
bitwise_xor = tawrap_binarybroadcast(np.bitwise_xor)
copysign = tawrap_binarybroadcast(np.copysign)
divide = tawrap_binarybroadcast(np.true_divide)
float_power = tawrap_binarybroadcast(np.float_power)
floor_divide = tawrap_binarybroadcast(np.floor_divide)
fmax = tawrap_binarybroadcast(np.fmax)
fmin = tawrap_binarybroadcast(np.fmin)
fmod = tawrap_binarybroadcast(np.fmod)
gcd = tawrap_binarybroadcast(np.gcd)
heaviside = tawrap_binarybroadcast(np.heaviside)
hypot = tawrap_binarybroadcast(np.hypot)
isclose = tawrap_binarybroadcast(np.isclose, dtype=bool)
lcm = tawrap_binarybroadcast(np.lcm)
ldexp = tawrap_binarybroadcast(np.ldexp)
left_shift = tawrap_binarybroadcast(np.left_shift)
logaddexp = tawrap_binarybroadcast(np.logaddexp)
logaddexp2 = tawrap_binarybroadcast(np.logaddexp2)
maximum = tawrap_binarybroadcast(np.maximum)
minimum = tawrap_binarybroadcast(np.minimum)
mod = tawrap_binarybroadcast(np.remainder)
nextafter = tawrap_binarybroadcast(np.nextafter)
not_equal = tawrap_binarybroadcast(np.not_equal, dtype=bool)
remainder = tawrap_binarybroadcast(np.remainder)
right_shift = tawrap_binarybroadcast(np.right_shift)
