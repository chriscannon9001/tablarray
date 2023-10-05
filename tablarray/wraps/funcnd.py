#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 18:56:18 2023

@author: chris
"""

import functools as _functools
import numpy as _np

from .. import misc


def tawrap_passthrough(func):
    """
    passthrough wrapper, to extract .base from any TablArray found
    in any *args, **kwargs
    """
    @_functools.wraps(func)
    def wrap_a_pthrough(*args, **kwargs):
        args2 = []
        kwargs2 = {}
        for arg in args:
            args2.append(arg.base if misc.istablarray(arg) else arg)
        for key, val in kwargs.items():
            kwargs2[key] = val.base if misc.istablarray(val) else val
        func(*tuple(args2), **kwargs2)
    wrap_a_pthrough.__doc__ = (
        "**TablArray compatible (passthrough)** %s\n\n" % func.__name__
        + wrap_a_pthrough.__doc__)
    return wrap_a_pthrough


def tawrap_elementwise(func):
    """
    TablArray wrap for numpy-compatible functions which have elementwise
    unary input.

    After wrap, the function will allow TablArray-like inputs including
    np.ndarray, or scalar.
    """
    @_functools.wraps(func)
    def wrap_elop_cast(x, *args, **kwargs):
        if misc.istablarray(x):
            rarray = func(x.base, *args, **kwargs)
            rclass = x.__class__
            # once a TablArray, usually a TablArray
            return misc._rval_once_a_ta(rclass, rarray, x.ts.cdim, x.view)
        else:
            # a is presumably array-like
            return func(x, *args, **kwargs)
    return wrap_elop_cast


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
    @_functools.wraps(func)
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
                    axis = tuple(_np.array(a._viewdims)[list(axis)])
                else:
                    axis = a._viewdims[axis]
                if a._cellular:
                    # if one of the cellular dims collapses to a scalar,
                    # then cdims will decrease
                    if _np.isscalar(axis):
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
            cdim = a.ts.cdim - delta_cdim if (_np.ndim(rarray) < _np.ndim(a.base)) else 0
            # once a TablArray, usually a TablArray
            return misc._rval_once_a_ta(rclass, rarray, cdim, a.view)
        else:
            # just passthrough
            return func(a, axis=axis, **kwargs)
    wrap_ax_bcast.__doc__ = (
        _doc_prepend + wrap_ax_bcast.__doc__)
    return wrap_ax_bcast


def tawrap_broadcastaxial(func, default_view=None):
    """
    TablArray wrap for numpy-compatible functions where the operation
    aligns to some axis. E.g. cumsum.

    After wrap, the function will allow TablArray-like inputs including
    np.ndarray, or scalar.
    """
    _doc_prepend = ("    **TablArray compatible** %s, where axis aligns w.r.t. view\n\n" % func.__name__
                    + "    view: 'cell', 'table', or None (default=%s)\n" % default_view
                    + "        overrides a.view if istablarray(a)\n"
                    + "    -----\n\n")
    @_functools.wraps(func)
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


def _cast_other_type(other, ta):
    """when a TablArray and other type are cast in a binary operator, make sure
    other is np.ndarray compatible, also maybe reorient for broadcasting
    if the TablArray is in a tabular view"""
    o_type = type(other)
    other = _np.array(other) if (o_type is list or o_type is tuple) else other
    if ta._tabular and not _np.isscalar(other):
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
    @_functools.wraps(func)
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
    from ..tashape import taShape
    Narg0 = len(arg_ctl)
    @_functools.wraps(func)
    def wrap_multi_bcast(*args, **kwargs):
        # get map of important arg types
        arg_is_ta = _np.zeros(Narg0, dtype=bool)
        for i in range(min(len(args), Narg0)):
            # if the arg_ctl masked off the argument here, ignore it
            if arg_ctl[i]:
                arg_is_ta[i] = misc.istablarray(args[i])
        # find the broadcast shape first based only on TablArray args
        bc = taShape((), 0)
        idx_is_ta, = _np.nonzero(arg_is_ta)
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
        if _np.sum(arg_is_ta) > 0:
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


def tawrap_new(func):
    """
    TablArray wrap for numpy-compatible functions which make new arrays::
        
        original(shape, **kwargs) --> wrapped(shape, cdim, **kwargs)
    
    decorator for ATC compatibility for new numpy generators,
    but I'm not so sure I should use a decorator or even have the same names
    the thing is that this decorator changes the input args..."""
    from ..ta import TablArray
    @_functools.wraps(func)
    def wrapper_atc_new(shape, cdim, view='cell', **kwargs):
        rarray = func(shape, **kwargs)
        return TablArray(rarray, cdim, view=view)
    return wrapper_atc_new
