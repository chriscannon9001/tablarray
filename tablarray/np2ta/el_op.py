#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:58:22 2020

@author: chris
"""

import functools
import numpy as np

from .. import misc


def tawrap_elementwise(func):
    """
    TablArray wrap for numpy-compatible functions which have elementwise
    unary input. - once a TablArray, always a TablArray

    After wrap, the function will allow TablArray-like inputs including
    np.ndarray, or scalar.
    """
    @functools.wraps(func)
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


abs = tawrap_elementwise(np.abs)
absolute = tawrap_elementwise(np.absolute)
angle = tawrap_elementwise(np.angle)
arccos = tawrap_elementwise(np.arccos)
arccosh = tawrap_elementwise(np.arccosh)
arcsin = tawrap_elementwise(np.arcsin)
arcsinh = tawrap_elementwise(np.arcsinh)
arctan = tawrap_elementwise(np.arctan)
arctanh = tawrap_elementwise(np.arctanh)
bitwise_not = tawrap_elementwise(np.bitwise_not)
cbrt = tawrap_elementwise(np.cbrt)
ceil = tawrap_elementwise(np.ceil)
conj = tawrap_elementwise(np.conjugate)
conjugate = tawrap_elementwise(np.conjugate)
cos = tawrap_elementwise(np.cos)
cosh = tawrap_elementwise(np.cosh)
deg2rad = tawrap_elementwise(np.deg2rad)
degrees = tawrap_elementwise(np.degrees)
exp = tawrap_elementwise(np.exp)
exp2 = tawrap_elementwise(np.exp2)
expm1 = tawrap_elementwise(np.expm1)
fabs = tawrap_elementwise(np.fabs)
floor = tawrap_elementwise(np.floor)
imag = tawrap_elementwise(np.imag)
invert = tawrap_elementwise(np.invert)
iscomplex = tawrap_elementwise(np.iscomplex)
isfinite = tawrap_elementwise(np.isfinite)
isinf = tawrap_elementwise(np.isinf)
isnan = tawrap_elementwise(np.isnan)
isnat = tawrap_elementwise(np.isnat)
isreal = tawrap_elementwise(np.isreal)
log = tawrap_elementwise(np.log)
log10 = tawrap_elementwise(np.log10)
log1p = tawrap_elementwise(np.log1p)
log2 = tawrap_elementwise(np.log2)
logical_not = tawrap_elementwise(np.logical_not)
negative = tawrap_elementwise(np.negative)
positive = tawrap_elementwise(np.positive)
rad2deg = tawrap_elementwise(np.rad2deg)
radians = tawrap_elementwise(np.radians)
round = tawrap_elementwise(np.round)
real = tawrap_elementwise(np.real)
real_if_close = tawrap_elementwise(np.real_if_close)
reciprocal = tawrap_elementwise(np.reciprocal)
rint = tawrap_elementwise(np.rint)
sign = tawrap_elementwise(np.sign)
signbit = tawrap_elementwise(np.signbit)
sin = tawrap_elementwise(np.sin)
sinc = tawrap_elementwise(np.sinc)
sinh = tawrap_elementwise(np.sinh)
spacing = tawrap_elementwise(np.spacing)
square = tawrap_elementwise(np.square)
sqrt = tawrap_elementwise(np.sqrt)
tan = tawrap_elementwise(np.tan)
tanh = tawrap_elementwise(np.tanh)
trunc = tawrap_elementwise(np.trunc)
