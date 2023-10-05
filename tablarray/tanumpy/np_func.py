#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 09:53:28 2023

@author: chris
"""

import numpy as _np

from ..wraps import tawrap_ax2scalar as _tawrap_ax2scalar
# these are also available as methods
all = _tawrap_ax2scalar(_np.all)
any = _tawrap_ax2scalar(_np.any)
argmax = _tawrap_ax2scalar(_np.argmax)
argmin = _tawrap_ax2scalar(_np.argmin)
max = _tawrap_ax2scalar(_np.max)
mean = _tawrap_ax2scalar(_np.mean)
min = _tawrap_ax2scalar(_np.min)
prod = _tawrap_ax2scalar(_np.prod)
std = _tawrap_ax2scalar(_np.std)
sum = _tawrap_ax2scalar(_np.sum)
# these are only available here - not as methods
amax = _tawrap_ax2scalar(_np.amax)
amin = _tawrap_ax2scalar(_np.amin)
median = _tawrap_ax2scalar(_np.median)
nanargmax = _tawrap_ax2scalar(_np.nanargmax)
nanargmin = _tawrap_ax2scalar(_np.nanargmin)
nanmax = _tawrap_ax2scalar(_np.nanmax)
nanmean = _tawrap_ax2scalar(_np.nanmean)
nanmedian = _tawrap_ax2scalar(_np.nanmedian)
nanmin = _tawrap_ax2scalar(_np.nanmin)
nanprod = _tawrap_ax2scalar(_np.nanprod)
nansum = _tawrap_ax2scalar(_np.nansum)
nanstd = _tawrap_ax2scalar(_np.nanstd)
nanvar = _tawrap_ax2scalar(_np.nanvar)
var = _tawrap_ax2scalar(_np.var)

from ..wraps import tawrap_broadcastaxial as _tawrap_broadcastaxial
cumprod = _tawrap_broadcastaxial(_np.cumprod)
cumsum = _tawrap_broadcastaxial(_np.cumsum)
nancumprod = _tawrap_broadcastaxial(_np.nancumprod)
nancumsum = _tawrap_broadcastaxial(_np.nancumsum)

from ..wraps import tawrap_binarybroadcast
# binary functions from numpy wrapped for TablArray compatibility
# these are also available as methods
add = tawrap_binarybroadcast(_np.add)
subtract = tawrap_binarybroadcast(_np.subtract)
multiply = tawrap_binarybroadcast(_np.multiply)
power = tawrap_binarybroadcast(_np.power)
true_divide = tawrap_binarybroadcast(_np.true_divide)
divmod = tawrap_binarybroadcast(_np.divmod)
equal = tawrap_binarybroadcast(_np.equal, dtype=bool)
greater_equal = tawrap_binarybroadcast(_np.greater_equal, dtype=bool)
greater = tawrap_binarybroadcast(_np.greater, dtype=bool)
less_equal = tawrap_binarybroadcast(_np.less_equal, dtype=bool)
less = tawrap_binarybroadcast(_np.less, dtype=bool)
logical_and = tawrap_binarybroadcast(_np.logical_and)
logical_or = tawrap_binarybroadcast(_np.logical_or)
logical_xor = tawrap_binarybroadcast(_np.logical_xor)
# these are only available here - not as methods
# allclose = tawrap_binarybroadcast(_np.allclose, dtype=bool)
arctan2 = tawrap_binarybroadcast(_np.arctan2)
bitwise_and = tawrap_binarybroadcast(_np.bitwise_and)
bitwise_or = tawrap_binarybroadcast(_np.bitwise_or)
bitwise_xor = tawrap_binarybroadcast(_np.bitwise_xor)
copysign = tawrap_binarybroadcast(_np.copysign)
divide = tawrap_binarybroadcast(_np.true_divide)
float_power = tawrap_binarybroadcast(_np.float_power)
floor_divide = tawrap_binarybroadcast(_np.floor_divide)
fmax = tawrap_binarybroadcast(_np.fmax)
fmin = tawrap_binarybroadcast(_np.fmin)
fmod = tawrap_binarybroadcast(_np.fmod)
gcd = tawrap_binarybroadcast(_np.gcd)
heaviside = tawrap_binarybroadcast(_np.heaviside)
hypot = tawrap_binarybroadcast(_np.hypot)
isclose = tawrap_binarybroadcast(_np.isclose, dtype=bool)
lcm = tawrap_binarybroadcast(_np.lcm)
ldexp = tawrap_binarybroadcast(_np.ldexp)
left_shift = tawrap_binarybroadcast(_np.left_shift)
logaddexp = tawrap_binarybroadcast(_np.logaddexp)
logaddexp2 = tawrap_binarybroadcast(_np.logaddexp2)
maximum = tawrap_binarybroadcast(_np.maximum)
minimum = tawrap_binarybroadcast(_np.minimum)
mod = tawrap_binarybroadcast(_np.remainder)
nextafter = tawrap_binarybroadcast(_np.nextafter)
not_equal = tawrap_binarybroadcast(_np.not_equal, dtype=bool)
remainder = tawrap_binarybroadcast(_np.remainder)
right_shift = tawrap_binarybroadcast(_np.right_shift)

from ..wraps import tawrap_elementwise
# elementwise functions
abs = tawrap_elementwise(_np.abs)
absolute = tawrap_elementwise(_np.absolute)
angle = tawrap_elementwise(_np.angle)
arccos = tawrap_elementwise(_np.arccos)
arccosh = tawrap_elementwise(_np.arccosh)
arcsin = tawrap_elementwise(_np.arcsin)
arcsinh = tawrap_elementwise(_np.arcsinh)
arctan = tawrap_elementwise(_np.arctan)
arctanh = tawrap_elementwise(_np.arctanh)
bitwise_not = tawrap_elementwise(_np.bitwise_not)
cbrt = tawrap_elementwise(_np.cbrt)
ceil = tawrap_elementwise(_np.ceil)
conj = tawrap_elementwise(_np.conjugate)
conjugate = tawrap_elementwise(_np.conjugate)
cos = tawrap_elementwise(_np.cos)
cosh = tawrap_elementwise(_np.cosh)
deg2rad = tawrap_elementwise(_np.deg2rad)
degrees = tawrap_elementwise(_np.degrees)
exp = tawrap_elementwise(_np.exp)
exp2 = tawrap_elementwise(_np.exp2)
expm1 = tawrap_elementwise(_np.expm1)
fabs = tawrap_elementwise(_np.fabs)
floor = tawrap_elementwise(_np.floor)
imag = tawrap_elementwise(_np.imag)
invert = tawrap_elementwise(_np.invert)
iscomplex = tawrap_elementwise(_np.iscomplex)
isfinite = tawrap_elementwise(_np.isfinite)
isinf = tawrap_elementwise(_np.isinf)
isnan = tawrap_elementwise(_np.isnan)
isnat = tawrap_elementwise(_np.isnat)
isreal = tawrap_elementwise(_np.isreal)
log = tawrap_elementwise(_np.log)
log10 = tawrap_elementwise(_np.log10)
log1p = tawrap_elementwise(_np.log1p)
log2 = tawrap_elementwise(_np.log2)
logical_not = tawrap_elementwise(_np.logical_not)
negative = tawrap_elementwise(_np.negative)
positive = tawrap_elementwise(_np.positive)
rad2deg = tawrap_elementwise(_np.rad2deg)
radians = tawrap_elementwise(_np.radians)
round = tawrap_elementwise(_np.round)
real = tawrap_elementwise(_np.real)
real_if_close = tawrap_elementwise(_np.real_if_close)
reciprocal = tawrap_elementwise(_np.reciprocal)
rint = tawrap_elementwise(_np.rint)
sign = tawrap_elementwise(_np.sign)
signbit = tawrap_elementwise(_np.signbit)
sin = tawrap_elementwise(_np.sin)
sinc = tawrap_elementwise(_np.sinc)
sinh = tawrap_elementwise(_np.sinh)
spacing = tawrap_elementwise(_np.spacing)
square = tawrap_elementwise(_np.square)
sqrt = tawrap_elementwise(_np.sqrt)
tan = tawrap_elementwise(_np.tan)
tanh = tawrap_elementwise(_np.tanh)
trunc = tawrap_elementwise(_np.trunc)
