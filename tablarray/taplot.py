#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 15:52:31 2022

@author: chris
"""

from matplotlib import pyplot

from .np2ta import passwrap as pw


# passthrough without adding anything to pyplot functions
bar = pw._wrap_array_passthrough(pyplot.bar)
barbs = pw._wrap_array_passthrough(pyplot.barbs)
boxplot = pw._wrap_array_passthrough(pyplot.boxplot)
contour = pw._wrap_array_passthrough(pyplot.contour)
contourf = pw._wrap_array_passthrough(pyplot.contourf)
csd = pw._wrap_array_passthrough(pyplot.csd)
hist = pw._wrap_array_passthrough(pyplot.hist)
plot = pw._wrap_array_passthrough(pyplot.plot)
polar = pw._wrap_array_passthrough(pyplot.polar)
psd = pw._wrap_array_passthrough(pyplot.psd)
quiver = pw._wrap_array_passthrough(pyplot.quiver)
scatter = pw._wrap_array_passthrough(pyplot.scatter)
triplot = pw._wrap_array_passthrough(pyplot.triplot)
