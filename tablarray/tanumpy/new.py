#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 21:22:26 2020

@author: chris
"""

import numpy as _np

from ..tashape import taShape
from ..wraps import tawrap_new

empty = tawrap_new(_np.empty)
ones = tawrap_new(_np.ones)
zeros = tawrap_new(_np.zeros)
# full wouldn't have exaclty the same wrapper

def imply_zeros(a, *args):
    """based on TablArray inputs, """
    ts0 = taShape((), 0)
    ts, bc = a.ts.combine(ts0)
    for arg in args:
        ts, bc = ts.combine(arg.ts)
    # now make zeros with ts
    return zeros((*ts.tshape, *ts.cshape), ts.cdim)
