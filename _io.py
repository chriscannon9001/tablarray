#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities that tend to get used front and back (IO) of tablarray functions

Created on Sun Jan  3 15:34:31 2021

@author: chris
"""


def quackslike_Tablarray(obj):
    return hasattr(obj, 'ts') and hasattr(obj, 'view')


def rval_once_a_ta(rclass, rval, cdim, view):
    """"""
    if rval.ndim == cdim:
        return rval
    return rclass(rval, cdim, view)


def rval_always_ta(rclass, rval, cdim, view):
    return rclass(rval, cdim, view)
