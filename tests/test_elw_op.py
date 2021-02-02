#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 21:32:24 2021

@author: chris
"""

import numpy as np
import unittest

import tablarray as ta


# I picked elementwise functions that are valid for all real numbers
EL_FUNCS = ['abs',
            'exp',
            'floor',
            'square',
            'rad2deg']


class Test_ELop(unittest.TestCase):
    """show TablArray versions of element-wise tests work the same as numpy"""

    def setUp(self):
        v = np.random.randn(4, 3, 2)
        answers = {}
        for fname in EL_FUNCS:
            func = np.__dict__[fname]
            answers[fname] = func(v)
        self.tav = ta.TablArray(v, 1)
        self.v = v
        self.answers = answers

    def test_elementwise_operators(self):
        for fname in EL_FUNCS:
            ta_func = ta.__dict__[fname]
            ta_answ = ta_func(self.tav)
            self.assertEqual(self.tav.ts.cdim, ta_answ.ts.cdim,
                             msg="cdim not preserved after %s" % fname)
            allclose = np.allclose(self.answers[fname], ta_answ.base)
            self.assertTrue(allclose,
                            msg="results differ in %s" % fname)
