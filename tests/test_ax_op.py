#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 21:32:24 2021

@author: chris
"""

import numpy as np
import unittest

import tablarray as ta


# I picked axial functions that are valid for all real numbers
AX_FUNCS = ['sum',
            'std',
            'amax',
            'max',
            'prod',
            'median']

AX2_FUNCS = ['cumprod',
             'cumsum']


class Test_AXop(unittest.TestCase):
    """show TablArray versions of axial operands work the same as numpy"""

    def setUp(self):
        v = np.random.randn(4, 2, 3)
        answers = [{}, {}]
        for fname in AX_FUNCS + AX2_FUNCS:
            func = np.__dict__[fname]
            for ax in [1, 2]:
                answers[ax - 1][fname] = func(v, axis=ax)
        self.tav = ta.TablArray(v, 1)
        self.v = v
        self.answers = answers

    def test_tableaxial_operators(self):
        for fname in AX_FUNCS:
            ta_func = ta.__dict__[fname]
            ta_answ = ta_func(self.tav.table, axis=1)
            self.assertEqual(self.tav.ts.cdim, ta_answ.ts.cdim,
                             msg="cdim not preserved after %s" % fname)
            allclose = np.allclose(self.answers[0][fname], ta_answ.base)
            self.assertTrue(allclose,
                            msg="results differ in %s" % fname)

    def test_cellaxial_operators(self):
        for fname in AX_FUNCS:
            ta_func = ta.__dict__[fname]
            ta_answ = ta_func(self.tav.cell, axis=0)
            # self.assertEqual(self.tav.ts.cdim, ta_answ.ts.cdim,
            #                 msg="cdim not preserved after %s" % fname)
            allclose = np.allclose(self.answers[1][fname], ta_answ.base)
            self.assertTrue(allclose,
                            msg="results differ in %s" % fname)
