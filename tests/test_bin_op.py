#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 21:32:24 2021

@author: chris
"""

import numpy as np
import unittest

import tablarray as ta


# I picked binary operands that are valid for all real numbers
BIN_FUNCS = ['add',
             'multiply',
             'maximum',
             'copysign',
             'arctan2']
# NOTE: some of these actually return bool, which I did not consider!


class Test_BIop(unittest.TestCase):
    """show TablArray versions of element-wise tests work the same as numpy"""

    def setUp(self):
        """You do have to pay close attention to my reshapes calls to
        understand how I got the numpy and tablarray to do the same thing."""
        x1 = np.random.randn(4, 1, 2)
        x2 = np.random.randn(3, 2)
        answers = {}
        for fname in BIN_FUNCS:
            func = np.__dict__[fname]
            answers[fname] = func(x1, x2).reshape(4, 3, 1, 2)
        self.ta_x1 = ta.TablArray(x1, 1)
        self.ta_x2 = ta.TablArray(x2.reshape(3, 1, 2), 2)
        self.x1 = x1
        self.x1 = x2
        self.answers = answers

    def test_binary_operators(self):
        for fname in BIN_FUNCS:
            ta_func = ta.__dict__[fname]
            ta_answ = ta_func(self.ta_x1, self.ta_x2)
            self.assertEqual(2, ta_answ.ts.cdim,
                             msg="cdim not correct after %s" % fname)
            allclose = np.allclose(self.answers[fname], ta_answ.base)
            self.assertTrue(allclose,
                            msg="results differ in %s" % fname)
