#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:37:48 2020

@author: chris
"""

import numpy as np
import unittest

import tablarray as tarr


def norm(xy):
    r = (xy ** 2).sum()**(.5)
    return r


class Test_ATC(unittest.TestCase):
    """show that a simple norm function works on ATC's"""
    def setUp(self):
        theta = np.linspace(0, np.pi, 16)
        self.R = 5.1
        x0 = self.R * np.cos(theta)
        y0 = self.R * np.sin(theta)
        self.xy = tarr.TablArray(list(zip(x0, y0)), 1, 'cell')

    def test_rnorm(self):
        r = norm(self.xy)
        xynorm = self.xy / r
        # r.setview('table')
        self.assertAlmostEqual(r.table.mean().array, self.R)
        rnorm = norm(xynorm)
        # rnorm.setview('table')
        self.assertAlmostEqual(rnorm.table.mean().array, 1)
        self.assertAlmostEqual(rnorm.table.std().array, 0)


class Test_ATC2(unittest.TestCase):
    """show that """
    def setUp(self):
        self.theta = np.linspace(0, np.pi, 16)
        self.R = np.linspace(1, 8, 16)
        
