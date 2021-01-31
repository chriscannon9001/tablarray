#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests on kwargs_tools.array_2_args, the most important component 
of that lib.

Using kwargs_tools.test_file_2_args for leg-work setting up the problem.

Use a reference args, kwargs set to compare results from
tests/resources/scan_test_case1.csv.

Created on Tue Mar 26 22:22:58 2019

@author: chris
"""

import unittest
import pkg_resources as pkrs
import numpy as np
import copy

import tablarray.kwtools as kwt


def _equality_tester(obj1, obj2):
    """Do a detail level structural and values comparison of obj1 and obj2.

    Return False unless they are identical.
    """
    if type(obj1) is np.ndarray:
        if not type(obj2) is np.ndarray: return False
        if len(obj1) != len(obj2): return False
        for a in range(len(obj1)):
            eq = _equality_tester(obj1[a], obj2[a])
            if not eq: return False
        return True
    elif type(obj1) is dict:
        if not type(obj2) is dict: return False
        for key in obj1:
            if not key in obj2: return False
            eq = _equality_tester(obj1[key], obj2[key])
            if not eq: return False
        return True
    else: return np.isclose(obj1, obj2)


class TestTest(unittest.TestCase):
    """Units tests for _equality_tester.
    Catch 22.
    I made a module for testing others.
    """
    def test_1eq1(self):
        self.assertTrue(_equality_tester(1, 1), '_equality_tester returns False for 1==1')

    def test_floateqfloat(self):
        self.assertTrue(_equality_tester(3.13, 3.13), '_equality_tester returns False for 3.13==3.13')

    def test_1ne2(self):
        self.assertFalse(_equality_tester(1, 2), '_equality_tester returns True for 1==2')

    def test_a1eqa2(self):
        for a in range(20):
            s1 = int(np.random.rand() * 10)
            s2 = int(np.random.rand() * 4)
            a1 = np.random.randn(s1, s2)
            a2 = copy.deepcopy(a1)
            self.assertTrue(_equality_tester(a1, a2),
                            '_equality_tester returns False for mxn matrix copy')

    def test_a1neb1(self):
        for a in range(20):
            s1 = int(np.random.rand() * 11 + 2)
            s2 = int(np.random.rand() * 5 + 1)
            a1 = np.random.randn(s1, s2)
            a2 = copy.deepcopy(a1)
            a1[1,0] = -1e10
            self.assertFalse(_equality_tester(a1, a2),
                             '_equality_tester returns True for altered mxn matrix comparison')

    def test_dicteqdict(self):
        d1 = {'a':100, 'b':1, 'c':-12, 'd':3.14159}
        d2 = copy.deepcopy(d1)
        self.assertTrue(_equality_tester(d1, d2))

    def test_wrong_levels(self):
        a1 = np.array([[[[-1]]]])
        a2 = np.array([-1])
        self.assertFalse(_equality_tester(a1, a2))

    def test_empty1(self):
        a1 = np.array([])
        a2 = a1
        self.assertTrue(_equality_tester(a1, a2))
    
    def test_empty2(self):
        a1 = np.array([])
        a2 = np.array([[]])
        self.assertFalse(_equality_tester(a1, a2))
    
    def test_empty3(self):
        a1 = np.array([[]])
        a2 = np.array([])
        self.assertFalse(_equality_tester(a1, a2))

# it's important that this actually matches the structure encoded in scan_test_case1.csv
ref1_args = (1,
            3.14,
            np.array([[0, 1], [-1, .01]]))

ref1_kwargs = {'bb':np.array([1.5]),
              'g':np.array([{'a':np.array([1.1, 100.325]), 'b':np.array([1.2, 80.52])},
                             {'a':np.array([1.15, -70.25]), 'b':np.array([1.13, 346])}
                             ]),
              'h':np.array([[[1.1, 100.325], [1.6, 65]],
                            [[1.2, 80.52], [1.65, 64.53]]]),
              'i':np.array([1.16, 2.13]),
              'j':np.array([[[[-1e-5]]]]),
              'k':np.array([]),
              'l':{'a':.01, 'b':-.05}
              }

class kwScanTestCase1(unittest.TestCase):
    '''Tests for kwargs_scan.py
    By loading scan_test_case1.csv and testing each arg, kwarg matches
    ref1_args and ref1_kwgargs defined above.
    '''
    def setUp(self):
        fname = pkrs.resource_filename(__name__, 'resources/scan_test_case1.csv')
        self.args, self.kwargs = kwt.test_file_2_args(fname)
    
    def test_arg1(self):
        '''Test whether 1 was cast as args[0] as expected.
        '''
        self.assertTrue(_equality_tester(self.args[0], ref1_args[0]))
    
    def test_arg2(self):
        '''Test whether 3.14 was cast as args[1] as expected'''
        self.assertTrue(_equality_tester(self.args[1], ref1_args[1]))
    
    def test_arg3(self):
        '''Test whether args[2] is the 2x2 array as expected'''
        self.assertTrue(_equality_tester(self.args[2], ref1_args[2]))
    
    def test_len_args(self):
        '''Test whether the len(args) eq 3 as expected'''
        self.assertEqual(len(self.args), 3, msg=('len(args)=%d, but expected 3'
                                                 % len(self.args)))
    
    def test_kwarg1(self):
        '''Test whether kwarg 'c' is missing'''
        self.assertNotIn('c', self.kwargs)
    
    def test_kwarg2(self):
        '''Test whether kwarg 'bb':[1.5] is as expected'''
        self.assertIn('bb', self.kwargs)
        self.assertTrue(_equality_tester(self.kwargs['bb'], ref1_kwargs['bb']))
    
    def test_kwarg4(self):
        '''Test whether kwarg 'g':[[[...]]] is as expected.'''
        self.assertIn('g', self.kwargs)
        self.assertTrue(_equality_tester(self.kwargs['g'], ref1_kwargs['g']))
    
    def test_kwarg5(self):
        '''Test whether kwarg 'h':[[[...]]] is as expected.'''
        self.assertIn('h', self.kwargs)
        self.assertTrue(_equality_tester(self.kwargs['h'], ref1_kwargs['h']))

    def test_kwarg6(self):
        '''Test whether kwarg 'i':[...] is as expected.'''
        self.assertIn('i', self.kwargs)
        self.assertTrue(_equality_tester(self.kwargs['i'], ref1_kwargs['i']))

    def test_kwarg7(self):
        '''Test whether kwarg 'j':[[[[-1e-5]]]] is as expected.'''
        self.assertIn('j', self.kwargs)
        self.assertTrue(_equality_tester(self.kwargs['j'], ref1_kwargs['j']))
    
    def test_kwarg8(self):
        '''Test whether kwarg 'k':[] is as expected.'''
        self.assertIn('k', self.kwargs)
        self.assertTrue(_equality_tester(self.kwargs['k'], ref1_kwargs['k']))

    def test_kwarg9(self):
        '''Test whether kwarg 'l':{'a':.01, 'b':-.05} is as expected.'''
        self.assertIn('l', self.kwargs)
        self.assertTrue(_equality_tester(self.kwargs['l'], ref1_kwargs['l']))

if __name__ == '__main__':
    unittest.main()