#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(This isn't testing framework)

Run this script to scan numpy for functions that aren't implemented in TablArray

Created on Tue May 19 17:46:31 2020

@author: chris
"""

import inspect


def argfind(func):
    args = []
    sig = ''
    try:
        argspec = inspect.getfullargspec(func)
        args = argspec.args
        name = func.__name__ if hasattr(func, '__name__') else ''
        sig = name + '(' + ', '.join(args) + ')'
    except TypeError:
        if hasattr(func, '__doc__') and type(func.__doc__) is str:
            sig = func.__doc__.split('\n')[0]
            args = ''.join(sig.split('(')[1:])
            args = ''.join(args.split(')')[:-1])
            args = args.split(', ')
            # remove defaults args
            args = [arg.split('=')[0] for arg in args]
    return args, sig


def dsub(a, b):
    d2 = dict()
    for key, val in a.items():
        if key not in b:
            d2[key] = val
    return d2


class MethodID(dict):
    def __init__(self, *signatures):
        self.signatures = signatures

    def ifmatch(self, name, func):
        args, signature = argfind(func)
        for sig in self.signatures:
            n = len(sig)
            if sig == args[:n]:
                # matches the signature
                self[name] = signature
                return True
        # no match found
        return False

    def __sub__(self, other):
        if isinstance(other, dict):
            d2 = MethodID(*self.signatures)
            for key, val in self.items():
                if key not in other:
                    d2[key] = val
            return d2

    def __str__(self):
        string = ''
        for key, val in self.items():
            sval = ('%s' % val)[:60]
            string += "'%s': %s\n" % (key, sval)
        return string


class ClassifyMethods(object):
    def __init__(self, bin_ops, elw_ops, ax_ops, new, unknown, properties):
        self.bin_ops = bin_ops
        self.elw_ops = elw_ops
        self.ax_ops = ax_ops
        self.new = new
        self.unknown = unknown
        self.properties = properties

    @classmethod
    def from_pkg(cls, package):
        """classify everything in numpy.__dict__"""
        bin_ops = MethodID(['x1', 'x2'], ['a', 'b'])
        ax_ops = MethodID(['x', 'axis'], ['a', 'axis'])
        elw_ops = MethodID(['x'])
        new = MethodID(['shape'])
        unknown = {}
        properties = {}
        for key, val in package.__dict__.items():
            if callable(val):
                # inspect.getargspec doesn't work on numpy
                # this is a crude method to get arg names from __doc__
                if bin_ops.ifmatch(key, val):
                    pass
                elif ax_ops.ifmatch(key, val):
                    pass
                elif elw_ops.ifmatch(key, val):
                    pass
                elif new.ifmatch(key, val):
                    pass
                else:
                    unknown[key] = val
            else:
                # if not callable, val is a property
                properties[key] = val
        return cls(bin_ops, elw_ops, ax_ops, new, unknown, properties)

    def __sub__(self, other):
        if isinstance(other, ClassifyMethods):
            bo2 = self.bin_ops - other.bin_ops
            eo2 = self.elw_ops - other.elw_ops
            ao2 = self.ax_ops - other.ax_ops
            nw2 = self.new - other.new
            un2 = dsub(self.unknown, other.unknown)
            pr2 = dsub(self.properties, other.properties)
            return ClassifyMethods(bo2, eo2, ao2, nw2, un2, pr2)


if __name__ == '__main__':
    import numpy as np
    import tablarray as ta
    cl_np = ClassifyMethods.from_pkg(np)
    cl_ta = ClassifyMethods.from_pkg(ta)
    cl_del1 = cl_np - cl_ta
    cl_del2 = cl_ta - cl_np
    print('=== bin_ops not in tablarray ===')
    print(cl_del1.bin_ops)
    print('=== elw_ops not in tablarray ===')
    print(cl_del1.elw_ops)
    print('=== ax_ops not in tablarray ===')
    print(cl_del1.ax_ops)
    print('=== new array funcs not in tablarray')
    print(cl_del1.new)
    cl_np = ClassifyMethods.from_pkg(np.linalg)
    cl_ta = ClassifyMethods.from_pkg(ta.linalg)
    cl_del1 = cl_np - cl_ta
    cl_del2 = cl_ta - cl_np
    print('=== bin_ops not in tablarray.linalg ===')
    print(cl_del1.bin_ops)
    print('=== elw_ops not in tablarray.linalg ===')
    print(cl_del1.elw_ops)
    print('=== ax_ops not in tablarray.linalg ===')
    print(cl_del1.ax_ops)
    print('=== new array funcs not in tablarray.linalg')
    print(cl_del1.new)
