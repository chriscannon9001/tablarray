#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo TablArray
--------
demonstrations of what TablArray can do

These are forms of a "bcastget/set" type"
a.q.set(row=0) = 5 * b.q.get(row=0) # ignored col of row,col
..
a.set('tabular')
b.set('tabular')
a[0, :] = 5 * b[0, :]
# where indices are always allowed according to broadcasting rules
# i.e. ignores dims like row if nonexistent, right aligned
# also flatten indices if col dimension is 1

Design / todo:
    1. Factor together TablArrayShapes, bcast, and bcastget/set
    (1b. similar TablArrayShapes, etc. using tshape:dict)
    2. single TablArray form with behavior switch

Created on Sun Mar  1 11:56:25 2020

@author: chris
"""

import numpy as np

import tablarray as tarr
print('import tablarray as tarr')

print('===  TablArray math demos  ===')
a = tarr.TablArray.from_tile(np.array([5., -1.]), (2, 1))
a.table[:, 0] *= [0.9, 1.1]
print('a = tarr.TablArray.from_tile(np.array([5, -1]), (2, 1))')
print('a.table[:, 0] *= [0.9, 1.1]')
b = tarr.TablArray.from_tile(np.identity(2), (2, 2))
b.cell[0, 1].table[:, :] += [[.2, .3], [-.3, -.5]]
print('b = tarr.TablArray.from_tile(np.identity(2), (2, 2))')
print('b.cell[0, 1].table[:, :] += [[.2, .3], [-.3, -.5]]')
print('===  c = a + b  ===')
c = a + b
print(c)
print('===  d = b / a  ===')
d = b / a
# print('shape of d: %s' % c.ts)
print(d)
print('===  e = tarr.matmul(b, a)  ===')
e = tarr.matmul(b, a)
# print('shape of e: %s' % c.ts)
print(e)
print("===  a.setview('cell')  ===")
a.setview('cell')
print('===  f = a / a[1]  ===')
f = a / a[1]
print(f)

print("===  TablArray bcast demos  ===")
m = tarr.TablArray.from_tile(
        np.array([np.pi, -1]), (2, 3), view='bcast')
print("m = tarr.TablArray.from_tile(\n\tnp.array([np.pi, -1]), (2, 3), view='bcast')")
m_r0 = m[0, :]
m_r0.setview('cell')
print('m_r0 = m[0, :]')
m[0, :] = m_r0 / 2
print('m[0, :] = m_r0 / 2')
print('===  m  ===')
print(m)
