#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATC - 'Array-Tabular-Cellular' or tables of cells
When the best structure of data is any-dimensional cells arranged
in any-dimensional tables - ATC provides fast numpy-like array
operations with broadcasting to handle both cellular-dimensions
and tabular-dimensions at once.

Portions of ATC looks like pass-through to numpy with a tiny overhead to manage
cdim. But the irreplaceable part of ATC is broadcasting for binary operands.

ATC was originally developed to manage large numbers of optical modes.

Created on Fri May 15 17:40:41 2020

@author: chris
"""

from . import linalg

from .ax_op import *
from .ax2_op import *
from .bin_op import *
from .cmatmul import *
from .db import *
from .el_op import *
from .mmul import *
from .new import *
from .ta import *
