#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 11:14:22 2020

@author: chris
"""

import numpy as np


def array2str_plus_tdim(array, tdim, precision=5, linewidth=75):
    """Given array-like argument and tdim (tabular-dimension)
    add a barrier '|' in the string conversion to demark tabular-cellular
    structure."""
    def array2str(a):
        """wrap array->str in set/unset of numpy options"""
        # capture original option settings
        d_opts = np.get_printoptions()
        precision0 = d_opts['precision']
        linewidth0 = d_opts['linewidth']
        # set my options
        np.set_printoptions(precision=precision, linewidth=linewidth)
        a_str = a.__str__()
        # restor original options (no side-effects)
        np.set_printoptions(precision=precision0, linewidth=linewidth0)
        return a_str
    # convert array to string using numpy options
    arraystr = array2str(array)
    # initial setup before parse
    new_arraystr = ''               # start output as blank
    lines = arraystr.split('\n')    # split input by lines
    n_lines = len(lines)            # number of input lines
    dim_i = 0                       # current dim = 0
    # parse line by line through the arraystr
    for i in range(n_lines):
        line = lines[i]
        if len(line) < 1:
            # blank lines are unmodified, nothing do to here
            pass
        else:
            # count number of brackets
            dim_opened = len(line.split('[')) - 1
            dim_closed = len(line.split(']')) - 1
            # determine whether to add front barrier or not
            if (dim_i + dim_opened) >= tdim:
                # front-side barrier
                new_arraystr += line[:tdim]
                new_arraystr += '|'
                # determine where to put back barrier
                dim_i2 = dim_i + (dim_opened - dim_closed)
                del_dim = dim_i2 - tdim
                back = ' ' * (dim_i2 - tdim) + '|'
                if del_dim >= 0:
                    # tdim does not have closure, so back-'|' is at end
                    new_arraystr += line[tdim:] + back
                else:
                    # tdim does have closure, so back-'|' precedes some ']'
                    new_arraystr += (line[tdim:del_dim] + back
                                     + line[del_dim:])
            else:
                # no barrier
                new_arraystr += line
            # update dim_i
            dim_i = dim_i2
        # newlines only only go between lines
        if i < (n_lines - 1):
            new_arraystr += '\n'
    return new_arraystr
