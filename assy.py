#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:44:05 2020

@author: chris
"""

import collections
import numpy as np

# internal imports
from . import tashape


class Assembly(object):
    """
    Assembly
    --------
    dictionary of broadcast-able TablArray's

    E.g.::

        import numpy as np
        import tablarray as ta
        x0, y0 = np.meshgrid(linspace(-2, 2, 4), linspace(-1.5, 1.5, 4))
        x = ta.TablArray(x0, 0)
        y = ta.TablArray(y0, 0)
        E = ta.zeros((4, 4, 2), cdim=1)
        ta.cell(E)[0] = 1.5 / (x**2 + y**2)
        ta.cell(E)[1] = -.01 + x * 0
        En = ta.abs(ta.linalg.norm(E)**2)
        Efield = ta.Assembly(x=x, y=y, E=E, En=En)

    Parameters
    ----------
    **kwargs: keyword=TablArray args
        name1=tablarray1, name2=tablarray2, ...

    With each additional element, broadcast compatibility is enforced::

        s1 = ta.Assembly()
        s1['x'] = ta.TablArray(np.linspace(-1, 3, 5), 0)
        s1['y'] = ta.TablArray([0, 0, 0], 0)
        >>> ValueError: refused to load incompatible shape t(3,)|c() into shape t(5,)|c()
    """

    def __init__(self, **kwargs):
        # this is used to track and check broadcastability
        self._ts = None
        # a facade for some dict methods of the table
        self._table = collections.OrderedDict()
        self.keys = self._table.keys
        self.items = self._table.items
        self.pop = self._table.pop
        # controls __str__(keys)
        self._str_keys = None
        self._str_excluded = None
        for key, val in kwargs.items():
            self[key] = val

    def __setitem__(self, key, val):
        # note if val is not ATC, we make ts where cdim=ndim!
        # in other words, arrays and ATC's may be mixed, in which case
        # arrays are aligned to the cellular shape of the ATC's
        if hasattr(val, 'ts') and hasattr(val, 'view'):
            ts = val.ts
        elif isinstance(val, np.ndarray):
            ts = tashape.taShape(val.shape, val.ndim)
        else:
            raise ValueError('values in dbTable need to be array or ATC type')
        # determine the new master broadcast shape
        if self._ts is None:
            self._ts = ts
        else:
            new_ts, _ = self._ts.combine(ts)
            if new_ts is None:
                raise ValueError(
                    "refused to set incompatible shape %s into shape %s"
                    % (ts, self._ts))
            # keep the broadcasted shape as master
            self._ts = new_ts
        self._table[key] = val

    def __getitem__(self, key):
        return self._table.__getitem__(key)

    def __contains__(self, key):
        return self._table.__contains__(key)
