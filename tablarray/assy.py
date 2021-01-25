#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:44:05 2020

@author: chris
"""

import collections
import numpy as np

# internal imports
# from . import _tabstr
from . import taprint
from . import base
from . import ta


class Assembly(object):
    """
    Assembly
    --------
    dictionary of broadcast-able TablArray's

    E.g.::

        import numpy as np
        import tablarray as ta
        x = ta.TablArray(np.linspace(-2, 2, 4), 0)
        y = ta.TablArray(np.linspace(-1.5, 1.5, 4).reshape(4,1), 0)
        E = ta.zeros((4, 4, 2), cdim=1)
        E.cell[0] = 1.5 / (x**2 + y**2)
        E.cell[1] = -.01 + x * 0
        En = ta.abs(ta.linalg.norm(E)**2)
        Efield = ta.Assembly(x=x, y=y, E=E, En=En)
        Exslice = Efield['x', 'E', 'En', 2, :]

    Assembly[] allows multi-level indexing:
        keys select elements, numeric and slice indexing select within
        TablArray elements (using 'bcast' rules).

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

    def __init__(self, view='table', **kwargs):
        # this is used to track and check broadcastability
        self._ts = None
        # a facade for some dict methods of the table
        self._tablarrays = collections.OrderedDict()
        self.keys = self._tablarrays.keys
        self.items = self._tablarrays.items
        self.pop = self._tablarrays.pop
        self.setview(view)
        for key, val in kwargs.items():
            self[key] = val

    def _set_ts(self, new_ts):
        self._ts = new_ts
        # only allow one view
        view = self.view
        if view == 'table' or view == 'bcast':
            self._shape = self._ts.tshape
            self._ndim = self._ts.tdim
            self._size = self._ts.tsize
        elif view == 'cell':
            self._shape = self._ts.cshape
            self._ndim = self._ts.cdim
            self._size = self._ts.csize
        elif view == 'array':
            self._shape = (*self._ts.tshape, *self._ts.cshape)
            self._ndim = self._ts.tdim + self._ts.cdim
            self._size = self._ts.tsize * self._ts.csize
        else:
            raise ValueError

    def __setitem__(self, key, val):
        # note if val is not ATC, we make ts where cdim=ndim!
        # in other words, arrays and ATC's may be mixed, in which case
        # arrays are aligned to the cellular shape of the ATC's
        if isinstance(val, np.ndarray):
            val = ta.TablArray(val, val.ndim)
        elif np.isscalar(val):
            val = ta.TablArray(val, 0)
        if not base.istablarray(val):
            raise ValueError('values in Array need to be array or TablArray type')
        # determine the new master broadcast shape
        ts = val.ts
        if self._ts is None:
            self._ts = ts
        else:
            new_ts, _ = self._ts.combine(ts)
            if new_ts is None:
                raise ValueError(
                    "refused to set incompatible shape %s into shape %s"
                    % (ts, self._ts))
            # keep the broadcasted shape as master
            self._set_ts(new_ts)
        if type(key) is not str:
            raise ValueError('keys must be str type')
        self._tablarrays[key] = val.__view__(self.view)

    def __getitem__(self, args):
        # __getitem__ args are not a tuple if there's only one
        args = args if type(args) is tuple else (args,)
        # sort args into keys (str type) and indices, both are allowed!
        keys = []
        indices = []
        for arg in args:
            if type(arg) is str:
                keys.append(arg)
            else:
                indices.append(arg)
        # but if keys were not specified - get them all!
        if len(keys) == 0:
            keys = list(self.keys())
        # gather return arrays
        rarrays = []
        for key in keys:
            element = self._tablarrays.__getitem__(key)
            if len(indices) == 0:
                rarrays.append(element)
            else:
                rarrays.append((element.bcast.__getitem__(indices)))
        if len(rarrays) == 1:
            # if there's only one key, return the TablArray within
            rval = rarrays[0]
        else:
            # if there's multiple keys, return another Assembly (view)
            rval = Assembly()
            for i in range(len(keys)):
                rval[keys[i]] = rarrays[i]
        return rval

    def __contains__(self, key):
        return self._tablarrays.__contains__(key)

    __str__ = taprint.assembly2string

    def __view__(self, view):
        """"""
        new_view = Assembly(view=view)
        for key in self.keys():
            new_view[key] = self[key]
        return new_view

    def setview(self, view):
        """Set all elements to a view - 'bcast', 'table', 'cell' or 'array'

        see TablArray"""
        self.view = view
        # set all existing values to same view
        for key in self.keys():
            self[key].setview(view)

    @property
    def bcast(self):
        return self.__view__('bcast')

    @property
    def table(self):
        return self.__view__('table')

    @property
    def cell(self):
        return self.__view__('cell')

    @property
    def array(self):
        return self.__view__('array')