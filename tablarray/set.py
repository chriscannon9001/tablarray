#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:44:05 2020

@author: chris
"""

import collections
import functools
import logging
import numpy as np

# internal imports
# from . import _tabstr
from . import taprint
from . import misc
from . import ta
from . import re


class TablaSet(object):
    """
    TablaSet
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
        Efield = ta.TablaSet(x=x, y=y, E=E, En=En)
        Exslice = Efield['x', 'E', 'En', 2, :]

    TablaSet[] allows multi-level indexing:
        keys select elements, numeric and slice indexing select within
        TablArray elements (using 'bcast' rules).

    Parameters
    ----------
    **kwargs: keyword=TablArray, ..
        name1=tablarray1, name2=tablarray2, ...

    With each additional element, broadcast compatibility is enforced::

        s1 = ta.TablaSet()
        s1['x'] = ta.TablArray(np.linspace(-1, 3, 5), 0)
        s1['y'] = ta.TablArray([0, 0, 0], 0)
        >>> ValueError: refused to load incompatible shape t(3,)|c() into
        shape t(5,)|c()
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
        if not misc.istablarray(val):
            raise ValueError('values in Array need to be array or TablArray'
                             'type')
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
            # if there's multiple keys, return another TablaSet (view)
            rval = TablaSet()
            for i in range(len(keys)):
                rval[keys[i]] = rarrays[i]
        return rval

    def __contains__(self, key):
        return self._tablarrays.__contains__(key)

    __str__ = taprint.tablaset2string

    def __view__(self, view):
        """"""
        new_view = TablaSet(view=view)
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

    def meshtile(self, key):
        """Tile an element so that it matches the maximum table
        shape of the TablaSet, same as it appears when printing a bcast view.
        Often this is the preferred alternate to meshgrid.

        This can be useful e.g. for plotting, while normally bcast view is
        preferred for writing formulas (for memory consumption)."""
        bcast_tshape = self._ts.tshape
        array = self[key]
        tshape = array.ts.tshape
        if bcast_tshape == tshape:
            # skip the rest
            return array
        # determine the number of repetitions along each axis
        tshape2 = np.ones(self._ts.tdim)
        tshape2[-array.ts.tdim:] = tshape
        repsArr = np.array(bcast_tshape)
        repsArr[np.equal(tshape2, bcast_tshape)] = 1
        reps = tuple(repsArr)
        return re.tile(array.table, reps)

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


def _survey_view(args, kwargs):
    """find the most popular view of TablArrays in args and kwargs"""
    views = ['table', 'bcast', 'cell', 'array']
    idx = dict(zip(views, range(4)))
    counts = np.zeros(4, dtype=int)
    for arg in args:
        if misc.istablarray(arg):
            view = arg.view
            counts[idx[view]] += 1
    for _, arg in kwargs.items():
        if misc.istablarray(arg):
            view = arg.view
            counts[idx[view]] += 1
    print('counts %s' % counts)
    popular_idx = np.argsort(counts)[-1]
    return views[popular_idx]


def tablaset_args(setargs=(), setkwargs=(), argview='cell',
                  rvalview='ignore', debugging=False):
    """
    Decorator tablaset_args
    =======================
    Give clients of your func the option of covering args using a TablaSet.

    Also handles pass-through to original func(*args, **kwargs)

    Parameters
    ----------
    setargs: tuple of str ('arg1', 'arg2', ...)
        Names of elements that can be taken from a TablaSet and passed
        as *args in this order.
    setkwargs: tuple of str ('karg1', 'kwarg2', ...)
        Names of elements that can be taken from a TablaSet and passed
        as **kwargs.
    [argview]: str 'ignore', 'cell' (default), 'table', 'bcast', or 'array'
        TablArray or TablaSet inputs will be cast to this view. See TablArray
    [rvalview]: str 'preserve', 'ignore' (default), 'cell', 'table', 'bcast',
        or 'array'
        Return values can be set to a view
    [debugging]: bool (default False)
        Activate logging.debug messages
    """
    if len(setargs) < 1 and len(setkwargs) < 1:
        raise ValueError('need to spec args or kwargs')
    for arg in setargs:
        if type(arg) is not str:
            raise TypeError('got type %s instead of str' % type(arg))
    for arg in setkwargs:
        if type(arg) is not str:
            raise TypeError('got type %s instead of str' % type(arg))
    if argview not in ['ignore', 'cell', 'table', 'bcast', 'array']:
        raise ValueError('unrecognized argview "%s"' % argview)
    if rvalview not in ['preserve', 'ignore', 'cell', 'table', 'bcast',
                        'array']:
        raise ValueError('unrecognized argview "%s"' % argview)

    def _viewargs(args):
        """for each TablArray in args, force using the expected argview"""
        if argview == 'ignore':
            return args
        args2 = []
        for arg in args:
            arg2 = arg.__view__(argview) if misc.istablarray(arg) else arg
            args2.append(arg2)
        return tuple(args2)

    def _viewkwargs(kwargs):
        """for each TablArray in kwargs, force using the expected argview"""
        if argview == 'ignore':
            return kwargs
        kwargs2 = {}
        for key, item in kwargs.items():
            item2 = item.__view__(argview) if misc.istablarray(item) else item
            kwargs2[key] = item2
        return kwargs2

    def args_assembler_decorator(func):
        if debugging:
            log = logging.getLogger(__name__ + ': ' + func.__name__)

        @functools.wraps(func)
        def args_assembler_wrapper(*args, **kwargs):
            if misc.istablaset(args[0]):
                if debugging:
                    log.debug('client gave a TablaSet to load args')
                    if len(args) > 1:
                        log.warning('ignoring %d args from client',
                                    len(args[1:]))
                set1 = args[0]
                keys = set1.keys()
                # get args1 from set1
                args1 = []
                for setarg in setargs:
                    if setarg in keys:
                        args1.append(set1[setarg])
                    elif debugging:
                        log.debug('element "%s" not found in TablaSet',
                                  setarg)
                # get kwargs1 from set1
                kwargs1 = {}
                for setkwarg in setkwargs:
                    if setkwarg in keys:
                        kwargs1[setkwarg] = set1[setkwarg]
                    elif debugging:
                        log.debug('element "%s" not found in TablaSet',
                                  setkwarg)
                # combine kwargs from the line
                if debugging and len(kwargs) > 0:
                    log.debug('combining keywords %s from client',
                              list(kwargs.keys()))
                kwargs1.update(kwargs)
            else:
                if debugging:
                    log.debug("fall back on func(*args, **kwargs)")
                args1 = args
                kwargs1 = kwargs
            # finished mangling args and kwargs, now call func
            rvals = func(*_viewargs(args1), **_viewkwargs(kwargs1))
            if rvalview == 'ignore':
                if debugging:
                    log.debug('ignoring rval views')
                return rvals
            # the view of all TablArray returns will be by survey or constant
            rview = (_survey_view(args1, kwargs1) if rvalview == 'preserve'
                     else rvalview)
            if debugging:
                log.debug('set all views to "%s"', rview)
            # find TablArray returns and force the view
            rvals = [rvals] if type(rvals) is not tuple else rvals
            rvals2 = []
            for rval in rvals:
                rval2 = (rval.__view__(rview) if misc.istablarray(rval)
                         else rval)
                if debugging:
                    log.debug('.view %s to %s', rval.view, rval2.view)
                rvals2.append(rval2)
            return rvals[0] if len(rvals) == 1 else tuple(rvals)
        return args_assembler_wrapper
    return args_assembler_decorator
