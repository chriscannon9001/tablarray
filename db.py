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


class dbTA(object):
    """databaseTable - a dict of broadcastable ATC's (Array/TableCell's)"""
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
            **kwargs : dict of args
                name1=atc1, name2=atc2, ...
        """
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
            assert new_ts is not None, (
                    "refused to load incompatible shape %s into dbATC shape %s"
                    % (ts, self._ts))
            # keep the broadcasted shape as master
            self._ts = new_ts
        self._table[key] = val

    def __getitem__(self, key):
        return self._table.__getitem__(key)

    def __contains__(self, key):
        return self._table.__contains__(key)
