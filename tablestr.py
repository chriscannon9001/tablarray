#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:07:33 2019

@author: chris
"""

import attr
import numpy as np


class TableCell(object):
    """obj in a table's cell needs methods for managing .__str__"""
    @staticmethod
    def _str_cell0(obj):
        """intelligently cast an object into a string"""
        if type(obj) is int:
            return '%d' % obj
        elif np.isscalar(obj):
            return '%0.4g' % obj
        else:
            return ('%s' % obj)

    def __init__(self, obj):
        self._str = self._str_cell0(obj)
        self._str_split = self._str.split('\n')
        self.nlines = len(self._str_split)
        self.width = self._get_width()

    def str_cell(self, line):
        """if a string has multi lines, slice line-by-line"""
        s2 = self._str_split
        return s2[line] if line < len(s2) else ''

    def _get_width(self):
        """character width of a cell needed for obj"""
        width = 6
        for i in range(self.nlines):
            s2 = self._str_split[i]
            width = max(width, len(s2))
        return width


def _get_shape(obj):
    """"""
    return obj.shape if isinstance(obj, np.ndarray) else ()


@attr.s
class TableStr(object):
    _table = attr.ib(type=dict)
    _max_lines = attr.ib(60, type=int)
    _headerkeys = attr.ib(None, type=list)

    def __attrs_post_init__(self):
        # grab some key, val
        key = list(self._table.keys())[0]
        val = self._table[key]
        # then set isscalar and length
        self._isscalar = _get_shape(val) == ()
        self._length = 1 if self._isscalar else len(val)

    # some helper functions
    def get_header(self):
        """header can be limited by set_str_keys"""
        # allow str_keys to control columns of __str__()
        keys = self._headerkeys
        if keys is None:
            header = list(self._table.keys())
            excluded = None
        else:
            header = []
            for key in keys:
                if key in self._table:
                    # any keys not in self will just be ignored
                    header.append(key)
            excluded = []
        return header, excluded

    def get_row(self, header, line):
        row = []
        for col in header:
            val = (self._table[col] if self._isscalar
                   else self._table[col][line])
            row.append(val)
        return row

    def get_format(self, header):
        my_format = '' if self._isscalar else '{:<5}'
        underline = [] if self._isscalar else ['='*5]
        nscan = min(20, self._length)
        ncol = len(header)
        nlines = np.ones((nscan, ncol), dtype=int)
        widths = 5 * np.ones((nscan, ncol), dtype=int)
        for line in range(nscan):
            row = self.get_row(header, line)
            for i, cell0 in enumerate(row):
                cell1 = TableCell(cell0)
                widths[line, i] = cell1.width
                nlines[line, i] = cell1.nlines
        widths2 = np.max(widths, axis=0)
        nlines2 = np.max(nlines)
        for width in widths2:
            my_format += '|{:^%d}' % width
            underline.append('=' * width)
        return my_format, underline, nlines2

    def __call__(self):
        """A tabular pretty string display"""
        # make note of the current numpy display precision
        precision0 = np.get_printoptions()['precision']
        # I want to reduce precision of numpy array display
        np.set_printoptions(precision=2)
        string = ''
        header, excluded = self.get_header()
        if excluded is not None:
            string += 'Excluded: %s\n' % excluded
        my_format, underline, n_line = self.get_format(header)
        # using str.format(), my_format defines columns
        # setup the format
        header1 = header if self._isscalar else [''] + header
        header2 = tuple(header1)
        string += my_format.format(*header2) + '\n'
        string += my_format.format(*underline) + '\n'
        rows = list(range(self._length))
        if self._length > 60:
            rows = rows[0:30] + [None] + rows[-30:]
        for row in rows:
            if row is None:
                row_list = tuple(['...'] * len(header2))
                string += my_format.format(*row_list) + '\n'
            else:
                row0 = self.get_row(header, row)
                for line in range(n_line):
                    col0 = '%d' % row if (line == 0) else ''
                    row_list = [] if self._isscalar else [col0]
                    for cell0 in row0:
                        cell = TableCell(cell0)
                        row_list.append(cell.str_cell(line))
                    # print(my_format)
                    # print(row_list)
                    row_str = my_format.format(*tuple(row_list))
                    string += row_str + '\n'
        # reset numpy display precision
        np.set_printoptions(precision=precision0)
        return string
