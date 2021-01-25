#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Because I created special data structures, it's important to
make pretty string functions.

How does this work?
    1. Start, map 1-2 dims of table to table display.
    2. For each display cell:
        a. find method depending on type [int / float / array / tablarray]
            method(h,w / N,M) --> (h,w) for [min, neutral, max] cases
            these methods are just going to build [min, neutral, max] str
                and then count h,w
    3. Calculate sum(h) and sum(w) and select best case [min, neutral, max]
    4. Row, line, column loop:
        a. call .str for each method

Created on Fri Nov 29 17:07:33 2019

@author: chris
"""

from abc import ABC, abstractmethod
# import attr
import numpy as np

from . import base


def count_str(string):
    lines = string.split('\n')
    cols = len(lines)
    rows = 0
    for line in lines:
        rows = max(rows, len(line))
    return rows, cols


class _obj2str(ABC):
    # strings = ['', '', '']
    # rows = [0, 0, 0]
    # cols = [0, 0, 0]

    @staticmethod
    def is_handler(obj):
        """True/False - is this class the handler for obj"""
        return False

    @abstractmethod
    def __init__(self, obj, gl_mx_row=0, gl_mx_col=0):
        # e.g.
        # self._base = obj
        # self.strings = ['', '', '']
        # self._count_strings()
        pass

    def _count_strings(self):
        self.rows = [None] * 3
        self.cols = [None] * 3
        for i in range(3):
            rows, cols = count_str(self.strings[i])
            self.rows[i] = rows
            self.cols[i] = cols

    def __str__(self):
        return '"%s" | "%s" | "%s"' % tuple(self.strings)

    def __getitem__(self, indices):
        precision, j = indices if (type(indices) is tuple) else (1, indices)
        # print(precision, j)
        string = self.strings[precision]
        lines = string.split('\n')
        # print(lines, len(lines))
        if j >= len(lines):
            return ''
        return lines[j]


class _int2str(_obj2str):
    @staticmethod
    def is_handler(obj):
        """True/False - is this class the handler for obj"""
        return type(obj) is int

    def __init__(self, obj, gl_mx_row=0, gl_mx_col=0):
        # here are the 3 methods for int2str
        d = '%d' % obj
        e1 = '%0.4g' % obj
        e2 = '%0.8g' % obj
        # but it's hard to be sure where %d falls, so just sort the lengths
        sort = np.argsort([len(d), len(e1), len(e2)])
        tmp_strings = [d, e1, e2]
        # load those strings in order
        self.strings = [None] * 3
        for i in range(3):
            self.strings[i] = tmp_strings[sort[i]]
        # load a count of my strings
        self._count_strings()


class _float2str(_obj2str):
    @staticmethod
    def is_handler(obj):
        """True/False - is this class the handler for obj"""
        return type(obj) is float or np.isscalar(obj)

    def __init__(self, obj, gl_mx_row=0, gl_mx_col=0):
        # here are the 3 methods for int2str
        f1 = '%0.3f' % obj
        f2 = '%1.2e' % obj
        f3 = '%1.8e' % obj
        # but it's hard to be sure where %d falls, so just sort the lengths
        sort = np.argsort([len(f1), len(f2), len(f3)])
        tmp_strings = [f1, f2, f3]
        # load those strings in order
        self.strings = [None] * 3
        for i in range(3):
            self.strings[i] = tmp_strings[sort[i]]
        # load a count of my strings
        self._count_strings()


class _np2str(_obj2str):
    @staticmethod
    def is_handler(obj):
        """True/False - is this class the handler for obj"""
        return isinstance(obj, np.ndarray)

    def __init__(self, obj):
        poptions0 = np.get_printoptions()
        np.set_printoptions(precision=3, linewidth=12)
        np1 = '%s' % obj
        np.set_printoptions(precision=5, linewidth=16)
        np2 = '%s' % obj
        np.set_printoptions(precision=8, linewidth=20)
        np3 = '%s' % obj
        np.set_printoptions(**poptions0)
        self.strings = [np1, np2, np3]
        self._count_strings()


class _ta2str(_obj2str):
    @staticmethod
    def is_handler(obj):
        """True/False - is this class the handler for obj"""
        return base.istablarray(obj)

    def __init__(self, obj):
        string = obj.__str__()
        self.strings = [string, string, string]
        self._count_strings()


class _str2str(_obj2str):
    @staticmethod
    def is_handler(obj):
        """True/False - is this class the handler for obj"""
        return type(obj) is str

    def __init__(self, obj):
        self.strings = [None] * 3
        for i in range(3):
            self.strings[i] = obj
        self._count_strings()


class _none2str(_obj2str):
    @staticmethod
    def is_handler(obj):
        """True/False - is this class the handler for obj"""
        return obj is None

    def __init__(self, obj):
        self.strings = [''] * 3
        self._count_strings()


def cell2str(obj):
    for method in [_none2str, _str2str, _int2str, _float2str, _np2str,
                   _ta2str]:
        if method.is_handler(obj):
            return method(obj)
    raise TypeError('method for type %s not found' % type(obj))


def tabular_2str_flat(indices, header, table):
    """
    Convert table to pretty string using flat indices

    Parameters
    ----------
        indices: [index1, index2, index3, ...]
            list of indices of type int or tuple
        header: [key1, key2, key3, ...]
            list of header keys
        table: an indexable object
            e.g. cell1 = table[key1][index1], ...
            Where cell1 must be of type None, str, int, float, ndarray,
            (IndexError is treated as None)
        linewidth: int
            max linewidth
    """
    nhead = len(header)
    nidx = len(indices)
    # first pass, get string conversion objects, rows and cols
    # table_2str = [[None] * (nhead + 1)] * (nidx + 1)
    table_2str = []
    row_2str = []
    row_2str.append(cell2str(None))
    for j in range(nhead):
        key = header[j]
        row_2str.append(cell2str(key))
        # table_2str[0][j] = cell2str(key)
        # print(0, j, key, cell2str(key).strings[0])
    table_2str.append(row_2str)
    for i in range(nidx):
        row_2str = []
        index = indices[i]
        idx2str = cell2str(list(index).__str__() if (type(index) is tuple) else index)
        # table_2str[i + 1][0] = idx2str
        row_2str.append(idx2str)
        # print(i + 1, 0, index, idx2str.strings[0])
        for j in range(nhead):
            key = header[j]
            # print(key, index, len(index), table[key].ndim)
            # try:
            if ((not np.isscalar(index))
                and hasattr(table, 'view')
                and (table.view == 'table' or table.view == 'cell')
                and (len(index) > table[key].ndim)):
                kept_index = index[-table[key].ndim:]
                if len(kept_index) == 1:
                    kept_index = kept_index[0]
                drop_index = index[:len(index) - table[key].ndim]
                if np.allclose(drop_index, 0):
                    # print(key, index, kept_index, drop_index)
                    # here is a rule a bit between table and bcast
                    try:
                        val = table[key][kept_index]
                    except:
                        val = None
                else:
                    val = None
            else:
                try:
                    val = table[key][index]
                except:
                    val = None
            #except:
            #    val = None
            # cast the val into a string conversion object
            a2str = cell2str(val)
            # print(i + 1, j + 1, val, a2str.strings[0])
            # table_2str[i + 1][j + 1] = a2str
            row_2str.append(a2str)
        table_2str.append(row_2str)
    return table_2str


def digest_2str(tables_2str, linewidth=100):
    string = ''
    tlrows = len(tables_2str)
    ncol = len(tables_2str[0])
    rows = np.zeros((3, tlrows, ncol))
    cols = np.zeros((3, tlrows, ncol))
    # accumulate dimensions
    for i in range(tlrows):
        for j in range(ncol):
            for k in range(3):
                rows[k, i, j] = tables_2str[i][j].rows[k]
                cols[k, i, j] = tables_2str[i][j].cols[k]
    # decide on linewidth
    precision = None
    for k in reversed(range(3)):
        widths = rows[k].max(axis=0)
        # print('widths ', widths)
        mx_width = widths.sum() + len(widths) * 3
        # print('max width %d' % mx_width)
        if mx_width < linewidth:
            precision = k
            break
    assert precision is not None, "table is too wide"
    # now parse all the rows and setup the string
    my_format = ''
    underline = ''
    # print(widths)
    for width in widths:
        my_format += ' {:%d} |' % width
        underline += '-' * int(width + 2) + '+'
    for i in range(tlrows):
        nlines = int(cols[precision, i, :].max())
        # print('nlines', nlines)
        # print('nlines %d' % nlines)
        for line in range(nlines):
            row_args = []
            for j in range(ncol):
                str_ij = tables_2str[i][j][k, line]
                # print(i, j, str_ij)
                row_args.append(str_ij)
            row = my_format.format(*tuple(row_args))
            string += row + '\n'
        string += underline + '\n'
    return string





# --- old ----
'''

class _fstr(ABC):
    @staticmethod
    def is_handler(obj):
        """True/False - is this class the handler for obj"""
        return False

    @abstractmethod
    def __init__(self, obj, gl_mx_row=0, gl_mx_col=0):
        # e.g.
        # self._base = obj
        # self.rows = [1, 12]
        # self.cols = [1, 12]
        pass

    @abstractmethod
    def __call__(self, mx_row=0, mx_col=0):
        pass

    @staticmethod
    def _count_str(string):
        lines = string.split('\n')
        cols = len(lines)
        rows = 0
        for line in lines:
            rows = max(rows, len(line))
        return rows, cols

    def _load_strcounts(self, mn_string, mx_string):
        mn_rows, mn_cols = self._count_str(mn_string)
        mx_rows, mx_cols = self._count_str(mx_string)
        self.rows = [mn_rows, mx_rows]
        self.cols = [mn_cols, mx_cols]


class fstr_int(_fstr):
    @staticmethod
    def is_handler(obj):
        """True/False - is this class the handler for obj"""
        return type(obj) is int

    def __init__(self, obj, gl_mx_row=0, gl_mx_col=0):
        self._base = obj
        self._d = '%d' % obj
        self._e = '%1.1e' % obj
        rows_d, cols = self._count_str(self._d)
        assert cols == 1, "'%d' % int returned multiple lines?"
        rows_e, cols = self._count_str(self._e)
        assert cols == 1, "'%1.1e' % int returned multiple lines?"
        self._e_shorter = (rows_e < rows_d)
        # integers always display on one column
        self.cols = [1, 1]
        # record min and max number of rows
        if self._e_shorter:
            self.rows = [rows_e, rows_d]
        else:
            self.rows = [rows_d, rows_d]

    def __call__(self, mx_row=0, mx_col=0):
        if (mx_row < 1 or mx_col < 1) and not self._e_shorter:
            string = self._d
        else:
            del_rows = mx_row - len(self._e)
            if del_rows > 0:
                my_format = ('%1.' + ('%d' % (1+del_rows)) + 'e')
                string = my_format % self._base
            else:
                string = self._e
        return string


class fstr_tab(_fstr):
    @staticmethod
    def is_handler(obj):
        """True/False - is this class the handler for obj"""
        return hasattr(obj, 'ts') and hasattr(obj, 'view')

    def __init__(self, obj, gl_mx_row=0, gl_mx_col=0):
        self._base = obj.table
        ndim = max(2, obj._base.ndim)
        mydim = list(range(ndim))

    def __call__(self, mx_row=0, mx_col=0):
        pass


def get_fstr(obj):
    """"""
    all_fstr = [fstr_int, fstr_numpy, fstr_tab]
    for fstr in all_fstr:
        if fstr.is_handler(obj):
            return fstr(obj)


# =================


class _cellStr(ABC):
    base = None
    nlines = 0
    width = 0

    @abstractmethod
    def lines(self):
        # yield
        pass

    def _set_prop(self):
        self.form = ' {:^%d} |' % self.width
        self.underscore = '-' * (self.width + 2) + '+'
        self.dunderscore = '=' * (self.width + 2) + '+'


class CellStr(object):
    """obj in a table's cell needs methods for managing .__str__"""
    @staticmethod
    def _str_cell0(obj):
        """intelligently cast an object into a string"""
        if type(obj) is int:
            return '%d' % obj
        elif type(obj) is str:
            return ('%s' % obj)
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


@attr.s
class Table_Str(object):
    _table = attr.ib()
    _max_row = attr.ib(40, type=int)
    _max_col = attr.ib(10, type=int)

    def __attrs_post_init__(self):
        # setup loop structure for table
        if not hasattr(self._table, 'bcast'):
            # TablArray's bcast is useful for row,col display
            self._table = ta.TablArray(self._table, 0)
        shape = self._table.bcast.shape
        nrows = shape[-2] if len(shape) >= 2 else 1
        rows = list(range(nrows))
        if nrows < self._max_row:
            self._rows = rows
        else:
            self._rows = rows[0:(self._max_row-2)] + [None] + rows[-1:]
        ncol = shape[-1] if len(shape) >= 1 else 1
        cols = list(range(ncol))
        if ncol < self._max_col:
            self._cols = cols
        else:
            self._cols = cols[0:(self._max_col-2)] + [None] + cols[-1:]

    def get_row(self, row):
        new_row = []
        for col in self._cols:
            if (row is None or col is None):
                val = '...'
            else:
                val = self._table.bcast[row, col].base
            new_row.append(val)
        return new_row

    def get_format(self):
        my_format = ''
        seperator = []
        nrows = len(self._rows)
        ncol = len(self._cols)
        nlines = np.ones((nrows, ncol), dtype=int)
        widths = 5 * np.ones((nrows, ncol), dtype=int)
        for i in range(nrows):
            row = self.get_row(i)
            for j, cell0 in enumerate(row):
                cell1 = CellStr(cell0)
                widths[i, j] = cell1.width
                nlines[i, j] = cell1.nlines
        widths2 = np.max(widths, axis=0)
        nlines2 = np.max(nlines)
        for width in widths2:
            my_format += ' {:^%d} |' % width
            seperator.append('-' * width)
        return my_format, seperator, nlines2

    def __call__(self):
        """A tabular pretty string display"""
        # make note of the current numpy display precision
        precision0 = np.get_printoptions()['precision']
        # I want to reduce precision of numpy array display
        np.set_printoptions(precision=2)
        string = ''
        my_format, seperator, n_line = self.get_format()
        # using str.format(), my_format defines columns
        # setup the format
        for row in self._rows:
            row0 = self.get_row(row)
            for line in range(n_line):
                row_list = []
                for cell0 in row0:
                    cell = CellStr(cell0)
                    row_list.append(cell.str_cell(line))
                # print(my_format)
                # print(row_list)
                row_str = my_format.format(*tuple(row_list))
                string += row_str + '\n'
            string += my_format.format(*tuple(seperator)) + '\n'
        # reset numpy display precision
        np.set_printoptions(precision=precision0)
        return string


@attr.s
class DB_Str(object):
    _db = attr.ib(type=dict)
    _max_lines = attr.ib(60, type=int)
    _headerkeys = attr.ib(None, type=list)

    def __attrs_post_init__(self):
        # grab some key, val
        key = list(self._db.keys())[0]
        val = self._db[key]
        # then set isscalar and length
        self._isscalar = ta.shape(val) == ()
        self._length = 1 if self._isscalar else len(val)

    # some helper functions
    def get_header(self):
        """header can be limited by set_str_keys"""
        # allow str_keys to control columns of __str__()
        keys = self._headerkeys
        if keys is None:
            header = list(self._db.keys())
            excluded = None
        else:
            header = []
            for key in keys:
                if key in self._db:
                    # any keys not in self will just be ignored
                    header.append(key)
            excluded = []
        return header, excluded

    def get_row(self, header, line):
        row = []
        for col in header:
            val = (self._db[col] if self._isscalar
                   else self._db[col][line])
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
                cell1 = CellStr(cell0)
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
                        cell = CellStr(cell0)
                        row_list.append(cell.str_cell(line))
                    # print(my_format)
                    # print(row_list)
                    row_str = my_format.format(*tuple(row_list))
                    string += row_str + '\n'
        # reset numpy display precision
        np.set_printoptions(precision=precision0)
        return string
'''
