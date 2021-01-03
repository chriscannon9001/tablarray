#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 10:50:40 2021

@author: chris
"""

def bcast(obj):
    """get view of obj with broadcast-style tabular indexing, if available,
    else return obj"""
    if hasattr(obj, 'ts') and hasattr(obj, '__view__'):
        return obj.__view__('bcast')
    return obj


def cell(obj):
    """get cellular view of obj, if available, else return obj"""
    if hasattr(obj, 'ts') and hasattr(obj, '__view__'):
        return obj.__view__('cell')
    return obj


def table(obj):
    """get tabular view of obj, if available, else return obj"""
    if hasattr(obj, 'ts') and hasattr(obj, '__view__'):
        return obj.__view__('table')
    return obj


def array(obj):
    """get simple array view of obj, if available, else return obj"""
    if hasattr(obj, 'ts') and hasattr(obj, '__view__'):
        return obj.__view__('array')
    return obj
