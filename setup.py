#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 13:31:40 2021

@author: chris
"""

import setuptools

# this is the right way to import version.py here:
main_ns = {}
fname_version = setuptools.convert_path("tablarray/version.py")
with open(fname_version) as vf:
    exec(vf.read(), main_ns)

setuptools.setup(
    name = 'tablarray',
    version = main_ns['__version__'],
    author = "Chris Cannon",
    author_email = 'chris.cannon.9001@gmail.com',
    description = 'Extend broadcasting rules of numpy to abstract tabular'
                    'shape of data from cellular shape',
    license = 'BSD',
    url = 'https://github.com/chriscannon9001/tablarray',
    packages=setuptools.find_packages(include=[
        'tablarray', 'tablarray.kwtools', 'tablarray.linalg',
        'tablarray.np2ta', 'tablarray.tests']),
    python_requires='>=3.2',
    install_requires=[
        'numpy',
        'attr'],
    tests_require=['pytest'])
