Introduction
============
When the best structure of data is any-dimensional cells arranged
in any-dimensional tables - tablarray provides fast numpy-like array
operations with broadcasting modified to handle both cellular-dimensions
and tabular-dimensions at once.

tablarray was originally developed to manage large numbers of optical modes.

This is a very short example:




.. code-block:: python

	import numpy as np
	import tablarray as ta
	a = ta.TablArray.from_tile(np.identity(2), (3, 1))
	print(ta.table(a).shape)

::

	(3, 1)

.. code-block:: python

	print(ta.cell(a).shape)

::

	(2, 2)

.. code-block:: python

	# selecting one corner of every cell
	print(ta.table(ta.cell(a)[0, 1])[:, :])

::

	[[|0.|]
	 [|0.|]
	 [|0.|]] t(3, 1)|c()

.. code-block:: python

	# align addition to one corner of every cell
	ta.table(ta.cell(a)[0, 1])[:, :] += [[4, 5, 6]]
	print(a)

::

	[[|[[1. 4.] |
	  | [0. 1.]]|]

	 [|[[1. 5.] |
	  | [0. 1.]]|]

	 [|[[1. 6.] |
	  | [0. 1.]]|]] t(3, 1)|c(2, 2)

Those '|' separate tabular vs cellular structure.

Installation
============
... coming soon...

Status
======
Alpha - some features are stable enough:

* Many features are implemented and not expected to change.
* A few of those features need further adaptation for certain cases.
* Some features are still missing.
* Packaging needs work.

Todo
====
* Add pretty strings to make TablArray objects more fun to interact with.
* Provide setup.py
* Add example documentation
* Add flattening features (ravel, flatiter, flatten)
* unittests
* benchmark, cythonize, re-benchmark
