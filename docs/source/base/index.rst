Base API
========

Core base classes shared across polyview modules.

Design contract for every subclass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* ``__init__`` stores hyperparameters only — no data, no fitting.
* Every hyperparameter must be a keyword argument with a default.
* Learned attributes are written as ``attr_`` (trailing underscore), following sklearn convention, so ``check_is_fitted`` works.
* Views are always passed as ``list[np.ndarray]`` (or a ``MultiViewDataset``), *never* as a single array.

.. automodule:: polyview.base
   :members:
   :show-inheritance:
   :undoc-members:
