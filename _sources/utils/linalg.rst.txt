Linear-algebra utilities
========================

The truncated symmetric eigensolvers shared by the spectral clustering and
kernel methods. ``truncated_eigh`` returns the leading eigenpairs of a
symmetric matrix, dispatching between a dense solver and ARPACK's Lanczos
iteration — with ``eigen_solver="auto"`` following the same heuristic as
scikit-learn's :class:`~sklearn.decomposition.KernelPCA`, since Lanczos only
pays off when few eigenpairs of a large matrix are needed.

.. automodule:: polyview.utils.linalg
   :members:
   :show-inheritance:
   :undoc-members:
