Kernel utilities
================

Helpers to build, check and precondition kernel matrices.
:class:`~polyview.utils.kernels.KernelSpec` describes how one view is turned
into a kernel — linear, RBF, polynomial, or an already precomputed matrix —
and is what :class:`~polyview.fusion.kernel_fusion.KernelFusion` consumes per
view.

The three functions are standalone: ``center_kernel`` removes the mean in the
reproducing-kernel Hilbert space, ``normalize_kernel`` rescales so that
``K[i, i] = 1``, and ``is_valid_kernel`` checks symmetry and positive
semi-definiteness, which is worth doing before trusting a precomputed matrix.

These names are also re-exported from :mod:`polyview.fusion` and from the
package root.

.. automodule:: polyview.utils.kernels
   :members:
   :show-inheritance:
   :undoc-members:
