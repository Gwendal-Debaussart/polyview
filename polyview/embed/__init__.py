"""
Multiview embedding methods.
"""

from .gcca import GCCA
from .mcca import MCCA
from .multiviewmds import MVMDS, MultiViewMDS

__all__ = ["GCCA", "MCCA", "MVMDS", "MultiViewMDS"]
