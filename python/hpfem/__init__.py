"""hpfem — adaptive hp-FEM for nano-optics.

Public Python API. Unit conversion (nm, µm, eV, THz → SI) happens in this layer;
the C++ core is strictly SI (see CLAUDE.md §6).
"""

from hpfem._hpfem import version

__all__ = ["version"]
__version__ = version()
