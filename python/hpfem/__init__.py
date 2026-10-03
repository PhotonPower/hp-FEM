"""hpfem — adaptive hp-FEM for nano-optics.

Public Python API over the C++ core (``hpfem._hpfem``). The core is strictly SI with the
time dependence ``exp(-i omega t)`` (CLAUDE.md §6); :mod:`hpfem.units` converts nm, µm,
eV, THz and friends at the boundary. Dimension-templated classes exist as ``<Name>2D`` /
``<Name>3D`` (``Mesh2D``, ``NedelecDofMap3D``, ``Scattering2D``, …); free functions are
overloaded on the argument type, so ``refine_uniform(mesh)`` or ``extract(mesh, cells)``
work for either dimension.

Typical use::

    import hpfem
    mesh = hpfem.square_with_disc(4, radius=0.25, half_width=1.0, outer=2.0)
    dofs = hpfem.NedelecDofMap2D(mesh, 3)
    setup = hpfem.ScatteringSetup2D()
    setup.omega = 6.0 * hpfem.constants.c0
    setup.materials.set(2, hpfem.Material.dielectric(1.5))
    setup.incident = hpfem.plane_wave([0.0, 1.0], [6.0, 0.0])
    setup.formulation = hpfem.Formulation.SCATTERED_FIELD
    setup.pml = hpfem.PmlBox2D.uniform([-1, -1], [1, 1], 1.0, 6.0)
    setup.pec_tags = [hpfem.box_tag.X_MIN, hpfem.box_tag.X_MAX,
                      hpfem.box_tag.Y_MIN, hpfem.box_tag.Y_MAX]
    problem = hpfem.Scattering2D(dofs, setup)
    solution = problem.solve()
    cs = hpfem.cross_sections(problem, solution, hpfem.Surface2D.around_cells(mesh, 2), 1.0)
"""

from hpfem import (  # noqa: E402  (after the core names)
    _hpfem,
    interop,
    materials,
    project,
    pv,
    units,
)
from hpfem._hpfem import *  # noqa: F401, F403  (the bound API)
from hpfem._hpfem import version  # noqa: E402

__all__ = [name for name in dir(_hpfem) if not name.startswith("_")] + [
    "units",
    "materials",
    "project",
    "interop",
    "pv",
]
__version__ = version()
