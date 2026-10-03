"""Writes representative hp-FEM system matrices as Matrix Market files for the GPU
micro-benchmark (`spmv_solve_bench.cu`): the Newmark operator of the transient solver on
the unit square (symmetric positive definite, complex storage) and the time-harmonic
scattering operator S - k0^2 M with a PML on the unit cube (complex symmetric, indefinite).

Usage: python export_matrices.py <out_dir>
"""

import sys
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse

import hpfem

c0 = hpfem.constants.c0


def newmark_2d(n, p):
    mesh = hpfem.rectangle(n, n)
    nd = hpfem.NedelecDofMap2D(mesh, p)
    omega = c0 * np.pi * np.sqrt(2)
    setup = hpfem.TimeDomainSetup2D()
    setup.pec_tags = [
        hpfem.box_tag.X_MIN,
        hpfem.box_tag.X_MAX,
        hpfem.box_tag.Y_MIN,
        hpfem.box_tag.Y_MAX,
    ]
    setup.dt = 2 * np.pi / omega / 40
    problem = hpfem.TimeDomain2D(nd, setup)
    dt = setup.dt
    k = problem.mass + 0.5 * dt * problem.damping + 0.25 * dt * dt * problem.stiffness
    return scipy.sparse.csr_matrix(k)


def scattering_3d(n, p):
    mesh = hpfem.box(n, n, n)
    nd = hpfem.NedelecDofMap3D(mesh, p)
    setup = hpfem.ScatteringSetup3D()
    k0 = 2 * np.pi / 0.5
    setup.omega = k0 * c0
    setup.incident = hpfem.plane_wave(np.array([0, 1, 0], dtype=complex), np.array([k0, 0.0, 0.0]))
    setup.pml = hpfem.PmlBox3D.uniform([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.2, k0)
    problem = hpfem.Scattering3D(nd, setup)
    system = hpfem.assemble_maxwell_operator(nd, problem.form_of_cell, k0 * k0)
    return scipy.sparse.csr_matrix(system.matrix)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    cases = [
        ("newmark2d_n128_p2", lambda: newmark_2d(128, 2)),
        ("newmark2d_n256_p3", lambda: newmark_2d(256, 3)),
        ("scattering3d_n12_p2", lambda: scattering_3d(12, 2)),
    ]
    for name, build in cases:
        matrix = build()
        path = out / f"{name}.mtx"
        scipy.io.mmwrite(path, matrix, field="complex", precision=17)
        print(f"{path}: {matrix.shape[0]} rows, {matrix.nnz} nonzeros", flush=True)


if __name__ == "__main__":
    main()
