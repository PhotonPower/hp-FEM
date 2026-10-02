/// Bindings of `assembly`: scalar (H1) and Maxwell forms with coefficient callbacks, global
/// assembly to SciPy sparse matrices, Dirichlet data and elimination, interpolation,
/// evaluation and error norms of discrete fields, prolongation between meshes, the discrete
/// gradient and point-sampled functionals.
#include <functional>
#include <optional>
#include <vector>

#include "common.hpp"
#include "hpfem/assembly/dirichlet.hpp"
#include "hpfem/assembly/discrete_gradient.hpp"
#include "hpfem/assembly/functionals.hpp"
#include "hpfem/assembly/h1_forms.hpp"
#include "hpfem/assembly/interpolation.hpp"
#include "hpfem/assembly/maxwell_forms.hpp"
#include "hpfem/assembly/prolongation.hpp"
#include "hpfem/mesh/point_location.hpp"

namespace hpfem::python {

namespace {

template <int Dim>
void bind_assembly_dim(py::module_& m) {
  using H1 = fespace::DofMap<Dim>;
  using ND = fespace::NedelecDofMap<Dim>;
  using assembly::ComplexCurl;
  using assembly::ComplexVector;
  using assembly::MaxwellForm;
  using assembly::ScalarForm;

  // --- forms ----------------------------------------------------------------------------
  py::class_<ScalarForm<Dim>>(m, named("ScalarForm", Dim).c_str(),
                              "Coefficients of a(u, v) = (alpha grad u, grad v) + (beta u, v), "
                              "l(v) = (f, v) as callables of the point; None means zero")
      .def(py::init([](std::function<Complex(const Point<Dim>&)> diffusion,
                       std::function<Complex(const Point<Dim>&)> reaction,
                       std::function<Complex(const Point<Dim>&)> source) {
             ScalarForm<Dim> f;
             f.diffusion = std::move(diffusion);
             f.reaction = std::move(reaction);
             f.source = std::move(source);
             return f;
           }),
           py::arg("diffusion") = py::none(), py::arg("reaction") = py::none(),
           py::arg("source") = py::none())
      .def_readwrite("diffusion", &ScalarForm<Dim>::diffusion)
      .def_readwrite("reaction", &ScalarForm<Dim>::reaction)
      .def_readwrite("source", &ScalarForm<Dim>::source);

  py::class_<MaxwellForm<Dim>>(
      m, named("MaxwellForm", Dim).c_str(),
      "Coefficients of the Maxwell forms: inverse_permeability(x) (tensor acting on curls: "
      "1x1 in 2D, 3x3 in 3D), permittivity(x) (dim x dim), source f(x) (vector) and "
      "curl_source g(x) paired with curl v; None means identity (tensors) or zero (sources). "
      "quadrature_order overrides the rule degree of the cell.")
      .def(
          py::init(
              [](assembly::CurlTensorField<Dim> inverse_permeability,
                 assembly::TensorField<Dim> permittivity, assembly::ComplexVectorField<Dim> source,
                 assembly::ComplexCurlField<Dim> curl_source, std::optional<int> quadrature_order) {
                MaxwellForm<Dim> f;
                f.inverse_permeability = std::move(inverse_permeability);
                f.permittivity = std::move(permittivity);
                f.source = std::move(source);
                f.curl_source = std::move(curl_source);
                f.quadrature_order = quadrature_order;
                return f;
              }),
          py::arg("inverse_permeability") = py::none(), py::arg("permittivity") = py::none(),
          py::arg("source") = py::none(), py::arg("curl_source") = py::none(),
          py::arg("quadrature_order") = py::none())
      .def_readwrite("inverse_permeability", &MaxwellForm<Dim>::inverse_permeability)
      .def_readwrite("permittivity", &MaxwellForm<Dim>::permittivity)
      .def_readwrite("source", &MaxwellForm<Dim>::source)
      .def_readwrite("curl_source", &MaxwellForm<Dim>::curl_source)
      .def_readwrite("quadrature_order", &MaxwellForm<Dim>::quadrature_order);

  // --- global assembly ------------------------------------------------------------------
  m.def(
      "assemble_h1",
      [](const H1& dofs, const ScalarForm<Dim>& form, int extra_order) {
        return assembly::assemble_h1<Dim>(dofs, form, extra_order);
      },
      py::arg("dofs"), py::arg("form"), py::arg("extra_order") = 2, Release(),
      "Matrix and load of the scalar form, rules exact for degree 2p + extra_order");
  m.def(
      "assemble_h1",
      [](const H1& dofs, const std::function<ScalarForm<Dim>(Index)>& form_of_cell,
         int extra_order) { return assembly::assemble_h1<Dim>(dofs, form_of_cell, extra_order); },
      py::arg("dofs"), py::arg("form_of_cell"), py::arg("extra_order") = 2, Release(),
      "As above with a form per cell, form_of_cell(c)");
  m.def(
      "assemble_maxwell",
      [](const ND& dofs, const MaxwellForm<Dim>& form, int extra_order) {
        return assembly::assemble_maxwell<Dim>(dofs, form, extra_order);
      },
      py::arg("dofs"), py::arg("form"), py::arg("extra_order") = 2, Release(),
      "Stiffness S (weight mu^-1), mass M (weight eps) and load b of the Maxwell forms");
  m.def(
      "assemble_maxwell",
      [](const ND& dofs, const std::function<MaxwellForm<Dim>(Index)>& form_of_cell,
         int extra_order) {
        return assembly::assemble_maxwell<Dim>(dofs, form_of_cell, extra_order);
      },
      py::arg("dofs"), py::arg("form_of_cell"), py::arg("extra_order") = 2, Release(),
      "As above with a form per cell, form_of_cell(c)");
  m.def(
      "assemble_maxwell_operator",
      [](const ND& dofs, const std::function<MaxwellForm<Dim>(Index)>& form_of_cell, Real k_squared,
         int extra_order) {
        return assembly::assemble_maxwell_operator<Dim>(dofs, form_of_cell, k_squared, extra_order);
      },
      py::arg("dofs"), py::arg("form_of_cell"), py::arg("k_squared"), py::arg("extra_order") = 2,
      Release(), "A = S - k^2 M and the load in one pass");
  m.def("discrete_gradient", &assembly::discrete_gradient<Dim>, py::arg("h1"), py::arg("nedelec"),
        Release(), "Sparse G (N_ND x N_H1) with grad(sum u_i phi_i) = sum (G u)_j phi_j exactly");

  // --- Dirichlet data -------------------------------------------------------------------
  m.def(
      "dirichlet_values",
      [](const H1& dofs, mesh::Tag tag, const assembly::ScalarField<Dim>& g) {
        return assembly::dirichlet_values<Dim>(dofs, tag, g);
      },
      py::arg("dofs"), py::arg("tag"), py::arg("g"), Release(),
      "Boundary values of g on the facets carrying the tag (hierarchical interpolation)");
  m.def(
      "dirichlet_values",
      [](const H1& dofs, const std::vector<Index>& facets, const assembly::ScalarField<Dim>& g) {
        return assembly::dirichlet_values<Dim>(dofs, facets, g);
      },
      py::arg("dofs"), py::arg("facets"), py::arg("g"), Release());
  m.def(
      "tangential_dirichlet_values",
      [](const ND& dofs, mesh::Tag tag, const assembly::ComplexVectorField<Dim>& g) {
        return assembly::tangential_dirichlet_values<Dim>(dofs, tag, g);
      },
      py::arg("dofs"), py::arg("tag"), py::arg("g"), Release(),
      "Prescribed tangential trace n x E = n x g on the facets carrying the tag");
  m.def(
      "tangential_dirichlet_values",
      [](const ND& dofs, const std::vector<Index>& facets,
         const assembly::ComplexVectorField<Dim>& g) {
        return assembly::tangential_dirichlet_values<Dim>(dofs, facets, g);
      },
      py::arg("dofs"), py::arg("facets"), py::arg("g"), Release());
  m.def(
      "homogeneous_dirichlet",
      [](const H1& dofs, const std::vector<Index>& facets) {
        return assembly::homogeneous_dirichlet(dofs, facets);
      },
      py::arg("dofs"), py::arg("facets"), "All DoFs on the facets with value 0");
  m.def(
      "homogeneous_dirichlet",
      [](const ND& dofs, const std::vector<Index>& facets) {
        return assembly::homogeneous_dirichlet(dofs, facets);
      },
      py::arg("dofs"), py::arg("facets"), "PEC: all DoFs on the facets with value 0");

  // --- interpolation, evaluation, errors ------------------------------------------------
  m.def(
      "interpolate",
      [](const H1& dofs, const std::function<Complex(const Point<Dim>&)>& g) {
        return assembly::interpolate<Dim>(dofs, assembly::physical_sampler<Dim>(g));
      },
      py::arg("dofs"), py::arg("g"), Release(),
      "Hierarchical interpolant of g(x) on the whole mesh as a coefficient vector");
  m.def(
      "interpolate",
      [](const ND& dofs, const assembly::ComplexVectorField<Dim>& g) {
        return assembly::interpolate<Dim>(dofs, assembly::physical_sampler<Dim>(g));
      },
      py::arg("dofs"), py::arg("g"), Release(),
      "Hierarchical H(curl) interpolant of the vector field g(x)");
  m.def(
      "evaluate_h1",
      [](const H1& dofs, const Vector& u, Index c, const Point<Dim>& xi) {
        return assembly::evaluate_h1<Dim>(dofs, u, c, xi);
      },
      py::arg("dofs"), py::arg("u"), py::arg("cell"), py::arg("xi"),
      "Value at reference point xi of a cell");
  m.def(
      "evaluate_h1",
      [](const H1& dofs, const Vector& u, const mesh::PointLocator<Dim>& locator,
         const Point<Dim>& x) { return assembly::evaluate_h1<Dim>(dofs, u, locator, x); },
      py::arg("dofs"), py::arg("u"), py::arg("locator"), py::arg("x"),
      "Value at a physical point, None outside the mesh");
  m.def(
      "evaluate_hcurl",
      [](const ND& dofs, const Vector& e, Index c, const Point<Dim>& xi) {
        return assembly::evaluate_hcurl<Dim>(dofs, e, c, xi);
      },
      py::arg("dofs"), py::arg("e"), py::arg("cell"), py::arg("xi"),
      "Physical field value at reference point xi of a cell");
  m.def(
      "evaluate_hcurl",
      [](const ND& dofs, const Vector& e, const mesh::PointLocator<Dim>& locator,
         const Point<Dim>& x) { return assembly::evaluate_hcurl<Dim>(dofs, e, locator, x); },
      py::arg("dofs"), py::arg("e"), py::arg("locator"), py::arg("x"),
      "Field value at a physical point, None outside the mesh");
  m.def(
      "evaluate_hcurl_curl",
      [](const ND& dofs, const Vector& e, Index c, const Point<Dim>& xi) {
        return assembly::evaluate_hcurl_curl<Dim>(dofs, e, c, xi);
      },
      py::arg("dofs"), py::arg("e"), py::arg("cell"), py::arg("xi"),
      "Physical curl (scalar in 2D) at reference point xi of a cell");
  m.def(
      "evaluate_hcurl_curl",
      [](const ND& dofs, const Vector& e, const mesh::PointLocator<Dim>& locator,
         const Point<Dim>& x) { return assembly::evaluate_hcurl_curl<Dim>(dofs, e, locator, x); },
      py::arg("dofs"), py::arg("e"), py::arg("locator"), py::arg("x"));
  m.def(
      "sample_h1",
      [](const H1& dofs, const Vector& u, const std::vector<Index>& cells, const RealArray& xi) {
        const auto points = array_to_points<Dim>(xi, "xi");
        if (points.size() != cells.size()) {
          throw InvalidArgument("sample_h1: cells and xi must have the same length");
        }
        Vector out(static_cast<Index>(points.size()));
        {
          py::gil_scoped_release release;
          for (std::size_t i = 0; i < points.size(); ++i) {
            out(static_cast<Index>(i)) = assembly::evaluate_h1<Dim>(dofs, u, cells[i], points[i]);
          }
        }
        return out;
      },
      py::arg("dofs"), py::arg("u"), py::arg("cells"), py::arg("xi"),
      "Values at reference points xi (n, dim) of the given cells (n), e.g. the vertices of "
      "a subdivision");
  m.def(
      "sample_hcurl",
      [](const ND& dofs, const Vector& e, const std::vector<Index>& cells, const RealArray& xi,
         bool curl) {
        const auto points = array_to_points<Dim>(xi, "xi");
        if (points.size() != cells.size()) {
          throw InvalidArgument("sample_hcurl: cells and xi must have the same length");
        }
        constexpr int kCurl = Dim == 2 ? 1 : 3;
        Matrix out(static_cast<Index>(points.size()), curl ? kCurl : Dim);
        {
          py::gil_scoped_release release;
          for (std::size_t i = 0; i < points.size(); ++i) {
            if (curl) {
              out.row(static_cast<Index>(i)) =
                  assembly::evaluate_hcurl_curl<Dim>(dofs, e, cells[i], points[i]).transpose();
            } else {
              out.row(static_cast<Index>(i)) =
                  assembly::evaluate_hcurl<Dim>(dofs, e, cells[i], points[i]).transpose();
            }
          }
        }
        return out;
      },
      py::arg("dofs"), py::arg("e"), py::arg("cells"), py::arg("xi"), py::arg("curl") = false,
      "Field values (n, dim) — or curls (n, 1 / 3) with curl=True — at reference points xi "
      "(n, dim) of the given cells (n)");
  m.def(
      "h1_error",
      [](const H1& dofs, const Vector& u, const assembly::ScalarField<Dim>& exact,
         const assembly::VectorField<Dim>& grad,
         int extra_order) { return assembly::h1_error<Dim>(dofs, u, exact, grad, extra_order); },
      py::arg("dofs"), py::arg("u"), py::arg("exact"), py::arg("grad_exact"),
      py::arg("extra_order") = 2, Release(), "L2 and H1-semi errors against (u, grad u)");
  m.def(
      "hcurl_error",
      [](const ND& dofs, const Vector& e, const assembly::ComplexVectorField<Dim>& field,
         const assembly::ComplexCurlField<Dim>& curl,
         int extra_order) { return assembly::hcurl_error<Dim>(dofs, e, field, curl, extra_order); },
      py::arg("dofs"), py::arg("e"), py::arg("field"), py::arg("curl"), py::arg("extra_order") = 2,
      Release(), "L2 errors of the field and its curl against (E, curl E)");
  m.def(
      "hcurl_error",
      [](const ND& dofs, const Vector& e, const assembly::ComplexVectorField<Dim>& field,
         const assembly::ComplexCurlField<Dim>& curl, const std::vector<Index>& cells,
         int extra_order) {
        return assembly::hcurl_error<Dim>(dofs, e, field, curl, cells, extra_order);
      },
      py::arg("dofs"), py::arg("e"), py::arg("field"), py::arg("curl"), py::arg("cells"),
      py::arg("extra_order") = 2, Release(), "The same restricted to the listed cells");

  // --- prolongation ---------------------------------------------------------------------
  m.def(
      "prolongate",
      [](const H1& old_dofs, const Vector& old, const H1& new_dofs,
         const mesh::RefinementStep& step) {
        return assembly::prolongate(old_dofs, old, new_dofs, step);
      },
      py::arg("old_dofs"), py::arg("old_coefficients"), py::arg("new_dofs"), py::arg("step"),
      Release(), "Coefficients of the old function on the DoF map after a refinement step");
  m.def(
      "prolongate",
      [](const ND& old_dofs, const Vector& old, const ND& new_dofs,
         const mesh::RefinementStep& step) {
        return assembly::prolongate(old_dofs, old, new_dofs, step);
      },
      py::arg("old_dofs"), py::arg("old_coefficients"), py::arg("new_dofs"), py::arg("step"),
      Release());

  // --- functionals ----------------------------------------------------------------------
  m.def(
      "point_functional",
      [](const ND& dofs, const mesh::PointLocator<Dim>& locator,
         const std::vector<Point<Dim>>& points, const std::vector<ComplexVector<Dim>>& weights,
         const std::vector<ComplexCurl<Dim>>& curl_weights) {
        return assembly::point_functional<Dim>(dofs, locator, points, weights, curl_weights);
      },
      py::arg("dofs"), py::arg("locator"), py::arg("points"), py::arg("value_weights"),
      py::arg("curl_weights") = std::vector<ComplexCurl<Dim>>{}, Release(),
      "Vector q of Q(E) = sum_j E(x_j) . w_j + curl E(x_j) . v_j, so that Q(E_h) = q^T e_h");
}

}  // namespace

void bind_assembly(py::module_& m) {
  py::class_<assembly::AssembledSystem>(m, "AssembledSystem", "Sparse matrix and load vector")
      .def_readwrite("matrix", &assembly::AssembledSystem::matrix)
      .def_readwrite("rhs", &assembly::AssembledSystem::rhs);
  py::class_<assembly::MaxwellSystem>(m, "MaxwellSystem",
                                      "Stiffness S, mass M and load b: (S - omega^2 M) e = b")
      .def_readwrite("stiffness", &assembly::MaxwellSystem::stiffness)
      .def_readwrite("mass", &assembly::MaxwellSystem::mass)
      .def_readwrite("rhs", &assembly::MaxwellSystem::rhs);
  py::class_<assembly::ErrorNorms>(m, "ErrorNorms")
      .def_readonly("l2", &assembly::ErrorNorms::l2)
      .def_readonly("h1_semi", &assembly::ErrorNorms::h1_semi)
      .def_readonly("l2_norm", &assembly::ErrorNorms::l2_norm)
      .def_readonly("h1_norm", &assembly::ErrorNorms::h1_norm);
  py::class_<assembly::HcurlErrorNorms>(m, "HcurlErrorNorms")
      .def_readonly("l2", &assembly::HcurlErrorNorms::l2)
      .def_readonly("curl", &assembly::HcurlErrorNorms::curl)
      .def_readonly("l2_norm", &assembly::HcurlErrorNorms::l2_norm)
      .def_readonly("curl_norm", &assembly::HcurlErrorNorms::curl_norm);
  py::class_<assembly::DofValues>(m, "DofValues", "Sorted DoFs with prescribed values")
      .def(py::init([](std::vector<Index> dofs, Vector values) {
             assembly::DofValues d;
             d.dofs = std::move(dofs);
             d.values = std::move(values);
             return d;
           }),
           py::arg("dofs"), py::arg("values"))
      .def_property_readonly("dofs", [](const assembly::DofValues& d) { return to_array(d.dofs); })
      .def_readwrite("values", &assembly::DofValues::values)
      .def_property_readonly("size", &assembly::DofValues::size);
  m.def(
      "merge_dirichlet",
      [](const std::vector<assembly::DofValues>& parts) {
        return assembly::merge_dirichlet(parts);
      },
      py::arg("parts"), "Union of constraint sets; a DoF listed twice keeps its first value");
  m.def(
      "apply_dirichlet",
      [](SparseMatrix matrix, Vector rhs, const assembly::DofValues& data) {
        assembly::apply_dirichlet(matrix, rhs, data);
        return std::make_pair(std::move(matrix), std::move(rhs));
      },
      py::arg("matrix"), py::arg("rhs"), py::arg("data"), Release(),
      "Symmetric elimination keeping the system size: returns the modified (matrix, rhs)");
  m.def(
      "free_dofs",
      [](Index n, const std::vector<Index>& constrained) {
        return to_array(assembly::free_dofs(n, constrained));
      },
      py::arg("num_dofs"), py::arg("constrained"), "Complement of the sorted constrained DoFs");
  m.def("evaluate_functional", &assembly::evaluate_functional, py::arg("q"), py::arg("e"),
        "q^T e (no conjugation)");
  bind_assembly_dim<2>(m);
  bind_assembly_dim<3>(m);
}

}  // namespace hpfem::python
