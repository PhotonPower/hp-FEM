// Periodic face pairs of the adaptive mesh (M15 F16, stage 1): refining a cell at one face
// of a Bloch pair, or the closure refining one next to it, is mirrored onto the partner face,
// so the leaf facets of both faces stay identical up to the shift and the Bloch constraints
// can be built after every step. Without the declaration the same marking leaves the faces
// different.
#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/adaptivity/refinement.hpp"
#include "hpfem/assembly/periodic.hpp"
#include "hpfem/core/error.hpp"
#include "hpfem/fespace/dof_map.hpp"
#include "hpfem/mesh/adaptive_mesh.hpp"
#include "hpfem/mesh/generators.hpp"
#include "hpfem/mesh/geometry.hpp"

using hpfem::as_size;
using hpfem::Index;
using hpfem::Point;
using hpfem::Real;
using hpfem::mesh::AdaptiveMesh;
using hpfem::mesh::Mesh;
using hpfem::mesh::PeriodicFace;
namespace box_tag = hpfem::mesh::box_tag;

namespace {

/// Sorted (centroid - shift, diameter) of the facets with a tag, projected onto the face.
template <int Dim>
std::vector<std::pair<Point<Dim>, Real>> face_facets(const Mesh<Dim>& mesh, hpfem::mesh::Tag tag,
                                                     const Point<Dim>& shift) {
  std::vector<std::pair<Point<Dim>, Real>> out;
  for (const Index f : mesh.facets_with_tag(tag)) {
    const auto& fv = mesh.facet_vertices(f);
    Point<Dim> c = Point<Dim>::Zero();
    Real d = 0;
    for (const Index v : fv) c += mesh.vertex(v);
    c = c / static_cast<Real>(fv.size()) + shift;
    for (std::size_t a = 0; a < fv.size(); ++a) {
      for (std::size_t b = a + 1; b < fv.size(); ++b) {
        d = std::max(d, (mesh.vertex(fv[a]) - mesh.vertex(fv[b])).norm());
      }
    }
    out.emplace_back(c, d);
  }
  std::sort(out.begin(), out.end(), [](const auto& x, const auto& y) {
    for (int k = 0; k < Dim; ++k) {
      if (std::abs(x.first(k) - y.first(k)) > 1e-12) return x.first(k) < y.first(k);
    }
    return x.second < y.second;
  });
  return out;
}

template <int Dim>
bool faces_match(const Mesh<Dim>& mesh, const PeriodicFace<Dim>& face) {
  const auto master = face_facets(mesh, face.master, face.shift);
  const Point<Dim> zero = Point<Dim>::Zero();
  const auto slave = face_facets(mesh, face.slave, zero);
  if (master.size() != slave.size()) return false;
  for (std::size_t i = 0; i < master.size(); ++i) {
    if ((master[i].first - slave[i].first).norm() > 1e-12) return false;
    if (std::abs(master[i].second - slave[i].second) > 1e-12) return false;
  }
  return true;
}

/// A leaf cell with a facet on the face carrying `tag`.
template <int Dim>
Index cell_at_face(const Mesh<Dim>& mesh, hpfem::mesh::Tag tag) {
  return mesh.facet_cells(mesh.facets_with_tag(tag).front())[0];
}

}  // namespace

TEST_CASE("periodic faces: refinement at one face is mirrored onto the partner",
          "[mesh][adaptive][periodic]") {
  const PeriodicFace<2> pair{box_tag::kXMin, box_tag::kXMax, Point<2>(1.0, 0.0)};
  // without the declaration the faces drift apart
  {
    AdaptiveMesh<2> plain(hpfem::mesh::rectangle(4, 4));
    const Index c = cell_at_face(plain.mesh(), box_tag::kXMin);
    plain.refine(std::vector<Index>{c});
    REQUIRE_FALSE(faces_match(plain.mesh(), pair));
  }
  AdaptiveMesh<2> adaptive(hpfem::mesh::rectangle(4, 4));
  adaptive.set_periodic({pair});
  REQUIRE(adaptive.periodic().size() == 1);
  // refine a cell at the left face, then one at the right face, then one next to a face
  // (the closure refines the face cell), then everything twice
  const Index left = cell_at_face(adaptive.mesh(), box_tag::kXMin);
  adaptive.refine(std::vector<Index>{left});
  REQUIRE(faces_match(adaptive.mesh(), pair));
  REQUIRE(adaptive.mesh().facets_with_tag(box_tag::kXMin).size() > 4);
  const Index right = cell_at_face(adaptive.mesh(), box_tag::kXMax);
  adaptive.refine(std::vector<Index>{right});
  REQUIRE(faces_match(adaptive.mesh(), pair));
  Index inner = hpfem::kInvalidIndex;
  for (Index c = 0; c < adaptive.mesh().num_cells(); ++c) {
    const Point<2> x = hpfem::mesh::affine_map(adaptive.mesh(), c).centroid();
    if (x(0) > 0.25 && x(0) < 0.5 && x(1) < 0.25) inner = c;
  }
  REQUIRE(inner != hpfem::kInvalidIndex);
  adaptive.refine(std::vector<Index>{inner});
  REQUIRE(faces_match(adaptive.mesh(), pair));
  // the Bloch constraints of both spaces can be built on the result
  const hpfem::fespace::NedelecDofMap<2> nd(adaptive.mesh(), 2);
  const hpfem::fespace::DofMap<2> h1(adaptive.mesh(), 2);
  const std::vector<hpfem::assembly::PeriodicPair<2>> pairs = {
      {box_tag::kXMin, box_tag::kXMax, Point<2>(1.0, 0.0), hpfem::Complex{1.0, 0.0}}};
  REQUIRE_NOTHROW(hpfem::assembly::bloch_constraints<2>(nd, pairs));
  REQUIRE_NOTHROW(hpfem::assembly::bloch_constraints<2>(h1, pairs));
  // hp_refine goes through the same path
  std::vector<int> orders(as_size(adaptive.mesh().num_cells()), 2);
  const Index again = cell_at_face(adaptive.mesh(), box_tag::kXMin);
  const auto hp = hpfem::adaptivity::hp_refine<2>(adaptive, orders, std::vector<Index>{again},
                                                  std::vector<Index>{});
  REQUIRE(hp.orders.size() == as_size(adaptive.mesh().num_cells()));
  REQUIRE(faces_match(adaptive.mesh(), pair));
  REQUIRE_THROWS_AS(
      adaptive.set_periodic({PeriodicFace<2>{99, box_tag::kXMax, Point<2>(1.0, 0.0)}}),
      hpfem::InvalidArgument);
}

TEST_CASE("periodic faces in 3D: mirrored refinement across a box", "[mesh][adaptive][periodic]") {
  const PeriodicFace<3> pair{box_tag::kXMin, box_tag::kXMax, Point<3>(1.0, 0.0, 0.0)};
  AdaptiveMesh<3> adaptive(hpfem::mesh::box(2, 2, 2));
  adaptive.set_periodic({pair});
  const Index left = cell_at_face(adaptive.mesh(), box_tag::kXMin);
  adaptive.refine(std::vector<Index>{left});
  REQUIRE(faces_match(adaptive.mesh(), pair));
  const Index right = cell_at_face(adaptive.mesh(), box_tag::kXMax);
  adaptive.refine(std::vector<Index>{right});
  REQUIRE(faces_match(adaptive.mesh(), pair));
  const hpfem::fespace::NedelecDofMap<3> nd(adaptive.mesh(), 1);
  const std::vector<hpfem::assembly::PeriodicPair<3>> pairs = {
      {box_tag::kXMin, box_tag::kXMax, Point<3>(1.0, 0.0, 0.0), hpfem::Complex{1.0, 0.0}}};
  REQUIRE_NOTHROW(hpfem::assembly::bloch_constraints<3>(nd, pairs));
}
