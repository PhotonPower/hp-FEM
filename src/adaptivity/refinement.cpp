#include "hpfem/adaptivity/refinement.hpp"

#include <algorithm>

#include <fmt/format.h>

#include "hpfem/core/error.hpp"
#include "hpfem/core/log.hpp"
#include "hpfem/mesh/geometry.hpp"
#include "hpfem/mesh/point_location.hpp"

namespace hpfem::adaptivity {

namespace {

void check_marked(std::span<const Index> marked, Index num_cells, const char* what) {
  for (const Index c : marked) {
    if (c < 0 || c >= num_cells) {
      throw InvalidArgument(fmt::format("{}: cell {} outside 0..{}", what, c, num_cells - 1));
    }
  }
}

}  // namespace

std::vector<int> p_refine(std::span<const int> orders, std::span<const Index> marked, int increment,
                          int max_order) {
  if (increment < 1) {
    throw InvalidArgument(fmt::format("p_refine: increment {} must be positive", increment));
  }
  check_marked(marked, static_cast<Index>(orders.size()), "p_refine");
  std::vector<int> out(orders.begin(), orders.end());
  for (const Index c : marked) {
    int& p = out[as_size(c)];
    p += increment;
    if (max_order > 0) p = std::min(p, max_order);
  }
  return out;
}

mesh::RefinementStep identity_step(Index num_cells) {
  mesh::RefinementStep step;
  step.num_old_cells = num_cells;
  step.parent.resize(as_size(num_cells));
  step.path.assign(as_size(num_cells), {});
  for (Index c = 0; c < num_cells; ++c) step.parent[as_size(c)] = c;
  return step;
}

template <int Dim>
HpStep hp_refine(mesh::AdaptiveMesh<Dim>& mesh, std::span<const int> orders,
                 std::span<const Index> h_marked, std::span<const Index> p_marked, int increment,
                 int max_order, bool spread_p) {
  const mesh::Mesh<Dim>& leaf = mesh.mesh();
  const Index num_cells = leaf.num_cells();
  if (static_cast<Index>(orders.size()) != num_cells) {
    throw InvalidArgument(
        fmt::format("hp_refine: {} orders given for {} cells", orders.size(), num_cells));
  }
  check_marked(h_marked, num_cells, "hp_refine");
  // raise first on the old cells, then let the children inherit
  std::vector<int> raised = p_refine(orders, p_marked, increment, max_order);
  if (spread_p) {
    std::vector<char> split(as_size(num_cells), 0);
    for (const Index c : h_marked) split[as_size(c)] = 1;
    std::vector<char> marked(as_size(num_cells), 0);
    for (const Index c : p_marked) marked[as_size(c)] = 1;
    for (const Index c : p_marked) {
      const int target = raised[as_size(c)];
      std::vector<Index> neighbours;
      for (const Index f : leaf.cell_facets(c)) {
        const auto& fc = leaf.facet_cells(f);
        if (fc[1] != kInvalidIndex) neighbours.push_back(fc[0] == c ? fc[1] : fc[0]);
        const Index parent = leaf.hanging_parent_facet(f);
        if (parent != kInvalidIndex) neighbours.push_back(leaf.facet_cells(parent)[0]);
        for (const Index child : leaf.hanging_child_facets(f)) {
          neighbours.push_back(leaf.facet_cells(child)[0]);
        }
      }
      for (const Index n : neighbours) {
        if (split[as_size(n)] != 0 || marked[as_size(n)] != 0) continue;
        int& q = raised[as_size(n)];
        if (q < target) {
          q = std::min(q + increment, target);
          if (max_order > 0) q = std::min(q, max_order);
        }
      }
    }
  }
  HpStep out;
  out.step = mesh.refine(h_marked);
  out.orders.resize(as_size(out.step.num_cells()));
  for (Index c = 0; c < out.step.num_cells(); ++c) {
    out.orders[as_size(c)] = raised[as_size(out.step.parent[as_size(c)])];
  }
  log().debug("hp_refine<{}>: {} h-marked, {} p-marked, {} -> {} cells, orders {}..{}", Dim,
              h_marked.size(), p_marked.size(), num_cells, out.step.num_cells(),
              *std::min_element(out.orders.begin(), out.orders.end()),
              *std::max_element(out.orders.begin(), out.orders.end()));
  return out;
}

template <int Dim>
std::vector<mesh::RefinementStep> refine_at_points(mesh::AdaptiveMesh<Dim>& mesh,
                                                   std::span<const Point<Dim>> points, int levels,
                                                   Real tolerance) {
  if (levels < 0) throw InvalidArgument("refine_at_points: levels must be non-negative");
  std::vector<mesh::RefinementStep> steps;
  for (int level = 0; level < levels; ++level) {
    const auto& leaves = mesh.mesh();
    const mesh::PointLocator<Dim> locator(leaves);
    std::vector<Index> marked;
    for (Index c = 0; c < leaves.num_cells(); ++c) {
      const Real tol = tolerance * mesh::affine_map(leaves, c).h;
      bool touches = false;
      for (const Point<Dim>& x : points) {
        for (const Index v : leaves.cell_vertices(c)) {
          if ((leaves.vertex(v) - x).norm() <= tol) touches = true;
        }
      }
      if (touches) marked.push_back(c);
    }
    for (const Point<Dim>& x : points) {
      if (const auto located = locator.locate(x)) marked.push_back(located->cell);
    }
    std::sort(marked.begin(), marked.end());
    marked.erase(std::unique(marked.begin(), marked.end()), marked.end());
    if (marked.empty()) {
      throw InvalidArgument("refine_at_points: no cell touches any of the points");
    }
    steps.push_back(mesh.refine(marked));
  }
  log().info("refine_at_points: {} points, {} levels, {} cells now", points.size(), levels,
             mesh.mesh().num_cells());
  return steps;
}

template std::vector<mesh::RefinementStep> refine_at_points<2>(mesh::AdaptiveMesh<2>&,
                                                               std::span<const Point<2>>, int,
                                                               Real);
template std::vector<mesh::RefinementStep> refine_at_points<3>(mesh::AdaptiveMesh<3>&,
                                                               std::span<const Point<3>>, int,
                                                               Real);
template HpStep hp_refine<2>(mesh::AdaptiveMesh<2>&, std::span<const int>, std::span<const Index>,
                             std::span<const Index>, int, int, bool);
template HpStep hp_refine<3>(mesh::AdaptiveMesh<3>&, std::span<const int>, std::span<const Index>,
                             std::span<const Index>, int, int, bool);

}  // namespace hpfem::adaptivity
