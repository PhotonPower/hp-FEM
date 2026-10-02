#include <array>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "hpfem/assembly/sparse_assembler.hpp"
#include "hpfem/core/error.hpp"

using hpfem::Complex;
using hpfem::Index;
using hpfem::Matrix;
using hpfem::SparseMatrix;
using hpfem::Vector;
using hpfem::assembly::gather;
using hpfem::assembly::scatter;
using hpfem::assembly::SparseAssembler;

TEST_CASE("SparseAssembler sums overlapping element matrices into CSR", "[assembly]") {
  SparseAssembler assembler(4, 4);
  Matrix local = Matrix::Zero(2, 2);
  local << Complex{1.0, 0.0}, Complex{2.0, 1.0}, Complex{3.0, 0.0}, Complex{4.0, 0.0};
  const std::array<Index, 2> first{0, 1};
  const std::array<Index, 2> second{1, 3};
  assembler.add(first, first, local);
  assembler.add(second, second, local);
  assembler.add(2, 2, Complex{10.0, 0.0});
  REQUIRE(assembler.num_triplets() == 9);

  const SparseMatrix a = assembler.finalize();
  REQUIRE(a.rows() == 4);
  REQUIRE(a.cols() == 4);
  REQUIRE(a.IsRowMajor);
  REQUIRE(a.isCompressed());
  REQUIRE(a.nonZeros() == 8);  // (1,1) merged
  REQUIRE(a.coeff(0, 0) == Complex{1.0, 0.0});
  REQUIRE(a.coeff(0, 1) == Complex{2.0, 1.0});
  REQUIRE(a.coeff(1, 1) == Complex{5.0, 0.0});  // 4 + 1
  REQUIRE(a.coeff(1, 3) == Complex{2.0, 1.0});
  REQUIRE(a.coeff(3, 1) == Complex{3.0, 0.0});
  REQUIRE(a.coeff(3, 3) == Complex{4.0, 0.0});
  REQUIRE(a.coeff(2, 2) == Complex{10.0, 0.0});
  REQUIRE(a.coeff(0, 3) == Complex{0.0, 0.0});
  REQUIRE_THROWS_AS(SparseAssembler(-1, 2), hpfem::InvalidArgument);
}

TEST_CASE("scatter and gather", "[assembly]") {
  Vector global = Vector::Zero(5);
  Vector local(2);
  local << Complex{1.0, 2.0}, Complex{3.0, 0.0};
  const std::array<Index, 2> ids{4, 1};
  scatter(global, ids, local);
  scatter(global, ids, local);
  REQUIRE(global(4) == Complex{2.0, 4.0});
  REQUIRE(global(1) == Complex{6.0, 0.0});
  REQUIRE(global(0) == Complex{0.0, 0.0});
  const Vector back = gather(global, ids);
  REQUIRE(back.size() == 2);
  REQUIRE(back(0) == Complex{2.0, 4.0});
  REQUIRE(back(1) == Complex{6.0, 0.0});
}
