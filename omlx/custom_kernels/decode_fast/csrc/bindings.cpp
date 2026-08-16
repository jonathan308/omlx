#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "decode_fast.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Native decode fast-path kernels for oMLX";

  // ABI canary: see omlx.custom_kernels.qwen35_prefill (issue #2139).
  m.def(
      "abi_probe",
      [](const mlx::core::array& a) {
        return static_cast<int64_t>(a.size());
      },
      "a"_a);

  m.def(
      "rms_norm_residual_supported",
      &omlx::decode_fast_kernels::rms_norm_residual_supported,
      "x"_a,
      "weight"_a,
      "residual"_a,
      "stream"_a = nb::none());

  m.def(
      "rms_norm_residual",
      &omlx::decode_fast_kernels::rms_norm_residual,
      "x"_a,
      "weight"_a,
      "residual"_a,
      "eps"_a,
      "stream"_a = nb::none());
}
