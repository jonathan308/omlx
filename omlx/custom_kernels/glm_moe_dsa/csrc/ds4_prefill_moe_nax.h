#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::glm_kernels {

// Isolated M5/NAX DS4-Flash M=1024 routed-MoE projection. The primitive
// consumes the existing expert-local BM32 block plan and preserves stock
// gather_qmm's BF16 output boundary. No model/runtime path calls this symbol.
mx::array deepseek_mxfp4_gather_qmm_blocks_nax(
    const mx::array &x, const mx::array &weight, const mx::array &scales,
    const mx::array &block_meta, const mx::array &block_count,
    mx::StreamOrDevice s = {});

} // namespace omlx::glm_kernels
