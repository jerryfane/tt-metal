// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_tanhshrink_bf16_signed_abs_affine.h"
#include "sfpi.h"

namespace ckernel::sfpu::bf16::tanhshrink
{
template <typename Config, int Iterations = 8>
inline void calculate_signed_abs_affine()
{
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    ttpoly::tanhshrink::signed_abs_affine_pin_coefficients<Config>();
#if defined(DISABLE_SFPLOADMACRO)
    // The same rows without the load macro or the replay buffer: identical results.
#pragma GCC unroll 1
    for (int row = 0; row < Iterations; ++row)
    {
        ttpoly::tanhshrink::signed_abs_affine_body<Config, ADDR_MOD_7, ADDR_MOD_6, false>();
    }
#else
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    ttpoly::tanhshrink::signed_abs_affine_body<Config, ADDR_MOD_7, ADDR_MOD_6>();
#pragma GCC unroll 8
    for (int row = 1; row < Iterations; ++row)
    {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
    ttpoly::tanhshrink::signed_abs_affine_drain();
#endif
    // No deployment SETRWC: the upstream LLK owns face traversal/counters.
}
} // namespace ckernel::sfpu::bf16::tanhshrink
