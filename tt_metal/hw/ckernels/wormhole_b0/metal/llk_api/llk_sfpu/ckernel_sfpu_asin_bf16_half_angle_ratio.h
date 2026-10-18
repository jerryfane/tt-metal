// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_asin_bf16_half_angle_ratio_core.h"
namespace ckernel::sfpu::bf16::asin {
template <typename Config>
inline void init_half_angle_ratio() {
    sfpi::vConstIntPrgm0 = Config::kMagic;
    sfpi::vConstFloatPrgm1 = Config::kCoefficients[3];
    sfpi::vConstFloatPrgm2 = 1.5f;
}

template <typename Config, int Iterations = 32>
inline void calculate_half_angle_ratio() {
    static_assert(Iterations == 32);
    static_assert(Config::kSharedPool);
#pragma GCC unroll 2
    for (int row = 0; row < Iterations; ++row) {
        sfpi::vUInt raw_u16;
        sfpi::vFloat raw = sfpi::dst_reg[row];
        sfpi::vFloat result = ttpoly::asin::half_angle_evaluate<Config>(
            raw,
            [](sfpi::vFloat x) {
                return ttpoly::asin::half_angle_root<Config::kSharedPool, Config::kRootIterations>(x);
            },
            [](sfpi::vFloat x) {
                return ttpoly::asin::half_angle_ratio<3, Config::kSharedPool>(
                    x, [](unsigned k) { return Config::kCoefficients[k]; });
            });
        raw_u16 = sfpi::dst_reg[row].template mode<sfpi::DataLayout::U16>();
        ttpoly::asin::
            half_angle_negative_nan<Config::kRawMask, Config::kRawValue, Config::kRawExcluded, Config::kRawClass>(
                raw_u16, result);
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[row] = result;
    }
}
}  // namespace ckernel::sfpu::bf16::asin
