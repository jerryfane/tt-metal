// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
namespace ckernel::sfpu {
struct SoftshrinkBf16Config {
    static constexpr uint32_t kKind = 0x00000002u;
    static constexpr uint32_t kThresholdBits = 0x3f000000u;
    static constexpr uint32_t kBodySlots = 0x00000000u;
    static constexpr uint32_t kRowsPerReplay = 0x00000000u;
};
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_softshrink_bf16_simple_forward.h"

namespace ckernel::sfpu {

// The kernel is fitted on BF16 data; the SFPU reads DEST in the math thread's SrcB format.
inline bool bf16_dest_softshrink() {
#if defined(TRISC_MATH) || defined(LLK_TRISC_MATH)
    return ckernel::math::src_zero_flag_srcb_fmt == static_cast<std::uint32_t>(DataFormat::Float16_b);
#else
    return false;
#endif
}
// The Blackhole whole-tile call as the fitter's scheduler issues it (ttpoly/sfpu_schedule.py): the
// compiled instructions of
// ckernel::sfpu::bf16::softshrink::calculate_simple_forward<ckernel::sfpu::SoftshrinkBf16Config, 32> in a stall-filling
// order, rows addressed by their immediates, constants held in LRegs. It stores the same word for every BF16 input.
inline void calculate_softshrink_bf16_scheduled() {
    TTI_SFPLOAD(0, 0, 7, 0);
    TTI_SFPLOADI(7, 0, 0x3f00);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 2);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 2);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 2);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 4);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 4);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 4);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 6);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 6);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 6);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 8);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 8);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 8);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0xa);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0xa);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0xa);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0xc);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0xc);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0xc);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0xe);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0xe);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0xe);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x10);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x10);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x10);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x12);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x12);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x12);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x14);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x14);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x14);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x16);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x16);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x16);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x18);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x18);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x18);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x1a);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x1a);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x1a);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x1c);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x1c);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x1c);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x1e);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x1e);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x1e);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x20);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x20);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x20);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x22);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x22);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x22);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x24);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x24);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x24);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x26);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x26);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x26);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x28);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x28);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x28);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x2a);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x2a);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x2a);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x2c);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x2c);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x2c);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x2e);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x2e);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x2e);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x30);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x30);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x30);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x32);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x32);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x32);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x34);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x34);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x34);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x36);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x36);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x36);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x38);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x38);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x38);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x3a);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x3a);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x3a);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x3c);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x3c);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x3c);
    TTI_SFPENCC(3, 0, 0, 0xa);
    TTI_SFPLOAD(0, 0, 7, 0x3e);
    TTI_SFPMOV(0, 0, 1, 2);
    TTI_SFPSETSGN(0, 7, 1, 0);
    TTI_SFPADD(10, 1, 0, 1, 1);
    TTI_SFPSETSGN(0, 0, 0, 1);
    TTI_SFPSTORE(9, 0, 7, 0x3e);
    TTI_SFPGT(0, 7, 0, 1);
    TTI_SFPSTORE(1, 0, 7, 0x3e);
    TTI_SFPENCC(3, 0, 0, 0xa);
}

template <int ITERATIONS = 32>
inline void calculate_softshrink_bf16() {
    if constexpr (ITERATIONS == 32) {
        calculate_softshrink_bf16_scheduled();
    } else {
        ckernel::sfpu::bf16::softshrink::calculate_simple_forward<ckernel::sfpu::SoftshrinkBf16Config, ITERATIONS>();
    }
}
inline void init_softshrink_bf16() {
    if (bf16_dest_softshrink()) {
        ckernel::sfpu::bf16::softshrink::init_simple_forward<ckernel::sfpu::SoftshrinkBf16Config>();
    }
}

}  // namespace ckernel::sfpu
