// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <cmath>
#include <ipu_intrinsics>

#include <poplar/Vertex.hpp>

enum class RoundingMode { Nearest, Stochastic };

namespace {

// Helpers
template <class R, class T>
R as(T x) {
    return *reinterpret_cast<R*>(&x);
}
half4 andc(half4 a, half4 b) {
    return as<half4>(ipu::andc(as<float2>(a), as<float2>(b)));
}
half4 toHalf4(half a) {
    return half4{a, a, a, a};
}
typedef uint32_t uintv2 __attribute__((vector_size(sizeof(uint32_t) * 2)));
uintv2 repeat4(uint16_t x) {
    auto x32 = static_cast<uint32_t>(x);
    return uintv2{(x32 << 16) | x32, (x32 << 16) | x32};
}

// Core quantisation functions
template <RoundingMode Rounding>
half4 quantise4(half4 x, half absMax, half downscale, uint16_t mask) {
    x = ipu::clamp(x, half2{-absMax, absMax});
    x /= downscale;
    uintv2 offset;
    switch (Rounding) {
        case RoundingMode::Nearest:
            offset = repeat4(mask / 2);
            break;
        case RoundingMode::Stochastic:
            offset = {__builtin_ipu_urand32(), __builtin_ipu_urand32()};
            offset &= repeat4(mask);
            break;
    }
    // If a 16-bit integer add overflows, we have bigger problems than the int32 packing
    x = as<half4>(as<uintv2>(x) + offset);
    x = andc(x, toHalf4(as<half>(mask)));
    x *= downscale;
    return x;
}
// Note that we can't reuse `quantise4` here, otherwise we get slower code using function
// calls (non-inlined)
template <RoundingMode Rounding>
half quantise1(half x, half absMax, half downscale, uint16_t mask) {
    x = ipu::clamp(half2{x, 0}, half2{-absMax, absMax})[0];
    x /= downscale;
    auto offset = (Rounding == RoundingMode::Stochastic)
                      ? static_cast<uint16_t>(__builtin_ipu_urand32()) & mask
                      : (mask / 2);
    x = as<half>((as<uint16_t>(x) + offset) & ~mask);
    x *= downscale;
    return x;
}

// Loop
template <RoundingMode Rounding>
void quantise_loop(const half* __restrict__ input,
                   unsigned n,
                   half absMax,
                   half downscale,
                   uint16_t mask,
                   half* __restrict__ output) {
    for (auto i = 0u; i < n / 4; ++i) {
        reinterpret_cast<half4*>(output)[i] =
            quantise4<Rounding>(reinterpret_cast<const half4*>(input)[i], absMax, downscale, mask);
    }
    for (auto i = 4 * (n / 4); i < n; ++i) {
        output[i] = quantise1<Rounding>(input[i], absMax, downscale, mask);
    }
}
}  // namespace

template <RoundingMode Rounding>
struct CustomQuant : poplar::MultiVertex {
    poplar::Input<poplar::Vector<half, poplar::VectorLayout::SPAN, 8>> input;
    poplar::Output<poplar::Vector<half, poplar::VectorLayout::COMPACT_PTR, 8>> output;
    poplar::Input<half> absMax;
    poplar::Input<half> downscale;
    poplar::Input<uint16_t> mask;

    bool compute(unsigned id) {
        auto grainSize = 4 * numWorkers();
        auto blockSize = 4 * ((input.size() + grainSize - 1) / grainSize);
        auto start = std::min(blockSize * id, input.size());
        auto end = std::min(blockSize * (id + 1), input.size());
        quantise_loop<Rounding>(input.data() + start, end - start, *absMax, *downscale, *mask,
                                output.data() + start);
        return true;
    }
};

template struct CustomQuant<RoundingMode::Nearest>;
template struct CustomQuant<RoundingMode::Stochastic>;
