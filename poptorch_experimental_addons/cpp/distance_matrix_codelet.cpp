// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#ifdef __IPU__
#include <ipu_intrinsics>
#include <ipu_vector_math>
#endif

#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

static constexpr auto ONE_PTR = poplar::VectorLayout::ONE_PTR;
static constexpr auto SPAN = poplar::VectorLayout::SPAN;

#ifdef __IPU__

float l1dist(const float* a, const float* b, size_t size) {
    auto a2 = reinterpret_cast<const float2*>(a);
    auto b2 = reinterpret_cast<const float2*>(b);

    float2 sum = {0.0, 0.0};
    for (size_t i = 0; i < size / 2; ++i) {
        sum += ipu::fabs(a2[i] - b2[i]);
    }
    float res = sum[0] + sum[1];
    if (size % 2) res += ipu::fabs(a[size - 1] - b[size - 1]);
    return res;
}

float l1dist(const half* a, const half* b, size_t size) {
    auto a4 = reinterpret_cast<const half4*>(a);
    auto b4 = reinterpret_cast<const half4*>(b);

    half4 sum = {0.0, 0.0, 0.0, 0.0};
    for (size_t i = 0; i < size / 4; ++i) {
        sum += ipu::fabs(a4[i] - b4[i]);
    }
    float res = float(sum[0]) + float(sum[1]) + float(sum[2]) + float(sum[3]);
    size_t rem = size % 4;
    if (rem) {
        for (size_t i = size - rem; i < size; ++i) {
            res += ipu::fabs(float(a[i] - b[i]));
        }
    }
    return res;
}

#else  // !__IPU__

template <typename T>
float l1dist(const T* a, const T* b, size_t size) {
    float sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += std::fabs(float(a[i] - b[i]));
    }
    return sum;
}

#endif  // __IPU__

template <typename T>
class L1DistanceSingleVertex : public poplar::Vertex {
   public:
    poplar::Input<poplar::Vector<T, SPAN, 8>> a;
    poplar::Input<poplar::Vector<T, ONE_PTR, 8>> b;
    // Note: Output<half> generates a slower 16-bit write, so we always output float
    poplar::Output<float> out;

    bool compute() {
        *out = l1dist(&a[0], &b[0], a.size());
        return true;
    }
};
template class L1DistanceSingleVertex<float>;
template class L1DistanceSingleVertex<half>;

template <typename T>
static inline T signum(T x) {
    return T((T(0.0) < x) - (x < T(0.0)));
}

#ifdef __IPU__

static inline half4 signum(half4 x) {
    constexpr half4 one = {1.0f, 1.0f, 1.0f, 1.0f};
    const auto pOne = reinterpret_cast<const uint32_t*>(&one);
    const auto xNonzero = x != half4{0.0f, 0.0f, 0.0f, 0.0f};
    const auto pXNonzero = reinterpret_cast<const uint32_t*>(&xNonzero);
    const uint32_t pZeroOrOne[2] = {pXNonzero[0] & pOne[0], pXNonzero[1] & pOne[1]};
    return ipu::copysign(*reinterpret_cast<const half4*>(pZeroOrOne), x);
}

static inline float2 signum(float2 x) {
    constexpr float2 one = {1.0f, 1.0f};
    const auto pOne = reinterpret_cast<const uint32_t*>(&one);
    const auto xNonzero = x != float2{0.0f, 0.0f};
    const auto pXNonzero = reinterpret_cast<const uint32_t*>(&xNonzero);
    const uint32_t pZeroOrOne[2] = {pXNonzero[0] & pOne[0], pXNonzero[1] & pOne[1]};
    return ipu::copysign(*reinterpret_cast<const float2*>(pZeroOrOne), x);
}

float l1distgrad(const float& a, const float* b, const float* grad, size_t size) {
    float2 sum = {0.0, 0.0};
    auto grad2 = reinterpret_cast<const float2*>(grad);
    auto b2 = reinterpret_cast<const float2*>(b);
    float2 a2 = {a, a};

    for (size_t i = 0; i < size / 2; ++i) {
        sum += grad2[i] * signum(a2 - b2[i]);
    }
    float res = sum[0] + sum[1];
    if (size % 2) {
        res += grad[size - 1] * signum(a - b[size - 1]);
    }
    return res;
}

float l1distgrad(const half& a, const half* b, const half* grad, size_t size) {
    half4 sum = {0.0, 0.0};
    auto grad4 = reinterpret_cast<const half4*>(grad);
    auto b4 = reinterpret_cast<const half4*>(b);
    half4 a4 = {a, a, a, a};

    for (size_t i = 0; i < size / 4; ++i) {
        sum += grad4[i] * signum(a4 - b4[i]);
    }
    float res = float(sum[0]) + float(sum[1]) + float(sum[2]) + float(sum[3]);
    size_t rem = size % 4;
    if (rem) {
        for (size_t i = size - rem; i < size; ++i) {
            res += float(grad[i] * signum(a - b[i]));
        }
    }
    return res;
}

#else  // !__IPU__

template <typename T>
float l1distgrad(const T& a, const T* b, const T* grad, size_t size) {
    float sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += float(grad[i]) * float(signum(a - b[i]));
    }
    return sum;
}

#endif  // __IPU__

template <typename T>
class L1DistanceGradSingleVertex : public poplar::Vertex {
   public:
    poplar::Input<T> a;
    poplar::Input<poplar::Vector<T, SPAN, 8>> b;
    poplar::Input<poplar::Vector<T, ONE_PTR, 8>> gradOutput;
    // Note: Output<half> generates a slower 16-bit write, so we always output float
    poplar::Output<float> grad;

    bool compute() {
        *grad = l1distgrad(*a, &b[0], &gradOutput[0], b.size());
        return true;
    }
};

template class L1DistanceGradSingleVertex<float>;
template class L1DistanceGradSingleVertex<half>;

#ifdef __IPU__

float l2dist(const float* a, const float* b, size_t size) {
    auto a2 = reinterpret_cast<const float2*>(a);
    auto b2 = reinterpret_cast<const float2*>(b);

    float2 sum = {0.0, 0.0};
    for (size_t i = 0; i < size / 2; ++i) {
        float2 diff = a2[i] - b2[i];
        sum += diff * diff;
    }
    float res = sum[0] + sum[1];
    if (size % 2) {
        float diff = a[size - 1] - b[size - 1];
        res += diff * diff;
    }
    return ipu::sqrt(res);
}

float l2dist(const half* a, const half* b, size_t size) {
    auto a2 = reinterpret_cast<const half2*>(a);
    auto b2 = reinterpret_cast<const half2*>(b);

    float2 sum = {0.0, 0.0};
    for (size_t i = 0; i < size / 2; ++i) {
        auto diff = a2[i] - b2[i];
        float2 diff32 = {float(diff[0]), float(diff[1])};
        sum += diff32 * diff32;
    }
    auto res = sum[0] + sum[1];
    if (size % 2) {
        auto diff = float(a[size - 1] - b[size - 1]);
        res += diff * diff;
    }
    return ipu::sqrt(res);
}

#else  // !__IPU__

template <typename T>
float l2dist(const T* a, const T* b, size_t size) {
    float sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        float diff = float(a[i] - b[i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

#endif  // __IPU__

template <typename T>
class L2DistanceSingleVertex : public poplar::Vertex {
   public:
    poplar::Input<poplar::Vector<T, SPAN, 8>> a;
    poplar::Input<poplar::Vector<T, ONE_PTR, 8>> b;
    // Note: Output<half> generates a slower 16-bit write, so we always output float
    poplar::Output<float> out;

    bool compute() {
        *out = l2dist(&a[0], &b[0], a.size());
        return true;
    }
};
template class L2DistanceSingleVertex<float>;
template class L2DistanceSingleVertex<half>;

#ifdef __IPU__

float l2distgrad(const float& a,
                 const float* b,
                 const float* dist,
                 const float* grad,
                 size_t size) {
    float2 a2 = {a, a};
    auto b2 = reinterpret_cast<const float2*>(b);
    auto dist2 = reinterpret_cast<const float2*>(dist);
    auto grad2 = reinterpret_cast<const float2*>(grad);

    float2 sum = {0.0, 0.0};
    for (size_t i = 0; i < size / 2; ++i) {
        const float2 diff = a2 - b2[i];
        const float2 dist2_i = dist2[i];
        const auto comp = dist2_i == float2{0.0f, 0.0f};
        const uint32_t dist_mask[2] = {0x3f800000, 0x3f800000};  // 1.0f
        const uint32_t dist_delta[2] = {comp[0] & dist_mask[0], comp[1] & dist_mask[1]};
        // if dist == 0 -> safe_dist=1.0.
        const float2 safe_dist = dist2_i + *reinterpret_cast<const float2*>(dist_delta);
        const float2 masked_diff = ipu::andc(diff, *reinterpret_cast<const float2*>(&comp));
        sum += grad2[i] * masked_diff / safe_dist;
    }
    float res = sum[0] + sum[1];
    if (size % 2) {
        float diff = a - b[size - 1];
        float dist_i = float(dist[size - 1]);
        res += dist_i == 0.0f ? 0.0f : float(grad[size - 1]) * diff / dist_i;
    }
    return res;
}

float l2distgrad(const half& a, const half* b, const half* dist, const half* grad, size_t size) {
    half4 a4 = {a, a, a, a};
    auto b4 = reinterpret_cast<const half4*>(b);
    auto dist4 = reinterpret_cast<const half4*>(dist);
    auto grad4 = reinterpret_cast<const half4*>(grad);

    half4 sum = {0.0, 0.0, 0.0, 0.0};
    for (size_t i = 0; i < size / 4; ++i) {
        const half4 diff = a4 - b4[i];
        const half4 dist4_i = dist4[i];
        const auto comp = dist4_i == half4{0.0f, 0.0f, 0.0f, 0.0f};
        const uint16_t dist_mask[4] = {0x3f80, 0x3f80, 0x3f80, 0x3f80};  // 1.0f
        const auto p_comp = reinterpret_cast<const uint32_t*>(&comp);
        const auto p_mask = reinterpret_cast<const uint32_t*>(dist_mask);
        const uint32_t dist_delta[2] = {p_comp[0] & p_mask[0], p_comp[1] & p_mask[1]};
        // if dist == 0 -> safe_dist=1.0.
        const half4 safe_dist = dist4_i + *reinterpret_cast<const half4*>(dist_delta);
        const float2 masked_diff = ipu::andc(*reinterpret_cast<const float2*>(&diff),
                                             *reinterpret_cast<const float2*>(&comp));
        sum += grad4[i] * *reinterpret_cast<const half4*>(&masked_diff) / safe_dist;
    }
    float2 res2 = ipu::sum(sum);
    float res = res2[0] + res2[1];
    size_t rem = size % 4;
    for (size_t i = size - rem; i < size; ++i) {
        float diff = float(a) - float(b[i]);
        float dist_i = float(dist[i]);
        res += dist_i == 0.0f ? 0.0f : float(grad[i]) * diff / dist_i;
    }
    return res;
}

#else  // !__IPU__

template <typename T>
float l2distgrad(const T& a, const T* b, const T* dist, const T* grad, size_t size) {
    float sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        float diff = float(a - b[i]);
        float dist_i = float(dist[i]);
        sum += dist_i == 0.0f ? 0.0f : float(grad[i]) * diff / dist_i;
    }
    return sum;
}

#endif

template <typename T>
class L2DistanceGradSingleVertex : public poplar::Vertex {
   public:
    poplar::Input<T> a;
    poplar::Input<poplar::Vector<T, SPAN, 8>> b;
    poplar::Input<poplar::Vector<T, ONE_PTR, 8>> dist;
    poplar::Input<poplar::Vector<T, ONE_PTR, 8>> gradOutput;
    // Note: Output<half> generates a slower 16-bit write, so we always output float
    poplar::Output<float> grad;

    bool compute() {
        *grad = l2distgrad(*a, &b[0], &dist[0], &gradOutput[0], b.size());
        return true;
    }
};

template class L2DistanceGradSingleVertex<float>;
template class L2DistanceGradSingleVertex<half>;
