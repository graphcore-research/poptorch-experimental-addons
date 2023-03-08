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
    float2* a2 = (float2*)a;
    float2* b2 = (float2*)b;

    float2 sum = {0.0, 0.0};
    for (size_t i = 0; i < size / 2; ++i) {
        sum = ipu::absadd(sum, a2[i] - b2[i]);
    }
    float res = sum[0] + sum[1];
    if (size % 2) res += ipu::fabs(a[size - 1] - b[size - 1]);
    return res;
}

float l1dist(const half* a, const half* b, size_t size) {
    half4* a4 = (half4*)a;
    half4* b4 = (half4*)b;

    half4 sum = {0.0, 0.0, 0.0, 0.0};
    for (size_t i = 0; i < size / 4; ++i) {
        sum = ipu::absadd(sum, a4[i] - b4[i]);
        sum += ipu::fabs(a4[i] - b4[i]);
    }
    float2 sum2 = ipu::sum(sum);
    float res = sum2[0] + sum2[1];
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
    poplar::Output<T> out;

    bool compute() {
        *out = T(l1dist(&a[0], &b[0], a.size()));
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
    float2* grad2 = (float2*)grad;
    float2* b2 = (float2*)b;
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
    half4* grad4 = (half4*)grad;
    half4* b4 = (half4*)b;
    half4 a4 = {a, a, a, a};

    for (size_t i = 0; i < size / 4; ++i) {
        sum += grad4[i] * signum(a4 - b4[i]);
    }
    float2 sum2 = ipu::sum(sum);
    float res = sum2[0] + sum2[1];
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
    poplar::Output<T> grad;

    bool compute() {
        *grad = l1distgrad(*a, &b[0], &gradOutput[0], b.size());
        return true;
    }
};

template class L1DistanceGradSingleVertex<float>;
template class L1DistanceGradSingleVertex<half>;

#ifdef __IPU__

float l2dist(const float* a, const float* b, size_t size) {
    float2* a2 = (float2*)a;
    float2* b2 = (float2*)b;

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
    half4* a4 = (half4*)a;
    half4* b4 = (half4*)b;

    float2 sum = {0.0, 0.0};
    for (size_t i = 0; i < size / 4; ++i) {
        half4 diff = a4[i] - b4[i];
        sum += ipu::sum(diff * diff);
    }
    float res = sum[0] + sum[1];
    size_t rem = size % 4;
    if (rem) {
        for (size_t i = size - rem; i < size; ++i) {
            float diff = float(a[i] - b[i]);
            res += diff * diff;
        }
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
    poplar::Output<T> out;

    bool compute() {
        *out = T(l2dist(&a[0], &b[0], a.size()));
        return true;
    }
};
template class L2DistanceSingleVertex<float>;
template class L2DistanceSingleVertex<half>;

template <typename T>
float l2distgrad(const T& a, const T* b, const T* dist, const T* grad, size_t size) {
    float sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        float diff = float(a - b[i]);
        float dist_i = float(dist[i]);
        float val = dist_i == 0.0f ? 0.0f : float(grad[i]) * diff / dist_i;
        sum += val;
    }
    return sum;
}

template <typename T>
class L2DistanceGradSingleVertex : public poplar::Vertex {
   public:
    poplar::Input<T> a;
    poplar::Input<poplar::Vector<T, SPAN, 8>> b;
    poplar::Input<poplar::Vector<T, ONE_PTR, 8>> dist;
    poplar::Input<poplar::Vector<T, ONE_PTR, 8>> gradOutput;
    poplar::Output<T> grad;

    bool compute() {
        *grad = l2distgrad(*a, &b[0], &dist[0], &gradOutput[0], b.size());
        return true;
    }
};

template class L2DistanceGradSingleVertex<float>;
template class L2DistanceGradSingleVertex<half>;
