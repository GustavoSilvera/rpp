
#pragma once

#include "base.h"

// The implementation of these functions are compiled with gcc/clang vector intrinsics.

#define IS_BIG_ENDIAN false

namespace rpp::SIMD {

template<i32 T>
struct F32x {
    using floatT = f32 __attribute__((ext_vector_type(T)));
    alignas(4 * T) floatT data;

    constexpr F32x<T>(floatT in = {}) : data(in){};

    [[nodiscard]] static F32x<T> set1(f32 v) noexcept;
    [[nodiscard]] static F32x<T> zero() noexcept;
    [[nodiscard]] static F32x<T> one() noexcept;
    [[nodiscard]] static F32x<T> add(F32x<T> a, F32x<T> b) noexcept;
    [[nodiscard]] static F32x<T> sub(F32x<T> a, F32x<T> b) noexcept;
    [[nodiscard]] static F32x<T> mul(F32x<T> a, F32x<T> b) noexcept;
    [[nodiscard]] static F32x<T> div(F32x<T> a, F32x<T> b) noexcept;
    [[nodiscard]] static F32x<T> min(F32x<T> a, F32x<T> b) noexcept;
    [[nodiscard]] static F32x<T> max(F32x<T> a, F32x<T> b) noexcept;
    [[nodiscard]] static F32x<T> floor(F32x<T> a) noexcept;
    [[nodiscard]] static F32x<T> ceil(F32x<T> a) noexcept;
    [[nodiscard]] static F32x<T> abs(F32x<T> a) noexcept;
    [[nodiscard]] static f32 dp(F32x<T> a, F32x<T> b) noexcept;
    [[nodiscard]] static i32 cmpeq(F32x<T> a, F32x<T> b) noexcept;

    /// NOTE: We should switch to the big-endian format of set(e0, e1, e2, e3)
    ///       which is natively supported by the clang vector extensions, rather
    ///       than the litte-endian set(e3, e2, e1, e0) imposed by the API of
    ///       _mm_set_ps. Once this is done we can set IS_BIG_ENDIAN to true.
#if IS_BIG_ENDIAN
    template<typename... Args> // Variadic function for setting the parameters
    [[nodiscard]] static F32x<T> set(Args... args) noexcept {
        return {(floatT){args...}};
    }
#else
    /// NOTE: the following only works for 4-element and 8-element vectors.
    ///       as I could not find a generic vector reversal mechanism.
    [[nodiscard]] static F32x<T> set(f32 a, f32 b, f32 c, f32 d) noexcept;
    [[nodiscard]] static F32x<T> set(f32 a, f32 b, f32 c, f32 d, f32 e, f32 f, f32 g,
                                     f32 h) noexcept;
#endif
};

typedef F32x<4> F32x4;
typedef F32x<8> F32x8;
} // namespace rpp::SIMD
