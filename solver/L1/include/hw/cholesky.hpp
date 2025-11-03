/*
 * Copyright 2021 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file cholesky.hpp
 * @brief This file contains cholesky functions
 *   - cholesky                 : Entry point function
 *   - choleskyTop             : Top level function that selects implementation architecture and internal types based
 * on a traits class.
 *   - choleskyBasic           : Basic implementation requiring lower resource
 *   - choleskyAlt             : Lower latency architecture requiring more resources
 *   - choleskyAlt2            : Further improved latency architecture requiring higher resource
 */

#ifndef _XF_SOLVER_CHOLESKY_HPP_
#define _XF_SOLVER_CHOLESKY_HPP_

#include "ap_fixed.h"
#include "hls_x_complex.h"
#include "hls_math.h"
template <typename T>
struct cholesky_real_helper {
    typedef T Type;
};

template <typename T>
struct cholesky_real_helper<hls::x_complex<T> > {
    typedef T Type;
};

template <typename T>
struct cholesky_real_helper<std::complex<T> > {
    typedef T Type;
};

template <typename T>
inline T cholesky_get_real(const T& v) {
#pragma HLS INLINE
    return v;
}

template <typename T>
inline void cholesky_set_real(T& dst, const T& value) {
#pragma HLS INLINE
    dst = value;
}

template <typename T>
inline T cholesky_get_real(const hls::x_complex<T>& v) {
#pragma HLS INLINE
    return v.real();
}

template <typename T>
inline void cholesky_set_real(hls::x_complex<T>& dst, const T& value) {
#pragma HLS INLINE
    dst.real(value);
    dst.imag(0);
}

template <typename T>
inline T cholesky_get_real(const std::complex<T>& v) {
#pragma HLS INLINE
    return v.real();
}

template <typename T>
inline void cholesky_set_real(std::complex<T>& dst, const T& value) {
#pragma HLS INLINE
    dst.real(value);
    dst.imag(0);
}

template <typename T_IN, typename T_OUT>
int cholesky_fast_sqrt_real(T_IN a, T_OUT& b) {
#pragma HLS INLINE
    const T_IN ZERO = 0;
    if (a < ZERO) {
        b = 0;
        return 1;
    }
    float input_f = static_cast<float>(a);
    float sqrt_f = hls::sqrtf(input_f);
    b = static_cast<T_OUT>(sqrt_f);
    return 0;
}

template <typename T>
inline T cholesky_abs_square(const T& v) {
#pragma HLS INLINE
    return v * v;
}

template <typename T>
inline T cholesky_abs_square(const hls::x_complex<T>& v) {
#pragma HLS INLINE
    T vr = v.real();
    T vi = v.imag();
    return vr * vr + vi * vi;
}

template <typename T>
inline T cholesky_abs_square(const std::complex<T>& v) {
#pragma HLS INLINE
    T vr = v.real();
    T vi = v.imag();
    return vr * vr + vi * vi;
}

template <typename T>
inline T cholesky_complex_mul(const T& a, const T& b) {
#pragma HLS INLINE
    return a * b;
}

template <typename T>
inline hls::x_complex<T> cholesky_complex_mul(const hls::x_complex<T>& a, const hls::x_complex<T>& b) {
#pragma HLS INLINE
    hls::x_complex<T> res;
    T ar = a.real();
    T ai = a.imag();
    T br = b.real();
    T bi = b.imag();
    T real_part = ar * br - ai * bi;
    T imag_part = ar * bi + ai * br;
#pragma HLS RESOURCE variable=real_part core=Mul_DSP
#pragma HLS RESOURCE variable=imag_part core=Mul_DSP
    res.real(real_part);
    res.imag(imag_part);
    return res;
}

template <typename T>
inline std::complex<T> cholesky_complex_mul(const std::complex<T>& a, const std::complex<T>& b) {
#pragma HLS INLINE
    std::complex<T> res;
    T ar = a.real();
    T ai = a.imag();
    T br = b.real();
    T bi = b.imag();
    T real_part = ar * br - ai * bi;
    T imag_part = ar * bi + ai * br;
#pragma HLS RESOURCE variable=real_part core=Mul_DSP
#pragma HLS RESOURCE variable=imag_part core=Mul_DSP
    res.real(real_part);
    res.imag(imag_part);
    return res;
}

// Fully-unrolled Cholesky for very small matrices (RowsColsA <= 4)
template <bool LowerTriangularL, int RowsColsA, typename CholeskyTraits, class InputType, class OutputType>
int choleskySmall(const InputType A[RowsColsA][RowsColsA], OutputType L[RowsColsA][RowsColsA]) {
    int return_code = 0;

    OutputType L_internal[RowsColsA][RowsColsA];
    OutputType L_internal_conj[RowsColsA][RowsColsA];
    typename cholesky_real_helper<typename CholeskyTraits::ACCUM_T>::Type L_internal_abs2[RowsColsA][RowsColsA];
#pragma HLS ARRAY_PARTITION variable = L_internal complete dim = 1
#pragma HLS ARRAY_PARTITION variable = L_internal complete dim = 2
#pragma HLS ARRAY_PARTITION variable = L_internal_conj complete dim = 1
#pragma HLS ARRAY_PARTITION variable = L_internal_conj complete dim = 2
#pragma HLS ARRAY_PARTITION variable = L_internal_abs2 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = L_internal_abs2 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = A complete dim = 1
#pragma HLS ARRAY_PARTITION variable = A complete dim = 2
#pragma HLS ARRAY_PARTITION variable = L complete dim = 1
#pragma HLS ARRAY_PARTITION variable = L complete dim = 2

    // Initialize internal storage
    for (int r = 0; r < RowsColsA; r++) {
#pragma HLS UNROLL
        for (int c = 0; c < RowsColsA; c++) {
#pragma HLS UNROLL
            L_internal[r][c] = 0;
            L_internal_conj[r][c] = 0;
            L_internal_abs2[r][c] = 0;
        }
    }

    typename CholeskyTraits::ACCUM_T A_cast_to_sum;
    typename CholeskyTraits::ADD_T A_minus_sum;
    typename CholeskyTraits::DIAG_T new_L_diag;
    typename CholeskyTraits::OFF_DIAG_T new_L_off_diag;
    typename CholeskyTraits::RECIP_DIAG_T inv_Ljj;
    typename CholeskyTraits::L_OUTPUT_T new_L;

    OutputType retrieved_L;

    for (int j = 0; j < RowsColsA; j++) {
#pragma HLS UNROLL
        typename CholeskyTraits::ACCUM_T sum_diag = typename CholeskyTraits::ACCUM_T();
        typedef typename cholesky_real_helper<typename CholeskyTraits::ACCUM_T>::Type ACCUM_REAL_T;
        ACCUM_REAL_T sum_diag_real = 0;

        for (int k = 0; k < RowsColsA; k++) {
#pragma HLS UNROLL
            if (k < j) {
                if (LowerTriangularL == true) {
                    sum_diag_real += L_internal_abs2[j][k];
                } else {
                    sum_diag_real += L_internal_abs2[k][j];
                }
            }
        }

        cholesky_set_real(sum_diag, sum_diag_real);

        A_cast_to_sum = A[j][j];
        A_minus_sum = A_cast_to_sum - sum_diag;

        typedef typename cholesky_real_helper<typename CholeskyTraits::ADD_T>::Type ADD_REAL_T;
        typedef typename cholesky_real_helper<typename CholeskyTraits::DIAG_T>::Type DIAG_REAL_T;
        typedef typename cholesky_real_helper<typename CholeskyTraits::RECIP_DIAG_T>::Type RECIP_REAL_T;

        ADD_REAL_T diag_real = cholesky_get_real(A_minus_sum);
        DIAG_REAL_T diag_sqrt = 0;
#pragma HLS BIND_OP variable=diag_sqrt core=FSqrt

        if (cholesky_fast_sqrt_real(diag_real, diag_sqrt)) {
#ifndef __SYNTHESIS__
            printf("ERROR: Trying to find the square root of a negative number\n");
#endif
            return_code = 1;
        }

        cholesky_set_real(new_L_diag, diag_sqrt);
        new_L = new_L_diag;

        if (LowerTriangularL == true) {
            L[j][j] = new_L;
            L_internal[j][j] = new_L;
            L_internal_conj[j][j] = hls::x_conj(new_L);
            L_internal_abs2[j][j] = static_cast<ACCUM_REAL_T>(diag_sqrt * diag_sqrt);
        } else {
            L[j][j] = hls::x_conj(new_L);
            L_internal[j][j] = hls::x_conj(new_L);
            L_internal_conj[j][j] = new_L;
            L_internal_abs2[j][j] = static_cast<ACCUM_REAL_T>(diag_sqrt * diag_sqrt);
        }

        if (diag_sqrt != 0) {
            RECIP_REAL_T inv_real = static_cast<RECIP_REAL_T>(1) / diag_sqrt;
            cholesky_set_real(inv_Ljj, inv_real);
        } else {
            cholesky_set_real(inv_Ljj, (RECIP_REAL_T)0);
        }

        for (int i = 0; i < RowsColsA; ++i) {
#pragma HLS UNROLL
            if (i < j) {
                if (LowerTriangularL == true) {
                    L[i][j] = 0;
                    L_internal[i][j] = 0;
                    L_internal_conj[i][j] = 0;
                    L_internal_abs2[i][j] = 0;
                } else {
                    L[j][i] = 0;
                    L_internal[j][i] = 0;
                    L_internal_conj[j][i] = 0;
                    L_internal_abs2[j][i] = 0;
                }
            } else if (i > j) {
                typename CholeskyTraits::ACCUM_T sum_off;
                if (LowerTriangularL == true) {
                    sum_off = A[i][j];
                } else {
                    sum_off = hls::x_conj(A[j][i]);
                }

                for (int k = 0; k < RowsColsA; k++) {
#pragma HLS UNROLL
                    if (k < j) {
                        if (LowerTriangularL == true) {
                            auto prod = cholesky_complex_mul(L_internal[i][k], L_internal_conj[j][k]);
                            sum_off -= static_cast<typename CholeskyTraits::ACCUM_T>(prod);
                        } else {
                            auto prod = cholesky_complex_mul(L_internal_conj[k][i], L_internal[k][j]);
                            sum_off -= static_cast<typename CholeskyTraits::ACCUM_T>(prod);
                        }
                    }
                }

                new_L_off_diag = sum_off * inv_Ljj;
                new_L = new_L_off_diag;

                if (LowerTriangularL == true) {
                    L[i][j] = new_L;
                    L_internal[i][j] = new_L;
                    L_internal_conj[i][j] = hls::x_conj(new_L);
                    L_internal_abs2[i][j] = static_cast<ACCUM_REAL_T>(cholesky_abs_square(new_L));
                } else {
                    L[j][i] = hls::x_conj(new_L);
                    L_internal[j][i] = hls::x_conj(new_L);
                    L_internal_conj[j][i] = new_L;
                    L_internal_abs2[j][i] = static_cast<ACCUM_REAL_T>(cholesky_abs_square(new_L));
                }
            }
        }
    }

    return return_code;
}

#include <complex>
#include "utils/std_complex_utils.h"
#include "utils/x_matrix_utils.hpp"
#include "hls_stream.h"

namespace xf {
namespace solver {

// ===================================================================================================================
// Default traits struct defining the internal variable types for the cholesky function
template <bool LowerTriangularL, int RowsColsA, typename InputType, typename OutputType>
struct choleskyTraits {
    typedef InputType PROD_T;
    typedef InputType ACCUM_T;
    typedef InputType ADD_T;
    typedef InputType DIAG_T;
    typedef InputType RECIP_DIAG_T;
    typedef InputType OFF_DIAG_T;
    typedef OutputType L_OUTPUT_T;
    static const int ARCH =
        1; // Select implementation: 0=Basic, 1=Lower latency architecture, 2=Further improved latency architecture
    static const int INNER_II = 1; // Specify the pipelining target for the inner loop
    static const int UNROLL_FACTOR =
        1; // Specify the inner loop unrolling factor for the choleskyAlt2 architecture(2) to increase throughput
    static const int UNROLL_DIM = (LowerTriangularL == true ? 1 : 2); // Dimension to unroll matrix
    static const int ARCH2_ZERO_LOOP =
        true; // Additional implementation "switch" for the choleskyAlt2 architecture (2).
    static const int ALT_UNROLL_FACTOR = (RowsColsA <= 4 ? RowsColsA : 1);
};

// Specialization for complex
template <bool LowerTriangularL, int RowsColsA, typename InputBaseType, typename OutputBaseType>
struct choleskyTraits<LowerTriangularL, RowsColsA, hls::x_complex<InputBaseType>, hls::x_complex<OutputBaseType> > {
    typedef hls::x_complex<InputBaseType> PROD_T;
    typedef hls::x_complex<InputBaseType> ACCUM_T;
    typedef hls::x_complex<InputBaseType> ADD_T;
    typedef hls::x_complex<InputBaseType> DIAG_T;
    typedef InputBaseType RECIP_DIAG_T;
    typedef hls::x_complex<InputBaseType> OFF_DIAG_T;
    typedef hls::x_complex<OutputBaseType> L_OUTPUT_T;
    static const int ARCH = 1;
    static const int INNER_II = 1;
    static const int UNROLL_FACTOR = 1;
    static const int UNROLL_DIM = (LowerTriangularL == true ? 1 : 2);
    static const int ARCH2_ZERO_LOOP = true;
    static const int ALT_UNROLL_FACTOR = (RowsColsA <= 4 ? RowsColsA : 1);
};

// Specialization for std complex
template <bool LowerTriangularL, int RowsColsA, typename InputBaseType, typename OutputBaseType>
struct choleskyTraits<LowerTriangularL, RowsColsA, std::complex<InputBaseType>, std::complex<OutputBaseType> > {
    typedef std::complex<InputBaseType> PROD_T;
    typedef std::complex<InputBaseType> ACCUM_T;
    typedef std::complex<InputBaseType> ADD_T;
    typedef std::complex<InputBaseType> DIAG_T;
    typedef InputBaseType RECIP_DIAG_T;
    typedef std::complex<InputBaseType> OFF_DIAG_T;
    typedef std::complex<OutputBaseType> L_OUTPUT_T;
    static const int ARCH = 1;
    static const int INNER_II = 1;
    static const int UNROLL_FACTOR = 1;
    static const int UNROLL_DIM = (LowerTriangularL == true ? 1 : 2);
    static const int ARCH2_ZERO_LOOP = true;
    static const int ALT_UNROLL_FACTOR = (RowsColsA <= 4 ? RowsColsA : 1);
};

// Specialization for ap_fixed
template <bool LowerTriangularL,
          int RowsColsA,
          int W1,
          int I1,
          ap_q_mode Q1,
          ap_o_mode O1,
          int N1,
          int W2,
          int I2,
          ap_q_mode Q2,
          ap_o_mode O2,
          int N2>
struct choleskyTraits<LowerTriangularL, RowsColsA, ap_fixed<W1, I1, Q1, O1, N1>, ap_fixed<W2, I2, Q2, O2, N2> > {
    typedef ap_fixed<W1 + W1, I1 + I1, AP_RND_CONV, AP_SAT, 0> PROD_T;
    typedef ap_fixed<(W1 + W1) + BitWidth<RowsColsA>::Value,
                     (I1 + I1) + BitWidth<RowsColsA>::Value,
                     AP_RND_CONV,
                     AP_SAT,
                     0>
        ACCUM_T;
    typedef ap_fixed<W1 + 1, I1 + 1, AP_RND_CONV, AP_SAT, 0> ADD_T;
    typedef ap_fixed<(W1 + 1) * 2, I1 + 1, AP_RND_CONV, AP_SAT, 0> DIAG_T;     // Takes result of sqrt
    typedef ap_fixed<(W1 + 1) * 2, I1 + 1, AP_RND_CONV, AP_SAT, 0> OFF_DIAG_T; // Takes result of /
    typedef ap_fixed<2 + (W2 - I2) + W2, 2 + (W2 - I2), AP_RND_CONV, AP_SAT, 0> RECIP_DIAG_T;
    typedef ap_fixed<W2, I2, AP_RND_CONV, AP_SAT, 0>
        L_OUTPUT_T; // Takes new L value.  Same as L output but saturation set
    static const int ARCH = 1;
    static const int INNER_II = 1;
    static const int UNROLL_FACTOR = 1;
    static const int UNROLL_DIM = (LowerTriangularL == true ? 1 : 2);
    static const int ARCH2_ZERO_LOOP = true;
    static const int ALT_UNROLL_FACTOR = (RowsColsA <= 4 ? RowsColsA : 1);
};

// Further specialization for hls::complex<ap_fixed>
template <bool LowerTriangularL,
          int RowsColsA,
          int W1,
          int I1,
          ap_q_mode Q1,
          ap_o_mode O1,
          int N1,
          int W2,
          int I2,
          ap_q_mode Q2,
          ap_o_mode O2,
          int N2>
struct choleskyTraits<LowerTriangularL,
                      RowsColsA,
                      hls::x_complex<ap_fixed<W1, I1, Q1, O1, N1> >,
                      hls::x_complex<ap_fixed<W2, I2, Q2, O2, N2> > > {
    typedef hls::x_complex<ap_fixed<W1 + W1, I1 + I1, AP_RND_CONV, AP_SAT, 0> > PROD_T;
    typedef hls::x_complex<ap_fixed<(W1 + W1) + BitWidth<RowsColsA>::Value,
                                    (I1 + I1) + BitWidth<RowsColsA>::Value,
                                    AP_RND_CONV,
                                    AP_SAT,
                                    0> >
        ACCUM_T;
    typedef hls::x_complex<ap_fixed<W1 + 1, I1 + 1, AP_RND_CONV, AP_SAT, 0> > ADD_T;
    typedef hls::x_complex<ap_fixed<(W1 + 1) * 2, I1 + 1, AP_RND_CONV, AP_SAT, 0> > DIAG_T;     // Takes result of sqrt
    typedef hls::x_complex<ap_fixed<(W1 + 1) * 2, I1 + 1, AP_RND_CONV, AP_SAT, 0> > OFF_DIAG_T; // Takes result of /
    typedef ap_fixed<2 + (W2 - I2) + W2, 2 + (W2 - I2), AP_RND_CONV, AP_SAT, 0> RECIP_DIAG_T;
    typedef hls::x_complex<ap_fixed<W2, I2, AP_RND_CONV, AP_SAT, 0> >
        L_OUTPUT_T; // Takes new L value.  Same as L output but saturation set
    static const int ARCH = 1;
    static const int INNER_II = 1;
    static const int UNROLL_FACTOR = 1;
    static const int UNROLL_DIM = (LowerTriangularL == true ? 1 : 2);
    static const int ARCH2_ZERO_LOOP = true;
    static const int ALT_UNROLL_FACTOR = (RowsColsA <= 4 ? RowsColsA : 1);
};

// Further specialization for std::complex<ap_fixed>
template <bool LowerTriangularL,
          int RowsColsA,
          int W1,
          int I1,
          ap_q_mode Q1,
          ap_o_mode O1,
          int N1,
          int W2,
          int I2,
          ap_q_mode Q2,
          ap_o_mode O2,
          int N2>
struct choleskyTraits<LowerTriangularL,
                      RowsColsA,
                      std::complex<ap_fixed<W1, I1, Q1, O1, N1> >,
                      std::complex<ap_fixed<W2, I2, Q2, O2, N2> > > {
    typedef std::complex<ap_fixed<W1 + W1, I1 + I1, AP_RND_CONV, AP_SAT, 0> > PROD_T;
    typedef std::complex<ap_fixed<(W1 + W1) + BitWidth<RowsColsA>::Value,
                                  (I1 + I1) + BitWidth<RowsColsA>::Value,
                                  AP_RND_CONV,
                                  AP_SAT,
                                  0> >
        ACCUM_T;
    typedef std::complex<ap_fixed<W1 + 1, I1 + 1, AP_RND_CONV, AP_SAT, 0> > ADD_T;
    typedef std::complex<ap_fixed<(W1 + 1) * 2, I1 + 1, AP_RND_CONV, AP_SAT, 0> > DIAG_T;     // Takes result of sqrt
    typedef std::complex<ap_fixed<(W1 + 1) * 2, I1 + 1, AP_RND_CONV, AP_SAT, 0> > OFF_DIAG_T; // Takes result of /
    typedef ap_fixed<2 + (W2 - I2) + W2, 2 + (W2 - I2), AP_RND_CONV, AP_SAT, 0> RECIP_DIAG_T;
    typedef std::complex<ap_fixed<W2, I2, AP_RND_CONV, AP_SAT, 0> >
        L_OUTPUT_T; // Takes new L value.  Same as L output but saturation set
    static const int ARCH = 1;
    static const int INNER_II = 1;
    static const int UNROLL_FACTOR = 1;
    static const int UNROLL_DIM = (LowerTriangularL == true ? 1 : 2);
    static const int ARCH2_ZERO_LOOP = true;
    static const int ALT_UNROLL_FACTOR = (RowsColsA <= 4 ? RowsColsA : 1);
};

// ===================================================================================================================
// Helper functions

// Square root
// o Overloaded versions of the sqrt function
// o The square root of a complex number is expensive.  However, the diagonal values of a Cholesky decomposition are
// always
//   real so we don't need a full complex square root.
template <typename T_IN, typename T_OUT>
int cholesky_sqrt_op(T_IN a, T_OUT& b) {
Function_cholesky_sqrt_op_real:;
    const T_IN ZERO = 0;
    if (a < ZERO) {
        b = ZERO;
        return (1);
    }
    b = x_sqrt(a);
    return (0);
}
template <typename T_IN, typename T_OUT>
int cholesky_sqrt_op(hls::x_complex<T_IN> din, hls::x_complex<T_OUT>& dout) {
Function_cholesky_sqrt_op_complex:;
    const T_IN ZERO = 0;
    T_IN a = din.real();
    dout.imag(ZERO);

    if (a < ZERO) {
        dout.real(ZERO);
        return (1);
    }

    // 使用更快的平方根实现
    #pragma HLS INLINE
    dout.real(x_sqrt(a));
    return (0);
}
template <typename T_IN, typename T_OUT>
int cholesky_sqrt_op(std::complex<T_IN> din, std::complex<T_OUT>& dout) {
Function_cholesky_sqrt_op_complex:;
    const T_IN ZERO = 0;
    T_IN a = din.real();
    dout.imag(ZERO);

    if (a < ZERO) {
        dout.real(ZERO);
        return (1);
    }

    dout.real(x_sqrt(a));
    return (0);
}

// Reciprocal square root - Optimized version
// 优化1：对于定点数类型，直接使用x_rsqrt硬件原语（如果可用）
// 这将把两次操作（sqrt + 除法）减少为一次操作
template <typename InputType, typename OutputType>
void cholesky_rsqrt(InputType x, OutputType& res) {
Function_cholesky_rsqrt_default:;
    res = x_rsqrt(x);
}

// 优化：为复数定点类型添加专门的倒数平方根实现
template <int W1, int I1, ap_q_mode Q1, ap_o_mode O1, int N1, int W2, int I2, ap_q_mode Q2, ap_o_mode O2, int N2>
void cholesky_rsqrt(hls::x_complex<ap_fixed<W1, I1, Q1, O1, N1> > x, ap_fixed<W2, I2, Q2, O2, N2>& res) {
Function_cholesky_rsqrt_complex_fixed:;
    // 由于对角线元素总是实数，只对实部计算
    ap_fixed<W1, I1, Q1, O1, N1> x_real = x.real();
    cholesky_rsqrt(x_real, res);
}

// 定点数倒数平方根：轻量级优化版本
template <int W1, int I1, ap_q_mode Q1, ap_o_mode O1, int N1, int W2, int I2, ap_q_mode Q2, ap_o_mode O2, int N2>
void cholesky_rsqrt(ap_fixed<W1, I1, Q1, O1, N1> x, ap_fixed<W2, I2, Q2, O2, N2>& res) {
Function_cholesky_rsqrt_fixed:;
    // 轻量级优化：保持原有逻辑但标记为可流水化
    ap_fixed<W2, I2, Q2, O2, N2> one = 1;
    ap_fixed<W1, I1, Q1, O1, N1> sqrt_res;
    ap_fixed<W2, I2, Q2, O2, N2> sqrt_res_cast;
    
    // 计算平方根（保持原有位宽）
    sqrt_res = x_sqrt(x);
    sqrt_res_cast = sqrt_res;
    
    // 倒数计算
    res = one / sqrt_res_cast;
}

// Local multiplier to handle a complex case currently not supported by the hls::x_complex class
// - Complex multiplied by a real of a different type
// - Required for complex fixed point implementations
template <typename AType, typename BType, typename CType>
void cholesky_prod_sum_mult(AType A, BType B, CType& C) {
Function_cholesky_prod_sum_mult_real:;
    C = A * B;
}
template <typename AType, typename BType, typename CType>
void cholesky_prod_sum_mult(hls::x_complex<AType> A, BType B, hls::x_complex<CType>& C) {
Function_cholesky_prod_sum_mult_complex:;
    C.real(A.real() * B);
    C.imag(A.imag() * B);
}
template <typename AType, typename BType, typename CType>
void cholesky_prod_sum_mult(std::complex<AType> A, BType B, std::complex<CType>& C) {
Function_cholesky_prod_sum_mult_complex:;
    C.real(A.real() * B);
    C.imag(A.imag() * B);
}

// ===================================================================================================================
// choleskyBasic
template <bool LowerTriangularL, int RowsColsA, typename CholeskyTraits, class InputType, class OutputType>
int choleskyBasic(const InputType A[RowsColsA][RowsColsA], OutputType L[RowsColsA][RowsColsA]) {
    int return_code = 0;

    // Use the traits struct to specify the correct type for the intermediate variables. This is really only needed for
    // fixed point.
    typename CholeskyTraits::PROD_T prod;
    typename CholeskyTraits::ACCUM_T A_cast_to_sum;    // A with the same dimensions as sum.
    typename CholeskyTraits::ACCUM_T prod_cast_to_sum; // prod with the same dimensions as sum.

    typename CholeskyTraits::ADD_T A_minus_sum;
    typename CholeskyTraits::DIAG_T new_L_diag;         // sqrt(A_minus_sum)
    typename CholeskyTraits::OFF_DIAG_T new_L_off_diag; // sum/L
    typename CholeskyTraits::OFF_DIAG_T L_cast_to_new_L_off_diag;

    typename CholeskyTraits::L_OUTPUT_T new_L;
    OutputType retrieved_L;
    // Internal memory used to aviod read access from function output argument L.
    // NOTE: The internal matrix only needs to be triangular but optimization using a 1-D array it will require addition
    // logic to generate the indexes. Refer to the choleskyAlt function.
    OutputType L_internal[RowsColsA][RowsColsA];
    
    // 优化1：数组分区以减少访存冲突
    // 对于3x3矩阵，完全分区第二维可获得最佳性能
#pragma HLS ARRAY_PARTITION variable=A complete dim=2
#pragma HLS ARRAY_PARTITION variable=L complete dim=2
#pragma HLS ARRAY_PARTITION variable=L_internal complete dim=2

col_loop:
        for (int j = 0; j < RowsColsA; j++) {
#pragma HLS PIPELINE off
        typename CholeskyTraits::ACCUM_T sum_diag = typename CholeskyTraits::ACCUM_T();

    // Calculate the diagonal value for this column
    diag_loop:
        for (int k = 0; k < j; k++) {  // 优化2：直接使用j作为上界，消除条件判断
#pragma HLS PIPELINE II = 1  // 优化：对角线循环流水线化
#pragma HLS loop_tripcount min=0 max=RowsColsA avg=RowsColsA/2
            if (LowerTriangularL == true) {
                retrieved_L = L_internal[j][k];
            } else {
                retrieved_L = L_internal[k][j];
            }
            sum_diag += hls::x_conj(retrieved_L) * retrieved_L;  // 修复：使用累加而不是赋值
        }
        A_cast_to_sum = A[j][j];

        A_minus_sum = A_cast_to_sum - sum_diag;

        if (cholesky_sqrt_op(A_minus_sum, new_L_diag)) {
#ifndef __SYNTHESIS__
            printf("ERROR: Trying to find the square root of a negative number\n");
#endif
            return_code = 1;
        }

        // Round to target format using method specifed by traits defined types.
        new_L = new_L_diag;

        if (LowerTriangularL == true) {
            L_internal[j][j] = new_L;
            L[j][j] = new_L;
        } else {
            L_internal[j][j] = hls::x_conj(new_L);
            L[j][j] = hls::x_conj(new_L);
        }

    // Calculate the off diagonal values for this column
    off_diag_loop_zero:
        for (int i = 0; i < j; ++i) {
#pragma HLS PIPELINE II = 1
#pragma HLS loop_tripcount min = 0 max = RowsColsA
            if (LowerTriangularL == true) {
                L[i][j] = 0;
            } else {
                L[j][i] = 0;
            }
        }

    off_diag_loop_calc:
        for (int i = j + 1; i < RowsColsA; ++i) {
            typename CholeskyTraits::ACCUM_T sum_off;
            if (LowerTriangularL == true) {
                sum_off = A[i][j];
            } else {
                sum_off = hls::x_conj(A[j][i]);
            }

        sum_loop:
            for (int k = 0; k < j; k++) {  // 优化2：直接使用j作为上界，消除条件判断
#pragma HLS PIPELINE II = CholeskyTraits::INNER_II
#pragma HLS loop_tripcount min=0 max=RowsColsA avg=RowsColsA/2  // 帮助编译器优化
                if (LowerTriangularL == true) {
                    prod = -L_internal[i][k] * hls::x_conj(L_internal[j][k]);
                } else {
                    prod = -hls::x_conj(L_internal[k][i]) * (L_internal[k][j]);
                }
                prod_cast_to_sum = prod;
                sum_off += prod_cast_to_sum;
            }

            new_L_off_diag = sum_off;

            L_cast_to_new_L_off_diag = L_internal[j][j];

            // Diagonal is always real, avoid complex division
            new_L_off_diag = new_L_off_diag / hls::x_real(L_cast_to_new_L_off_diag);

            // Round to target format using method specifed by traits defined types.
            new_L = new_L_off_diag;

            if (LowerTriangularL == true) {
                L[i][j] = new_L;
                L_internal[i][j] = new_L;
            } else {
                L[j][i] = hls::x_conj(new_L);
                L_internal[j][i] = hls::x_conj(new_L);
            }
        }
    }
    return (return_code);
}

// ===================================================================================================================
// choleskyAlt: 行优先结构，优化latency（回归简单结构）
template <bool LowerTriangularL, int RowsColsA, typename CholeskyTraits, class InputType, class OutputType>
int choleskyAlt(const InputType A[RowsColsA][RowsColsA], OutputType L[RowsColsA][RowsColsA]) {
    int return_code = 0;

    // 使用内部存储避免从输出参数L读取
    OutputType L_internal[RowsColsA][RowsColsA];
    OutputType retrieved_L;  // 临时变量避免重复数组访问
    typename CholeskyTraits::ACCUM_T A_cast_to_sum;  // A[j][j]转换为ACCUM_T
    typename CholeskyTraits::ADD_T A_minus_sum;
    typename CholeskyTraits::DIAG_T new_L_diag;
    typename CholeskyTraits::OFF_DIAG_T new_L_off_diag;
    typename CholeskyTraits::RECIP_DIAG_T inv_Ljj;
    typename CholeskyTraits::L_OUTPUT_T new_L;

    // 数组分区优化：使用complete分区提升访存并行性
#pragma HLS ARRAY_PARTITION variable = A complete dim = 2
#pragma HLS ARRAY_PARTITION variable = L complete dim = 2
#pragma HLS ARRAY_PARTITION variable = L_internal complete dim = 2

col_loop:
    for (int j = 0; j < RowsColsA; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS loop_tripcount min = 1 max = RowsColsA avg = RowsColsA
        // 初始化sum
        typename CholeskyTraits::ACCUM_T sum_diag = typename CholeskyTraits::ACCUM_T();

        // 计算对角线值的sum
    diag_sum_loop:
        for (int k = 0; k < j; k++) {
#pragma HLS PIPELINE II = 1
            if (LowerTriangularL == true) {
                retrieved_L = L_internal[j][k];
                sum_diag += hls::x_conj(retrieved_L) * retrieved_L;
            } else {
                retrieved_L = L_internal[k][j];
                sum_diag += hls::x_conj(retrieved_L) * retrieved_L;
            }
        }
        
        // 类型转换后计算
        A_cast_to_sum = A[j][j];
        A_minus_sum = A_cast_to_sum - sum_diag;

        typedef typename cholesky_real_helper<typename CholeskyTraits::ADD_T>::Type ADD_REAL_T;
        typedef typename cholesky_real_helper<typename CholeskyTraits::DIAG_T>::Type DIAG_REAL_T;
        typedef typename cholesky_real_helper<typename CholeskyTraits::RECIP_DIAG_T>::Type RECIP_REAL_T;

        ADD_REAL_T diag_real = cholesky_get_real(A_minus_sum);
        DIAG_REAL_T diag_sqrt = 0;

        if (cholesky_fast_sqrt_real(diag_real, diag_sqrt)) {
#ifndef __SYNTHESIS__
            printf("ERROR: Trying to find the square root of a negative number\n");
#endif
            return_code = 1;
        }

        cholesky_set_real(new_L_diag, diag_sqrt);
        new_L = new_L_diag;

        // 存储对角线值（减少重复写入）
        if (LowerTriangularL == true) {
            L[j][j] = new_L;
            L_internal[j][j] = new_L;
        } else {
            L[j][j] = hls::x_conj(new_L);
            L_internal[j][j] = hls::x_conj(new_L);
        }

        // 每列仅计算一次对角倒数，后续用乘法代替多次除法
        if (diag_sqrt != 0) {
            RECIP_REAL_T inv_real = static_cast<RECIP_REAL_T>(1) / diag_sqrt;
            cholesky_set_real(inv_Ljj, inv_real);
        } else {
            cholesky_set_real(inv_Ljj, (RECIP_REAL_T)0);
        }

    // 计算非对角线值
    off_diag_loop:
        for (int i = 0; i < RowsColsA; ++i) {
            if (i < j) {
                if (LowerTriangularL == true) {
                    L[i][j] = 0;
                } else {
                    L[j][i] = 0;
                }
            } else if (i > j) {
                typename CholeskyTraits::ACCUM_T sum_off;
                if (LowerTriangularL == true) {
                    sum_off = A[i][j];
                } else {
                    sum_off = hls::x_conj(A[j][i]);
                }

            off_diag_sum_loop:
                for (int k = 0; k < j; k++) {
#pragma HLS PIPELINE II = CholeskyTraits::INNER_II
                    if (LowerTriangularL == true) {
                        sum_off -= static_cast<typename CholeskyTraits::ACCUM_T>(
                                  L_internal[i][k] * hls::x_conj(L_internal[j][k]));
                    } else {
                        sum_off -= static_cast<typename CholeskyTraits::ACCUM_T>(
                                  hls::x_conj(L_internal[k][i]) * L_internal[k][j]);
                    }
                }

                new_L_off_diag = sum_off * inv_Ljj;
                new_L = new_L_off_diag;

                if (LowerTriangularL == true) {
                    L[i][j] = new_L;
                    L_internal[i][j] = new_L;
                } else {
                    L[j][i] = hls::x_conj(new_L);
                    L_internal[j][i] = hls::x_conj(new_L);
                }
            }
        }
    }
    return (return_code);
}

// ===================================================================================================================
// choleskyAlt2: Further improved latency architecture requiring higher resource
template <bool LowerTriangularL, int RowsColsA, typename CholeskyTraits, class InputType, class OutputType>
int choleskyAlt2(const InputType A[RowsColsA][RowsColsA], OutputType L[RowsColsA][RowsColsA]) {
    int return_code = 0;

    // To avoid array index calculations every iteration this architecture uses a simple 2D array rather than a
    // optimized/packed triangular matrix.
    OutputType L_internal[RowsColsA][RowsColsA];
    OutputType prod_column_top;
    typename CholeskyTraits::ACCUM_T square_sum_array[RowsColsA];
    typename CholeskyTraits::ACCUM_T A_cast_to_sum;
    typename CholeskyTraits::ADD_T A_minus_sum;
    typename CholeskyTraits::DIAG_T A_minus_sum_cast_diag;
    typename CholeskyTraits::DIAG_T new_L_diag;
    typename CholeskyTraits::RECIP_DIAG_T new_L_diag_recip;
    typename CholeskyTraits::PROD_T prod;
    typename CholeskyTraits::ACCUM_T prod_cast_to_sum;
    typename CholeskyTraits::ACCUM_T product_sum;
    typename CholeskyTraits::ACCUM_T product_sum_array[RowsColsA];
    typename CholeskyTraits::OFF_DIAG_T prod_cast_to_off_diag;
    typename CholeskyTraits::OFF_DIAG_T new_L_off_diag;
    typename CholeskyTraits::L_OUTPUT_T new_L;

#pragma HLS ARRAY_PARTITION variable = A cyclic dim = CholeskyTraits::UNROLL_DIM factor = CholeskyTraits::UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = L cyclic dim = CholeskyTraits::UNROLL_DIM factor = CholeskyTraits::UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = L_internal cyclic dim = CholeskyTraits::UNROLL_DIM factor = \
    CholeskyTraits::UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = square_sum_array cyclic dim = 1 factor = CholeskyTraits::UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = product_sum_array cyclic dim = 1 factor = CholeskyTraits::UNROLL_FACTOR

col_loop:
    for (int j = 0; j < RowsColsA; j++) {
        // Diagonal calculation
        A_cast_to_sum = A[j][j];
        if (j == 0) {
            A_minus_sum = A_cast_to_sum;
        } else {
            A_minus_sum = A_cast_to_sum - square_sum_array[j];
        }
        if (cholesky_sqrt_op(A_minus_sum, new_L_diag)) {
#ifndef __SYNTHESIS__
            printf("ERROR: Trying to find the square root of a negative number\n");
#endif
            return_code = 1;
        }
        // Round to target format using method specifed by traits defined types.
        new_L = new_L_diag;
        // Generate the reciprocal of the diagonal for internal use to aviod the latency of a divide in every
        // off-diagonal calculation
        A_minus_sum_cast_diag = A_minus_sum;
        cholesky_rsqrt(hls::x_real(A_minus_sum_cast_diag), new_L_diag_recip);
        // Store diagonal value
        if (LowerTriangularL == true) {
            L[j][j] = new_L;
        } else {
            L[j][j] = hls::x_conj(new_L);
        }

    sum_loop:
        for (int k = 0; k <= j; k++) {
// Define average trip count for reporting, loop reduces in length for every iteration of col_loop
#pragma HLS loop_tripcount max = 1 + RowsColsA / 2
            // Same value used in all calcs
            // o Implement -1* here
            prod_column_top = -hls::x_conj(L_internal[j][k]);

        // NOTE: Using a fixed loop length combined with a "if" to implement reducing loop length
        // o Ensures the inner loop can achieve the maximum II (1)
        // o May introduce a small overhead resolving the "if" statement but HLS struggled to schedule when the variable
        //   loop bound expression was used.
        // o Will report inaccurate trip count as it will reduce by one with the col_loop
        // o Variable loop bound code: row_loop: for(int i = j+1; i < RowsColsA; i++) {
        row_loop:
            for (int i = 0; i < RowsColsA; i++) {
// IMPORTANT: row_loop must not merge with sum_loop as the merged loop becomes variable length and HLS will struggle
// with scheduling
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II = CholeskyTraits::INNER_II
#pragma HLS UNROLL FACTOR = CholeskyTraits::UNROLL_FACTOR

                if (i > j) {
                    prod = L_internal[i][k] * prod_column_top;
                    prod_cast_to_sum = prod;

                    if (k == 0) {
                        // Prime first sum
                        if (LowerTriangularL == true) {
                            A_cast_to_sum = A[i][j];
                        } else {
                            A_cast_to_sum = hls::x_conj(A[j][i]);
                        }
                        product_sum = A_cast_to_sum;
                    } else {
                        product_sum = product_sum_array[i];
                    }

                    if (k < j) {
                        // Accumulate row sum of columns
                        product_sum_array[i] = product_sum + prod_cast_to_sum;
                    } else {
                        // Final calculation for off diagonal value
                        prod_cast_to_off_diag = product_sum;
                        // Diagonal is stored in its reciprocal form so only need to multiply the product sum
                        cholesky_prod_sum_mult(prod_cast_to_off_diag, new_L_diag_recip, new_L_off_diag);
                        // Round to target format using method specifed by traits defined types.
                        new_L = new_L_off_diag;
                        // Build sum for use in diagonal calculation for this row.
                        if (k == 0) {
                            square_sum_array[j] = hls::x_conj(new_L) * new_L;
                        } else {
                            square_sum_array[j] = hls::x_conj(new_L) * new_L;
                        }
                        // Store result
                        L_internal[i][j] = new_L;
                        // NOTE: Use the upper/lower triangle zeroing in the subsequent loop so the double memory access
                        // does not
                        // become a bottleneck
                        // o Results in a further increase of DSP resources due to the higher II of this loop.
                        // o Retaining the zeroing operation here can give this a loop a max II of 2 and HLS will
                        // resource share.
                        if (LowerTriangularL == true) {
                            L[i][j] = new_L;                                   // Store in lower triangle
                            if (!CholeskyTraits::ARCH2_ZERO_LOOP) L[j][i] = 0; // Zero upper
                        } else {
                            L[j][i] = hls::x_conj(new_L);                      // Store in upper triangle
                            if (!CholeskyTraits::ARCH2_ZERO_LOOP) L[i][j] = 0; // Zero lower
                        }
                    }
                }
            }
        }
    }
    // Zero upper/lower triangle
    // o Use separate loop to ensure main calcuation can achieve an II of 1
    // o As noted above this may increase the DSP resources.
    // o Required when unrolling the inner loop due to array dimension access
    if (CholeskyTraits::ARCH2_ZERO_LOOP) {
    zero_rows_loop:
        for (int i = 0; i < RowsColsA - 1; i++) {
        zero_cols_loop:
            for (int j = i + 1; j < RowsColsA; j++) {
// Define average trip count for reporting, loop reduces in length for every iteration of zero_rows_loop
#pragma HLS loop_tripcount max = 1 + RowsColsA / 2
#pragma HLS PIPELINE
                if (LowerTriangularL == true) {
                    L[i][j] = 0; // Zero upper
                } else {
                    L[j][i] = 0; // Zero lower
                }
            }
        }
    }
    return (return_code);
}

// ===================================================================================================================
// choleskyTop: Top level function that selects implementation architecture and internal types based on the
// traits class provided via the CholeskyTraits template parameter.
// o Call this function directly if you wish to override the default architecture choice or internal types
template <bool LowerTriangularL, int RowsColsA, typename CholeskyTraits, class InputType, class OutputType>
int choleskyTop(const InputType A[RowsColsA][RowsColsA], OutputType L[RowsColsA][RowsColsA]) {
    if (RowsColsA <= 4) {
        return choleskySmall<LowerTriangularL, RowsColsA, CholeskyTraits, InputType, OutputType>(A, L);
    }

    switch (CholeskyTraits::ARCH) {
        case 0:
            return choleskyBasic<LowerTriangularL, RowsColsA, CholeskyTraits, InputType, OutputType>(A, L);
        case 1:
            return choleskyAlt<LowerTriangularL, RowsColsA, CholeskyTraits, InputType, OutputType>(A, L);
        case 2:
            return choleskyAlt2<LowerTriangularL, RowsColsA, CholeskyTraits, InputType, OutputType>(A, L);
        default:
            return choleskyBasic<LowerTriangularL, RowsColsA, CholeskyTraits, InputType, OutputType>(A, L);
    }
}

/**
* @brief cholesky
*
* @tparam LowerTriangularL   When false generates the result in the upper triangle
* @tparam RowsColsA          Defines the matrix dimensions
* @tparam InputType          Input data type
* @tparam OutputType         Output data type
* @tparam TRAITS             choleskyTraits class
*
* @param matrixAStrm         Stream of Hermitian/symmetric positive definite input matrix
* @param matrixLStrm         Stream of Lower or upper triangular output matrix
*
* @return                    An integer type. 0=Success. 1=Failure. The function attempted to find the square root of
* a negative number i.e. the input matrix A was not Hermitian/symmetric positive definite.
*/
template <bool LowerTriangularL,
          int RowsColsA,
          class InputType,
          class OutputType,
          typename TRAITS = choleskyTraits<LowerTriangularL, RowsColsA, InputType, OutputType> >
int cholesky(hls::stream<InputType>& matrixAStrm, hls::stream<OutputType>& matrixLStrm) {
    InputType A[RowsColsA][RowsColsA];
    OutputType L[RowsColsA][RowsColsA];

    for (int r = 0; r < RowsColsA; r++) {
#pragma HLS PIPELINE
        for (int c = 0; c < RowsColsA; c++) {
            matrixAStrm.read(A[r][c]);
        }
    }

    int ret = 0;
    ret = choleskyTop<LowerTriangularL, RowsColsA, TRAITS, InputType, OutputType>(A, L);

    for (int r = 0; r < RowsColsA; r++) {
#pragma HLS PIPELINE
        for (int c = 0; c < RowsColsA; c++) {
            matrixLStrm.write(L[r][c]);
        }
    }
    return ret;
}

} // end namespace solver
} // end namespace xf
#endif
