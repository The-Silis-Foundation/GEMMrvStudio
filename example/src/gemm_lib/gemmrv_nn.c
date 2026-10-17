/**
 * GEMMrv32 Standard Neural Network Acceleration Implementation (gemmrv_nn.c)
 * The Silis Foundation - Open-Source RISC-V SoC with Systolic GEMM Accelerator
 */

#include "gemmrv_nn.h"
#include <neorv32.h>
#include <string.h>

/* Helper for saturation to signed 8-bit range [-128, 127] */
static inline int8_t saturate_s8(int32_t val) {
    if (val > 127) return 127;
    if (val < -128) return -128;
    return (int8_t)val;
}

int gemmrv_nn_init(void) {
    // Check if CFS hardware module is present in the SoC
    if ((NEORV32_SYSINFO->SOC & (1 << SYSINFO_SOC_IO_CFS)) == 0) {
        return 0; // CFS not synthesized
    }
    // Reset CFS accelerator state machine
    NEORV32_CFS->REG[6] = 0;
    asm volatile ("fence" ::: "memory");
    return 1;
}

uint64_t gemmrv_nn_get_cycles(void) {
    return neorv32_cpu_get_systime();
}

void gemmrv_nn_dense_s8(
    const int8_t* in,
    int8_t* out,
    const int8_t* weights,
    const int32_t* bias,
    int M, int K, int N,
    int in_stride, int weight_stride, int out_stride,
    int32_t quant_scale, int quant_shift,
    int act_relu
) {
    // Scratch buffer for 32-bit accumulation of an output row/tile
    // To minimize SRAM stack pressure, we accumulate tile by tile
    int32_t accum_row[64] __attribute__((aligned(4)));

    gemmrv_mat matA = GEMMRV_MAT(in, M, K, in_stride);
    gemmrv_mat matB = GEMMRV_MAT(weights, K, N, weight_stride);

    // If N <= 64 and M == 1 (common for edge classification inference vectors)
    if (M == 1 && N <= 64) {
        gemmrv_mat_out matC = GEMMRV_MAT_OUT(accum_row, 1, N, (N + 7) & ~7);
        gemmrv_mult(&matA, &matB, &matC);

        for (int n = 0; n < N; n++) {
            int32_t val = accum_row[n];
            if (bias) {
                val += bias[n];
            }
            if (act_relu && val < 0) {
                val = 0;
            }
            // Fixed-point scaling
            if (quant_scale != 1) {
                val = (int32_t)(((int64_t)val * quant_scale) >> 16);
            }
            if (quant_shift > 0) {
                val = val >> quant_shift;
            }
            out[n] = saturate_s8(val);
        }
        return;
    }

    // Generic multi-row / batch dense computation
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n += 64) {
            int cur_n = (N - n < 64) ? (N - n) : 64;
            int cur_n_stride = (cur_n + 7) & ~7;

            gemmrv_mat subA = GEMMRV_MAT(&in[m * in_stride], 1, K, in_stride);
            gemmrv_mat subB = GEMMRV_MAT(&weights[n], K, cur_n, weight_stride);
            gemmrv_mat_out subC = GEMMRV_MAT_OUT(accum_row, 1, cur_n, cur_n_stride);

            gemmrv_mult(&subA, &subB, &subC);

            for (int i = 0; i < cur_n; i++) {
                int col = n + i;
                int32_t val = accum_row[i];
                if (bias) {
                    val += bias[col];
                }
                if (act_relu && val < 0) {
                    val = 0;
                }
                if (quant_scale != 1) {
                    val = (int32_t)(((int64_t)val * quant_scale) >> 16);
                }
                if (quant_shift > 0) {
                    val = val >> quant_shift;
                }
                out[m * out_stride + col] = saturate_s8(val);
            }
        }
    }
}

void gemmrv_nn_conv2d_s8(
    const int8_t* in,
    int8_t* out,
    const int8_t* weights,
    const int32_t* bias,
    int in_h, int in_w, int in_c,
    int out_c,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int32_t quant_scale, int quant_shift,
    int act_relu
) {
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
    int k_dim = kernel_h * kernel_w * in_c;
    int k_dim_padded = (k_dim + 7) & ~7;
    int out_c_padded = (out_c + 7) & ~7;

    // Buffer for one im2col patch
    int8_t col_patch[128] __attribute__((aligned(4)));
    int32_t accum_out[64] __attribute__((aligned(4)));

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            // Unroll patch into col_patch buffer
            int patch_idx = 0;
            for (int kh = 0; kh < kernel_h; kh++) {
                int ih = oh * stride_h - pad_h + kh;
                for (int kw = 0; kw < kernel_w; kw++) {
                    int iw = ow * stride_w - pad_w + kw;
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        for (int c = 0; c < in_c; c++) {
                            col_patch[patch_idx++] = in[(ih * in_w + iw) * in_c + c];
                        }
                    } else {
                        // Zero padding
                        for (int c = 0; c < in_c; c++) {
                            col_patch[patch_idx++] = 0;
                        }
                    }
                }
            }
            // Pad patch up to systolic stride multiple of 8
            while (patch_idx < k_dim_padded && patch_idx < 128) {
                col_patch[patch_idx++] = 0;
            }

            // Multiply patch against weights using GEMM coprocessor
            gemmrv_mat matPatch = GEMMRV_MAT(col_patch, 1, k_dim, k_dim_padded);
            gemmrv_mat matWeights = GEMMRV_MAT(weights, k_dim, out_c, out_c_padded);
            gemmrv_mat_out matOut = GEMMRV_MAT_OUT(accum_out, 1, out_c, out_c_padded);

            gemmrv_mult(&matPatch, &matWeights, &matOut);

            // Requantize and write back
            for (int oc = 0; oc < out_c; oc++) {
                int32_t val = accum_out[oc];
                if (bias) val += bias[oc];
                if (act_relu && val < 0) val = 0;
                if (quant_scale != 1) {
                    val = (int32_t)(((int64_t)val * quant_scale) >> 16);
                }
                if (quant_shift > 0) val = val >> quant_shift;

                out[(oh * out_w + ow) * out_c + oc] = saturate_s8(val);
            }
        }
    }
}

void gemmrv_nn_attention_s8(
    const int8_t* Q,
    const int8_t* K,
    const int8_t* V,
    int8_t* out,
    int seq_len, int d_k, int d_v,
    int32_t qk_scale, int qk_shift
) {
    int stride_k = (d_k + 7) & ~7;
    int stride_v = (d_v + 7) & ~7;
    int stride_s = (seq_len + 7) & ~7;

    // Buffer for scores row
    int32_t raw_scores[64] __attribute__((aligned(4)));
    int8_t attn_weights[64] __attribute__((aligned(4)));

    for (int i = 0; i < seq_len; i++) {
        // Step 1: Compute Q_i * K^T (Systolic acceleration)
        gemmrv_mat q_row = GEMMRV_MAT(&Q[i * stride_k], 1, d_k, stride_k);
        gemmrv_mat k_mat = GEMMRV_MAT(K, seq_len, d_k, stride_k); // K transposed in memory layout
        gemmrv_mat_out score_out = GEMMRV_MAT_OUT(raw_scores, 1, seq_len, stride_s);

        gemmrv_mult(&q_row, &k_mat, &score_out);

        // Step 2: Scale and Softmax
        for (int j = 0; j < seq_len; j++) {
            int32_t s = raw_scores[j];
            if (qk_scale != 1) s = (int32_t)(((int64_t)s * qk_scale) >> 16);
            if (qk_shift > 0) s = s >> qk_shift;
            attn_weights[j] = saturate_s8(s);
        }
        gemmrv_nn_softmax_s8(attn_weights, seq_len);

        // Step 3: Attention * V (Systolic acceleration)
        int32_t out_accum[64] __attribute__((aligned(4)));
        gemmrv_mat a_row = GEMMRV_MAT(attn_weights, 1, seq_len, stride_s);
        gemmrv_mat v_mat = GEMMRV_MAT(V, seq_len, d_v, stride_v);
        gemmrv_mat_out out_mat = GEMMRV_MAT_OUT(out_accum, 1, d_v, stride_v);

        gemmrv_mult(&a_row, &v_mat, &out_mat);

        for (int k = 0; k < d_v; k++) {
            out[i * stride_v + k] = saturate_s8(out_accum[k] >> 7);
        }
    }
}

void gemmrv_nn_relu_s8(int8_t* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] < 0) data[i] = 0;
    }
}

void gemmrv_nn_maxpool2d_s8(
    const int8_t* in,
    int8_t* out,
    int in_h, int in_w, int channels,
    int pool_h, int pool_w,
    int stride_h, int stride_w
) {
    int out_h = (in_h - pool_h) / stride_h + 1;
    int out_w = (in_w - pool_w) / stride_w + 1;

    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int8_t max_val = -128;
                for (int ph = 0; ph < pool_h; ph++) {
                    int ih = oh * stride_h + ph;
                    for (int pw = 0; pw < pool_w; pw++) {
                        int iw = ow * stride_w + pw;
                        int8_t val = in[(ih * in_w + iw) * channels + c];
                        if (val > max_val) max_val = val;
                    }
                }
                out[(oh * out_w + ow) * channels + c] = max_val;
            }
        }
    }
}

int gemmrv_nn_argmax_s8(const int8_t* vec, int size) {
    if (size <= 0) return 0;
    int best_idx = 0;
    int8_t max_val = vec[0];
    for (int i = 1; i < size; i++) {
        if (vec[i] > max_val) {
            max_val = vec[i];
            best_idx = i;
        }
    }
    return best_idx;
}

void gemmrv_nn_softmax_s8(int8_t* vec, int size) {
    // Fast integer-approx softmax: subtract max, exp lookup/shift, normalize
    if (size <= 0) return;
    int8_t max_val = vec[0];
    for (int i = 1; i < size; i++) {
        if (vec[i] > max_val) max_val = vec[i];
    }

    int32_t sum = 0;
    int32_t exp_vals[64];
    int count = (size > 64) ? 64 : size;

    for (int i = 0; i < count; i++) {
        int diff = (int)vec[i] - (int)max_val; // <= 0
        // Approximation: 2^(diff / 16)
        int shift = (-diff) >> 3;
        int32_t val = (shift < 16) ? (256 >> shift) : 0;
        exp_vals[i] = val;
        sum += val;
    }

    if (sum == 0) sum = 1;
    for (int i = 0; i < count; i++) {
        vec[i] = (int8_t)((exp_vals[i] * 127) / sum);
    }
}
