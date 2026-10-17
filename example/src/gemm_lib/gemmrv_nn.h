/**
 * GEMMrv32 Standard Neural Network Acceleration Library (gemmrv_nn.h)
 * The Silis Foundation - Open-Source RISC-V SoC with Systolic GEMM Accelerator
 * 
 * Standardized CMSIS-NN / TFLite Micro compatible API for GEMMrv32.
 * Hardware-accelerated with 8x8 int8 systolic array CFS coprocessor.
 */

#ifndef GEMMRV_NN_H
#define GEMMRV_NN_H

#include <stdint.h>
#include "gemmrv_hw.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------- */
/* Hardware Coprocessor Initialization & Diagnostics                          */
/* -------------------------------------------------------------------------- */

/**
 * Initializes and resets the GEMMrv32 systolic coprocessor.
 * Returns 1 if coprocessor CFS is detected and responds, 0 otherwise.
 */
int gemmrv_nn_init(void);

/**
 * Reads the RISC-V 64-bit hardware cycle counter (mcycle/mcycleh).
 */
uint64_t gemmrv_nn_get_cycles(void);

/* -------------------------------------------------------------------------- */
/* Neural Network Core Layers (Hardware Accelerated via 8x8 Systolic Array)   */
/* -------------------------------------------------------------------------- */

/**
 * Fully Connected / Dense Layer (int8 in -> int8 out with int32 bias)
 * Computes: out = Saturate_s8((in * weights + bias) * quant_scale >> quant_shift)
 * 
 * @param in           Pointer to input activations [M x K], aligned(4)
 * @param out          Pointer to output activations [M x N], aligned(4)
 * @param weights      Pointer to weights [K x N] with stride multiple of 8, aligned(4)
 * @param bias         Pointer to int32 bias vector [N] (optional, can be NULL)
 * @param M            Batch size / sequence length
 * @param K            Input features count
 * @param N            Output features count
 * @param in_stride    Input row stride in memory (multiple of 8)
 * @param weight_stride Weight matrix stride in memory (multiple of 8)
 * @param out_stride   Output row stride in memory (multiple of 8)
 * @param quant_scale  Fixed-point requantization multiplier
 * @param quant_shift  Fixed-point right bitshift
 * @param act_relu     If non-zero, applies ReLU activation before output saturation
 */
void gemmrv_nn_dense_s8(
    const int8_t* in,
    int8_t* out,
    const int8_t* weights,
    const int32_t* bias,
    int M, int K, int N,
    int in_stride, int weight_stride, int out_stride,
    int32_t quant_scale, int quant_shift,
    int act_relu
);

/**
 * 2D Convolution Layer (int8 in -> int8 out)
 * Accelerates sliding 2D convolution by mapping flattened patches to systolic GEMM (im2col).
 * 
 * @param in           Input image / activation tensor [H x W x C_in], aligned(4)
 * @param out          Output feature map [H_out x W_out x C_out], aligned(4)
 * @param weights      Filter kernels [(kernel_h * kernel_w * in_c) x out_c], aligned(4)
 * @param bias         Per-channel int32 bias [C_out] (optional, can be NULL)
 * @param in_h         Input height
 * @param in_w         Input width
 * @param in_c         Input channels
 * @param out_c        Output channels
 * @param kernel_h     Kernel height (e.g., 3, 5)
 * @param kernel_w     Kernel width (e.g., 3, 5)
 * @param stride_h     Vertical stride
 * @param stride_w     Horizontal stride
 * @param pad_h        Vertical padding
 * @param pad_w        Horizontal padding
 * @param quant_scale  Fixed-point multiplier
 * @param quant_shift  Fixed-point bitshift
 * @param act_relu     Apply ReLU
 */
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
);

/**
 * Multi-Head / Self-Attention Layer for Micro-Transformers (int8)
 * Computes: Attention(Q, K, V) = Softmax((Q * K^T) / sqrt(d_k)) * V
 * Both Q*K^T and Score*V are mapped directly onto the systolic coprocessor.
 * 
 * @param Q            Query tensor [seq_len x d_k], aligned(4)
 * @param K            Key tensor [seq_len x d_k], aligned(4)
 * @param V            Value tensor [seq_len x d_v], aligned(4)
 * @param out          Attention output [seq_len x d_v], aligned(4)
 * @param seq_len      Sequence length (e.g. 16, 32, 64)
 * @param d_k          Query/Key hidden dimension (multiple of 8)
 * @param d_v          Value hidden dimension (multiple of 8)
 * @param qk_scale     Scale factor for Q*K product
 * @param qk_shift     Bitshift for Q*K product
 */
void gemmrv_nn_attention_s8(
    const int8_t* Q,
    const int8_t* K,
    const int8_t* V,
    int8_t* out,
    int seq_len, int d_k, int d_v,
    int32_t qk_scale, int qk_shift
);

/* -------------------------------------------------------------------------- */
/* Standard Vector Activations & Pooling Operations                           */
/* -------------------------------------------------------------------------- */

/**
 * In-place ReLU activation on int8 vector.
 */
void gemmrv_nn_relu_s8(int8_t* data, int size);

/**
 * 2D Max Pooling on HWC feature maps.
 */
void gemmrv_nn_maxpool2d_s8(
    const int8_t* in,
    int8_t* out,
    int in_h, int in_w, int channels,
    int pool_h, int pool_w,
    int stride_h, int stride_w
);

/**
 * Returns the index of the highest value in an int8 vector (Classification decision).
 */
int gemmrv_nn_argmax_s8(const int8_t* vec, int size);

/**
 * Approximated fixed-point Softmax on an int8 vector in-place.
 */
void gemmrv_nn_softmax_s8(int8_t* vec, int size);

#ifdef __cplusplus
}
#endif

#endif // GEMMRV_NN_H
