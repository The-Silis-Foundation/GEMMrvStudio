/**
 * GEMMrv32 Neural Network Inference Demo
 * The Silis Foundation - Open-Source RISC-V SoC with Systolic GEMM Accelerator
 * 
 * Verifies:
 *   - Auto-generated Makefile linkage with gemmrv_hw.c and gemmrv_nn.c
 *   - Autonomous int8 quantized weights (model_weights.h)
 *   - CMSIS-NN compatible gemmrv_nn_dense_s8() systolic accelerator execution
 *   - Argmax classification decision
 */

#include <neorv32.h>
#include "gemmrv_hw.h"
#include "gemmrv_nn.h"
#include "model_weights.h"

#define INPUT_DIM   32
#define HIDDEN_DIM  16
#define OUTPUT_DIM  8

#define INPUT_STRIDE   32
#define HIDDEN_STRIDE  16
#define OUTPUT_STRIDE  8

// Static tensor buffers aligned to 4-byte boundaries in on-chip DMEM
static int8_t input_activations[INPUT_STRIDE] __attribute__((aligned(4)));
static int8_t hidden_activations[HIDDEN_STRIDE] __attribute__((aligned(4)));
static int8_t output_logits[OUTPUT_STRIDE] __attribute__((aligned(4)));

int main(void) {
    // 1. Initialize SoC Runtime Environment & UART
    neorv32_gpio_port_set(0x01);
    neorv32_uart0_setup(19200, 0);
    neorv32_gpio_port_set(0x03);

    // 2. Initialize Systolic Array CFS Coprocessor
    if (!gemmrv_nn_init()) {
        neorv32_uart0_printf("[ERROR] Systolic CFS Coprocessor not detected!\n");
        neorv32_gpio_port_set(0xFF);
        return 1;
    }
    neorv32_gpio_port_set(0x07);

    uint32_t clk = neorv32_sysinfo_get_clk();
    uint32_t frame_count = 0;

    while (1) {
        neorv32_uart0_printf("\n========================================\n");
        neorv32_uart0_printf("   GEMMrv32 SoC Neural Inference #%u    \n", ++frame_count);
        neorv32_uart0_printf("========================================\n");

        // 3. Prepare synthetic test input vector (32 features)
        for (int i = 0; i < INPUT_DIM; i++) {
            input_activations[i] = (int8_t)(((i + frame_count) * 7) % 63 - 31);
        }

        neorv32_gpio_port_set(0x0F);
        uint32_t start_cycles = neorv32_cpu_csr_read(CSR_MCYCLE);

        // 4. LAYER 1: Dense (32 -> 16) with Fused ReLU
        gemmrv_nn_dense_s8(
            input_activations,
            hidden_activations,
            fc1_weight_data,
            fc1_bias_data,
            /* M */ 1, /* K */ INPUT_DIM, /* N */ HIDDEN_DIM,
            INPUT_STRIDE, HIDDEN_STRIDE, HIDDEN_STRIDE,
            /* scale */ FC1_WEIGHT_SCALE,
            /* shift */ FC1_WEIGHT_SHIFT,
            /* relu */ 1
        );

        // 5. LAYER 2: Dense (16 -> 8) Linear Logits
        gemmrv_nn_dense_s8(
            hidden_activations,
            output_logits,
            fc2_weight_data,
            fc2_bias_data,
            /* M */ 1, /* K */ HIDDEN_DIM, /* N */ OUTPUT_DIM,
            HIDDEN_STRIDE, OUTPUT_STRIDE, OUTPUT_STRIDE,
            /* scale */ FC2_WEIGHT_SCALE,
            /* shift */ FC2_WEIGHT_SHIFT,
            /* relu */ 0
        );

        uint32_t end_cycles = neorv32_cpu_csr_read(CSR_MCYCLE);
        uint32_t latency = end_cycles - start_cycles;

        // 6. Classification Output Decision via argmax
        int predicted_class = gemmrv_nn_argmax_s8(output_logits, OUTPUT_DIM);

        neorv32_uart0_printf("Input Feature[0]: %d\n", input_activations[0]);
        neorv32_uart0_printf("Output Logits:    [");
        for (int i = 0; i < OUTPUT_DIM; i++) {
            neorv32_uart0_printf("%d%s", output_logits[i], (i == OUTPUT_DIM - 1) ? "" : ", ");
        }
        neorv32_uart0_printf("]\n");

        neorv32_uart0_printf("Predicted Class:  %d\n", predicted_class);
        neorv32_uart0_printf("Systolic Latency: %u CPU clock cycles\n", latency);
        neorv32_uart0_printf("STATUS: SUCCESS\n");

        neorv32_gpio_port_set(0xAA);
        neorv32_aux_delay_ms(clk, 1000);
    }

    return 0;
}
