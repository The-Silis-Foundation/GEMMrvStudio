#include <stdint.h>
#include <neorv32.h>

#define BAUD_RATE 19200

int main(void) {
    // Turn on status LEDs (GPIO)
    neorv32_gpio_port_set(0x0F);

    // Initialize UART0 at 19200 baud, 8N1
    neorv32_uart0_setup(BAUD_RATE, 0);

    // Print welcome banner to console
    neorv32_uart0_printf("\n\n");
    neorv32_uart0_printf("==========================================\n");
    neorv32_uart0_printf("=== NEORV32 RISC-V SoC Hello World     ===\n");
    neorv32_uart0_printf("=== Custom Application Compiled via IDE===\n");
    neorv32_uart0_printf("==========================================\n\n");

    neorv32_uart0_printf("Core Clock Frequency: %u Hz\n", (uint32_t)NEORV32_SYSINFO->CLK);
    neorv32_uart0_printf("Execution successful. Board is running!\n\n");

    int counter = 0;
    while (1) {
        neorv32_uart0_printf("Tick: %d\n", ++counter);
        
        // Simple delay loop
        for (volatile int i = 0; i < 500000; i++);
    }

    return 0;
}