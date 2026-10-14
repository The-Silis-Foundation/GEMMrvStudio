#include <neorv32.h>
#include <stdint.h>
#include <stdbool.h>

// ==============================================================================
// Memory-Mapped I/O Base Addresses
// ==============================================================================
#define VGA_VRAM_BASE       0x90000000UL  // 160x120 Framebuffer (19,200 bytes)
#define VGA_PALETTE_BASE    0x90010000UL  // 256 x 12-bit RGB Color Lookup Table
#define VGA_CTRL_BASE       0x90010400UL  // Status Register ([0]: in_vblank)

#define PS2_DATA_REG        0x90020000UL  // Keyboard Scancode FIFO read
#define PS2_STAT_REG        0x90020004UL  // Keyboard Status ([0]: data_ready, [1]: full)

#define VGA_WIDTH           160
#define VGA_HEIGHT          120
#define VGA_NUM_PIXELS      (VGA_WIDTH * VGA_HEIGHT)

// 12-bit RGB Color values (0xRGB)
#define COLOR_BLACK         0x000
#define COLOR_BLUE          0x00F
#define COLOR_GREEN         0x0F0
#define COLOR_CYAN          0x0FF
#define COLOR_RED           0xF00
#define COLOR_MAGENTA       0xF0F
#define COLOR_YELLOW        0xFF0
#define COLOR_WHITE         0xFFF
#define COLOR_DARK_GRAY     0x444
#define COLOR_LIGHT_GRAY    0xAAA
#define COLOR_ORANGE        0xFA0
#define COLOR_PURPLE        0x808

static inline void vga_set_palette_entry(uint8_t index, uint16_t rgb12) {
    volatile uint32_t *pal = (volatile uint32_t *)(VGA_PALETTE_BASE + ((uint32_t)index << 2));
    *pal = rgb12 & 0x0FFF;
}

static inline void vga_put_pixel(int x, int y, uint8_t color_idx) {
    if ((unsigned)x < VGA_WIDTH && (unsigned)y < VGA_HEIGHT) {
        volatile uint8_t *vram = (volatile uint8_t *)(VGA_VRAM_BASE + (y * VGA_WIDTH + x));
        *vram = color_idx;
    }
}

static void vga_clear(uint8_t color_idx) {
    volatile uint8_t *vram = (volatile uint8_t *)VGA_VRAM_BASE;
    for (int i = 0; i < VGA_NUM_PIXELS; i++) {
        vram[i] = color_idx;
    }
}

static void vga_draw_hline(int x, int y, int length, uint8_t color_idx) {
    if (y < 0 || y >= VGA_HEIGHT) return;
    if (x < 0) { length += x; x = 0; }
    if (x + length > VGA_WIDTH) length = VGA_WIDTH - x;
    if (length <= 0) return;

    volatile uint8_t *line = (volatile uint8_t *)(VGA_VRAM_BASE + (y * VGA_WIDTH + x));
    for (int i = 0; i < length; i++) {
        line[i] = color_idx;
    }
}

static void vga_draw_vline(int x, int y, int length, uint8_t color_idx) {
    if (x < 0 || x >= VGA_WIDTH) return;
    if (y < 0) { length += y; y = 0; }
    if (y + length > VGA_HEIGHT) length = VGA_HEIGHT - y;
    if (length <= 0) return;

    volatile uint8_t *ptr = (volatile uint8_t *)(VGA_VRAM_BASE + (y * VGA_WIDTH + x));
    for (int i = 0; i < length; i++) {
        *ptr = color_idx;
        ptr += VGA_WIDTH;
    }
}

static void setup_palette(void) {
    vga_set_palette_entry(0, COLOR_BLACK);
    vga_set_palette_entry(1, COLOR_BLUE);
    vga_set_palette_entry(2, COLOR_GREEN);
    vga_set_palette_entry(3, COLOR_CYAN);
    vga_set_palette_entry(4, COLOR_RED);
    vga_set_palette_entry(5, COLOR_MAGENTA);
    vga_set_palette_entry(6, COLOR_YELLOW);
    vga_set_palette_entry(7, COLOR_WHITE);
    vga_set_palette_entry(8, COLOR_DARK_GRAY);
    vga_set_palette_entry(9, COLOR_LIGHT_GRAY);
    vga_set_palette_entry(10, COLOR_ORANGE);
    vga_set_palette_entry(11, COLOR_PURPLE);
}

static void render_test_screen(void) {
    vga_clear(0); // Black

    // Draw 8 solid color bars across the screen
    int bar_height = VGA_HEIGHT / 8;
    for (int b = 0; b < 8; b++) {
        for (int y = b * bar_height; y < (b + 1) * bar_height; y++) {
            vga_draw_hline(0, y, VGA_WIDTH, (uint8_t)b);
        }
    }

    // Outer border
    vga_draw_hline(0, 0, VGA_WIDTH, 7);
    vga_draw_hline(0, VGA_HEIGHT - 1, VGA_WIDTH, 7);
    vga_draw_vline(0, 0, VGA_HEIGHT, 7);
    vga_draw_vline(VGA_WIDTH - 1, 0, VGA_HEIGHT, 7);

    // Center crosshairs
    vga_draw_hline(VGA_WIDTH / 2 - 10, VGA_HEIGHT / 2, 20, 7);
    vga_draw_vline(VGA_WIDTH / 2, VGA_HEIGHT / 2 - 10, 20, 7);
}

// Convert Set-2 make scancodes to ASCII
static char ps2_decode(uint8_t sc) {
    switch (sc) {
        case 0x1C: return 'a'; case 0x32: return 'b'; case 0x21: return 'c';
        case 0x23: return 'd'; case 0x24: return 'e'; case 0x2B: return 'f';
        case 0x34: return 'g'; case 0x33: return 'h'; case 0x43: return 'i';
        case 0x3B: return 'j'; case 0x42: return 'k'; case 0x4B: return 'l';
        case 0x3A: return 'm'; case 0x31: return 'n'; case 0x44: return 'o';
        case 0x4D: return 'p'; case 0x15: return 'q'; case 0x2D: return 'r';
        case 0x1B: return 's'; case 0x2C: return 't'; case 0x3C: return 'u';
        case 0x2A: return 'v'; case 0x1D: return 'w'; case 0x22: return 'x';
        case 0x35: return 'y'; case 0x1A: return 'z'; case 0x29: return ' ';
        case 0x5A: return '\n';
        default:   return 0;
    }
}

int main(void) {
    neorv32_uart0_setup(19200, 0);

    neorv32_uart0_printf("\n==============================================\n");
    neorv32_uart0_printf("   GEMMrv32 Standalone Workstation Active     \n");
    neorv32_uart0_printf("   VGA Resolution: 160x120 -> 640x480 @ 60Hz  \n");
    neorv32_uart0_printf("   PS/2 Keyboard Controller Online            \n");
    neorv32_uart0_printf("==============================================\n\n");

    setup_palette();
    render_test_screen();

    neorv32_uart0_printf("[VGA] Color bars rendered successfully!\n");
    neorv32_uart0_printf("[PS/2] Ready for keyboard input...\n");

    int pen_x = (VGA_WIDTH / 2) - 2;   // Center at crosshairs
    int pen_y = (VGA_HEIGHT / 2) - 2;
    uint8_t pen_color = 1;

    while (1) {
        // Poll PS/2 Keyboard FIFO
        if (*((volatile uint32_t *)PS2_STAT_REG) & 1) {
            uint8_t sc = (uint8_t)(*((volatile uint32_t *)PS2_DATA_REG) & 0xFF);
            char ch = ps2_decode(sc);

            neorv32_uart0_printf("[KEY] Scancode: 0x%02x", sc);
            if (ch) {
                neorv32_uart0_printf(" ('%c')", ch);
            }
            neorv32_uart0_printf("\n");

            // Draw interactive 4x4 blocks on the screen
            pen_color = (pen_color % 7) + 1;
            for (int dy = 0; dy < 4; dy++) {
                vga_draw_hline(pen_x, pen_y + dy, 4, pen_color);
            }

            pen_x += 6;
            if (pen_x >= (VGA_WIDTH - 12)) {
                pen_x = 10;
                pen_y += 6;
                if (pen_y >= (VGA_HEIGHT - 12)) {
                    pen_y = 10;
                    render_test_screen();
                }
            }
        }
    }

    return 0;
}
