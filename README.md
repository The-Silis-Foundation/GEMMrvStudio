# GEMMrvStudio

**GEMMrvStudio** is an industrial development environment (IDE), build toolchain interface, and full-screen telemetry console for the **GEMMrv32** RISC-V SoC with an on-chip custom $8 \times 8$ systolic-array GEMM hardware accelerator, developed by **The Silis Foundation**.

GEMMrvStudio delivers a clean, focused, Arduino IDE 2.x-style workflow: write C/C++ firmware, compile with the bundled RISC-V cross-compiler with automatic Makefile and hardware driver management, upload binaries over UART to the NEORV32 serial bootloader in one click, and stream live interactive execution over a full-screen TUI terminal.

---

## Repository Structure

```text
GEMMrvStudio/
├── gemm_dashboard.py        # Main GUI IDE (PyQt5 + QScintilla, Arduino 2.x Industrial Style)
├── gemm_engine.py           # Core toolchain engine, auto-discovery & headless CLI
├── gemm_tui.py              # Full-screen universal serial console & telemetry monitor
├── resources/               # Crisp vector SVG icons (Verify, Upload, Folder, GEMM, Terminal, Theme)
├── example/
│   ├── src/                 # Example C/C++ firmware projects
│   │   ├── demo_blink_led/
│   │   ├── demo_cnn_classifier/
│   │   ├── demo_gemm/       # Hardware systolic matrix multiplication
│   │   ├── demo_lenet/
│   │   ├── demo_micro_transformer/
│   │   ├── demo_predictive_lm/
│   │   ├── demo_vga_doom/   # Standalone VGA framebuffer & graphics test
│   │   ├── gemm_lib/        # Official GEMMrv32 hardware driver library (gemmrv_hw.c/h)
│   │   ├── hello_c/
│   │   ├── hello_cpp/
│   │   └── hello_world/
│   └── bin/                 # Pre-compiled NEORV32 raw executable binaries (.bin)
│       ├── demo_blink_led.bin
│       ├── demo_cnn_classifier.bin
│       ├── demo_gemm.bin
│       ├── demo_vga_doom.bin
│       └── ...
└── README.md
```

---

## Key Features

### 1. Arduino IDE 2.x Industrial User Interface
- **Pure Vector SVGs**: Clean, resolution-independent vector icons for Verify, Upload, Sketchbook, GEMM Workbench, Terminal, and Theme toggles.
- **Dual Themes**:
  - **Industrial Light** (default): Crisp, high-contrast engineering light mode.
  - **Industrial Dark**: Arduino Dark (`#1e1e1e` / `#252526` with `#00979c` teal accents).
- **QScintilla Code Editor**: Industrial-grade editing engine with synchronized line numbering, bracket matching, auto-indentation, and C/C++ syntax highlighting tailored for RISC-V and GEMMrv32 primitives (`gemmrv_mat`, `gemmrv_mat_out`, `gemmrv_mult`).

### 2. Automatic Makefile & Dependency Management
- **Zero-Config Makefiles**: When opening any C/C++ folder, GEMMrvStudio automatically detects all source files, checks for `#include "gemmrv_hw.h"`, links the bundled `gemm_lib` driver, configures Basys 3 RAM/ROM bounds, and connects to the NEORV32 Board Support Package without requiring manual configuration.
- **1-Click Verify (`✓`)**: Builds the firmware using `make exe` via the bundled cross-compiler.
- **1-Click Upload (`➜`)**: Compiles, uploads via paced binary streaming to the bootloader, verifies checksums, boots the application (`e`), and opens the serial monitor.

### 3. Practical GEMM Coprocessor Workbench
- **Memory Stride & Zero-Padding Scaffolder**: Calculates the hardware-required 8-byte stride alignment and 4-byte pointer alignment for arbitrary matrix dimensions ($M \times K \times N$) and generates drop-in C initialization arrays and structs.
- **SRAM Tiling & Budget Planner**: Decomposes large matrices into $8 \times 8 \times 8$ systolic tiles and checks memory footprint against the **64 KB DMEM limit** of the Basys 3 FPGA.

### 4. Full-Screen TUI Serial Console
- Dedicated curses terminal with live streaming serial I/O.
- **Universal Character Streaming**: Every key typed streams directly to the UART without project-specific toggles—works seamlessly with DOOM controls (WASD, Space), interactive shells, or diagnostic menus.
- Real-time traffic metrics (RX/TX bytes, port info) and smooth scrollback buffer (`PageUp` / `PageDown`).

---

## Installation & Requirements

### System Requirements (Linux / Ubuntu)
```bash
# Python dependencies
sudo apt update
sudo apt install -y python3-pyqt5 python3-pyqt5.qsci python3-pyqt5.qtsvg python3-serial python3-pil python3-numpy

# RISC-V Cross-Compiler Toolchain (if not using bundled toolchain)
# e.g., riscv-none-elf-gcc or riscv64-unknown-elf-gcc + GNU make
```

---

## Usage Guide

### 1. Launching the Graphical IDE
```bash
python3 gemm_dashboard.py
```
* **Verify (`✓`)**: Compiles current sketch.
* **Upload (`➜`)**: Compiles and flashes to Basys 3 board over serial.
* **Theme Toggle**: Switch between Light and Dark mode on the fly.
* **Sidebar**: Switch between File Explorer and the GEMM Coprocessor Workbench.

### 2. Launching the Full-Screen TUI Serial Console
To monitor board output or interact with running firmware (e.g. DOOM, shell):
```bash
# Open interactive TUI console
python3 gemm_dashboard.py --open-console --port /dev/ttyUSB1 -b 19200

# Short flag / alias
python3 gemm_dashboard.py console -p /dev/ttyUSB1 --baude 19200
```
* **Type directly**: Keystrokes stream immediately to the FPGA.
* **`F3`**: Clear the screen.
* **`F5`**: Reconnect serial port.
* **`PgUp` / `PgDn`**: Scroll through historical logs.
* **`Ctrl+C`**: Cleanly exit the console.

### 3. CLI Command Reference Table

| Task | Command Syntax | Description |
| :--- | :--- | :--- |
| **Launch IDE GUI** | `python3 gemm_dashboard.py` | Starts PyQt5 / QScintilla IDE (Arduino 2.x theme) |
| **Open TUI Console** | `python3 gemm_dashboard.py --open-console` | Opens interactive full-screen streaming terminal |
| **Flash Firmware** | `python3 gemm_dashboard.py flash <firmware.bin>` | Streams binary to NEORV32 bootloader and executes |
| **Flash & Monitor** | `python3 gemm_dashboard.py flash <firmware.bin> --console` | Flashes firmware and immediately opens TUI console |
| **Compile Project** | `python3 gemm_dashboard.py build -p <folder> [-O2\|-Os]` | Compiles project with bundled RISC-V GCC |
| **List Serial Ports** | `python3 gemm_dashboard.py list-ports` | Lists physical UART ports (auto-filters dummy `ttyS` ports) |
| **Program FPGA (JTAG)** | `python3 gemm_dashboard.py flash-bitstream` | Loads bitstream into Basys 3 SRAM via `openFPGALoader` |
| **Program FPGA (SPI)** | `python3 gemm_dashboard.py flash-bitstream --spi-flash` | Writes bitstream to non-volatile SPI flash for persistent boot |

---

### 4. CLI Arguments & Flags Breakdown

#### `flash` Subcommand
```bash
python3 gemm_dashboard.py flash [binary] [--bin BIN] [--port PORT] [--baud BAUD] [--console] [--json]
```
| Flag / Argument | Aliases | Default | Description |
| :--- | :--- | :--- | :--- |
| `binary` | *(positional)* | `None` | Path to executable `.bin` file (e.g. `../DOOM-on-GEMMrv32/bin/demo_vga_doom.bin`) |
| `--bin` | `-b` | `None` | Path to executable `.bin` file (named flag) |
| `--port` | `-p` | `/dev/ttyUSB1` | Target UART device |
| `--baud` | `-baude`, `--baude` | `19200` | Baud rate |
| `--console` | `--open-console`, `-c`| `False` | Automatically launches interactive TUI monitor upon boot |
| `--json` | | `False` | Output structured JSON results for scripts / CI |

#### `--open-console` (or `console`) Subcommand
```bash
python3 gemm_dashboard.py --open-console [--port PORT] [-b BAUD]
```
| Flag / Argument | Aliases | Default | Description |
| :--- | :--- | :--- | :--- |
| `--port` | `-p` | `/dev/ttyUSB1` | Serial port to monitor |
| `-b` | `--baud`, `--baude` | `19200` | Baud rate |

#### TUI Console Keyboard Shortcuts
| Key | Action |
| :--- | :--- |
| **Any Typed Key** | Streams directly to SoC UART with zero latency (e.g. WASD, Space, Enter) |
| **`F3`** | Clears the terminal log screen |
| **`F5`** | Reconnects the serial port |
| **`PgUp` / `PgDn`** | Scrolls through terminal history buffer |
| **`Ctrl+C`** | Cleanly exits the console and restores terminal state |

#### `build` Subcommand
```bash
python3 gemm_dashboard.py build --project PROJECT [--opt OPT] [--output OUT] [--json]
```
| Flag / Argument | Aliases | Default | Description |
| :--- | :--- | :--- | :--- |
| `--project` | `-p` | *(Required)* | Project directory (e.g. `example/src/demo_gemm` or `../DOOM-on-GEMMrv32/src`) |
| `--opt` | `-O` | `-O2` | Optimization level (`-O2`, `-Os`, `-O0`, `-O1`, `-O3`) |
| `--output` | `-o` | `example/bin/<name>.bin` | Destination path for packed NEORV32 binary |
| `--json` | | `False` | Output memory footprint metrics (IMEM, DMEM) as JSON |

---

## License

Developed by **The Silis Foundation**. Open-source under the BSD 3-Clause License.
