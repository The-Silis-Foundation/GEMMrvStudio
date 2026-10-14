# GEMMrvStudio

**GEMMrvStudio** is an integrated development environment (IDE), build toolchain interface, and hardware console for the **GEMMrv32** NEORV32 RISC-V SoC with custom 8x8 systolic-array hardware acceleration.

---

## Repository Structure

```text
GEMMrvStudio/
├── gemm_dashboard.py        # Main GUI IDE & Hardware Console application
├── gemm_engine.py           # Core engine & headless CLI interface
├── example/
│   ├── src/                 # Example C/C++ firmware projects
│   │   ├── demo_blink_led/
│   │   ├── demo_cnn_classifier/
│   │   ├── demo_gemm/
│   │   ├── demo_lenet/
│   │   ├── demo_micro_transformer/
│   │   ├── demo_predictive_lm/
│   │   ├── demo_vga_doom/   # Standalone VGA & PS/2 keyboard graphics console
│   │   ├── gemm_lib/        # Hardware driver API for systolic coprocessor
│   │   ├── hello_c/
│   │   ├── hello_cpp/
│   │   └── hello_world/
│   └── bin/                 # Pre-compiled NEORV32 raw executable binaries (.bin)
│       ├── demo_blink_led.bin
│       ├── demo_cnn_classifier.bin
│       ├── demo_gemm.bin
│       ├── demo_lenet_car_color.bin
│       ├── demo_micro_transformer.bin
│       ├── demo_pretrained_llama2.bin
│       ├── demo_vga_doom.bin
│       ├── hello.bin
│       ├── hello_cpp.bin
│       └── hello_world.bin
└── README.md
```

---

## Features

1. **RISC-V Source Editor & Compiler Wizard**:
   - Tabbed C/C++ code editor with line numbers and syntax highlighting.
   - Built-in linker wizard for NEORV32 GCC toolchains.
   - One-click **Build Binary** and **Build & Run on Board**.
   - Real-time compiler diagnostics window.

2. **Hardware Console & Programmer**:
   - Integrated UART terminal (auto-reconnect, 19200 baud default).
   - Bootloader programmer for rapid binary download to FPGA RAM over serial.
   - Collapsible image inference testing tool for digit and vehicle neural network models.

3. **Headless CLI Operation**:
   All features can be scripted or executed in headless/CI mode:
   ```bash
   # List available serial ports
   python gemm_dashboard.py --list-ports

   # Headless compilation
   python gemm_dashboard.py --build example/src/demo_gemm

   # Headless binary flashing
   python gemm_dashboard.py --flash example/bin/demo_gemm.bin --port COM10

   # Headless image inference
   python gemm_dashboard.py --infer path/to/digit.png --dataset digits --port COM10
   ```

---

## Requirements

- Python 3.8+
- Dependencies: `pyserial`, `Pillow`, `numpy`
  ```bash
  pip install pyserial Pillow numpy
  ```
- RISC-V GCC cross-compiler (`riscv-none-elf-gcc` or `riscv32-unknown-elf-gcc`) and GNU `make`.

---

## Getting Started

Launch the IDE:
```bash
python gemm_dashboard.py
```
