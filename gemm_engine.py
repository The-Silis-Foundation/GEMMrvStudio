"""
GEMMrv32 Core Engine & Headless CLI
The Silis Foundation - Open-Source RISC-V SoC & Systolic GEMM Hardware Accelerator

Provides programmatic & command-line APIs for:
  - Cross-platform toolchain discovery (Linux/macOS/Windows)
  - Compiling RISC-V projects to NEORV32-compatible raw binaries
  - Bootloader handshake and firmware flashing over serial
  - UART terminal I/O & command sending
  - On-chip memory & SRAM budget calculation
  - Neural network image inference (28x28 digits / 32x32 vehicles)
  - Systolic Array CFS register parsing & telemetry
"""

import sys
import os
import time
import argparse
import json
import struct
import subprocess
import glob
import shutil
import serial
import serial.tools.list_ports
import numpy as np
from PIL import Image

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# If running inside a PyInstaller frozen bundle or AppImage mount (/tmp/.mount_*),
# copy bundled examples and bins to a persistent, writable user workspace (~/.gemmrvstudio).
if getattr(sys, "frozen", False):
    USER_WORKSPACE = os.path.expanduser("~/.gemmrvstudio")
    DEFAULT_BINS_DIR = os.path.join(USER_WORKSPACE, "example", "bin")
    DEFAULT_PROJECTS_DIR = os.path.join(USER_WORKSPACE, "example", "src")
    DEFAULT_GEMM_LIB_DIR = os.path.join(USER_WORKSPACE, "example", "src", "gemm_lib")

    # Seed workspace if not already present or out of date
    if not os.path.isdir(DEFAULT_PROJECTS_DIR):
        os.makedirs(DEFAULT_BINS_DIR, exist_ok=True)
        os.makedirs(DEFAULT_PROJECTS_DIR, exist_ok=True)
        bundled_example = os.path.join(REPO_DIR, "example")
        if os.path.isdir(bundled_example):
            shutil.copytree(bundled_example, os.path.join(USER_WORKSPACE, "example"), dirs_exist_ok=True)
else:
    DEFAULT_BINS_DIR = os.path.join(REPO_DIR, "example", "bin")
    DEFAULT_PROJECTS_DIR = os.path.join(REPO_DIR, "example", "src")
    DEFAULT_GEMM_LIB_DIR = os.path.join(REPO_DIR, "example", "src", "gemm_lib")


class GEMMEngine:
    def __init__(self, port=None, baud=19200):
        self.port = port or self.get_default_port()
        self.baud = baud
        self.ser = None

    # -------------------------------------------------------------
    # Cross-Platform Toolchain & Path Discovery
    # -------------------------------------------------------------
    @classmethod
    def find_neorv32_home(cls):
        """Locates NEORV32 / GEMMrv32 repository root."""
        env_home = os.environ.get("NEORV32_HOME")
        if env_home and os.path.isfile(os.path.join(env_home, "sw", "common", "common.mk")):
            return os.path.abspath(env_home)

        # Common sibling directories
        candidates = [
            os.path.join(REPO_DIR, "..", "GEMMrv32"),
            os.path.join(REPO_DIR, "..", "DOOM-on-GEMMrv32", "neorv32"),
            os.path.join(REPO_DIR, "..", "neorv32"),
            "/home/econsystems/random/GEMMrv32",
        ]
        for c in candidates:
            if os.path.isfile(os.path.join(c, "sw", "common", "common.mk")):
                return os.path.abspath(c)
        return os.path.abspath(os.path.join(REPO_DIR, "..", "GEMMrv32"))

    @classmethod
    def find_riscv_prefix(cls):
        """Auto-detects RISC-V GCC cross-compiler prefix."""
        env_prefix = os.environ.get("RISCV_PREFIX")
        if env_prefix:
            return env_prefix

        prefixes = ["riscv-none-elf-", "riscv64-unknown-elf-", "riscv32-unknown-elf-"]
        search_dirs = [
            os.path.join(REPO_DIR, "toolchain", "riscv", "bin"),
            os.path.join(REPO_DIR, "toolchain", "bin"),
            os.path.expanduser("~/.local/bin"),
            "/usr/bin",
            "/usr/local/bin",
            "/opt/riscv/bin",
        ]
        # Check bundled and explicit search dirs first
        for sdir in search_dirs:
            if os.path.isdir(sdir):
                for p in prefixes:
                    if os.path.isfile(os.path.join(sdir, f"{p}gcc")):
                        return p

        # Check PATH
        for p in prefixes:
            if shutil.which(f"{p}gcc"):
                return p

        return "riscv-none-elf-"

    @classmethod
    def find_make(cls):
        """Finds GNU make executable."""
        make_bin = shutil.which("make")
        if make_bin:
            return make_bin
        if sys.platform == "win32":
            candidates = [
                r"D:\vivado\2025.2\Vivado\gnuwin\bin\make.exe",
                r"C:\Program Files\Git\usr\bin\make.exe"
            ]
            for c in candidates:
                if os.path.isfile(c):
                    return c
        return "make"

    @classmethod
    def get_default_port(cls):
        """Returns the most appropriate serial port."""
        ports = cls.list_ports(filter_virtual=True)
        if not ports:
            return "/dev/ttyUSB1" if sys.platform.startswith("linux") else "COM10"

        # Check for FTDI UART (/dev/ttyUSB1 or interface 1)
        for p in ports:
            dev = p["device"]
            if "ttyUSB1" in dev or ("0403:6010" in p.get("hwid", "") and "LOCATION=" in p.get("hwid", "") and ".1" in p.get("hwid", "")):
                return dev
        for p in ports:
            if "ttyUSB" in p["device"]:
                return p["device"]
        return ports[0]["device"]

    # -------------------------------------------------------------
    # Toolchain: Project Build & Packaging
    # -------------------------------------------------------------
    @classmethod
    def ensure_makefile(cls, proj_dir, target_name):
        c_files = glob.glob(os.path.join(proj_dir, "*.c"))
        c_basenames = [os.path.basename(f) for f in c_files]
        if "main.c" in c_basenames:
            c_basenames.remove("main.c")
            c_sources = "main.c " + " ".join(c_basenames)
        elif f"{target_name}.c" in c_basenames:
            c_basenames.remove(f"{target_name}.c")
            c_sources = f"{target_name}.c " + " ".join(c_basenames)
        else:
            c_sources = " ".join(c_basenames) if c_basenames else "main.c"

        # Check if project uses GEMM hardware library
        uses_gemm = False
        for cf in c_files:
            try:
                with open(cf, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if "gemmrv" in content or "gemm_lib" in content:
                        uses_gemm = True
                        break
            except Exception:
                pass

        if uses_gemm:
            rel_gemm = os.path.relpath(DEFAULT_GEMM_LIB_DIR, proj_dir)
            c_sources += f" {rel_gemm}/gemmrv_hw.c {rel_gemm}/gemmrv_nn.c"

        neorv32_home = cls.find_neorv32_home()
        riscv_prefix = cls.find_riscv_prefix()

        makefile_path = os.path.join(proj_dir, "Makefile")
        content = f"""# Auto-generated NEORV32 GEMMrv32 Makefile
MARCH ?= rv32im_zicsr_zifencei
RISCV_PREFIX ?= {riscv_prefix}
EFFORT ?= -O2

APP_SRC ?= {c_sources}
APP_INC += -I . -I {DEFAULT_GEMM_LIB_DIR}

# GEMMrv32 On-Chip SRAM Configuration (128 KB IMEM, 64 KB DMEM)
USER_FLAGS += -Wl,--defsym,__neorv32_rom_size=128k
USER_FLAGS += -Wl,--defsym,__neorv32_ram_size=64k

NEORV32_HOME ?= {neorv32_home}
include $(NEORV32_HOME)/sw/common/common.mk
"""
        with open(makefile_path, "w", encoding="utf-8") as f:
            f.write(content)
        return makefile_path

    @classmethod
    def build_project(cls, proj_dir, opt_level="-O2", out_bin_path=None, log_cb=None):
        """
        Builds a RISC-V project in proj_dir using detected GNU make & RISC-V GCC.
        Returns (success: bool, info: dict)
        """
        if not os.path.isdir(proj_dir):
            return False, {"error": f"Project directory does not exist: {proj_dir}"}

        target_name = os.path.basename(os.path.normpath(proj_dir))
        if not out_bin_path:
            os.makedirs(DEFAULT_BINS_DIR, exist_ok=True)
            out_bin_path = os.path.join(DEFAULT_BINS_DIR, f"{target_name}.bin")

        def emit(msg):
            if log_cb:
                log_cb(msg)

        # Check makefile
        makefile_path = os.path.join(proj_dir, "Makefile")
        if not os.path.exists(makefile_path):
            makefile_path = os.path.join(proj_dir, "makefile")
        if not os.path.exists(makefile_path):
            emit(f"[INFO] Auto-generating Makefile for {target_name}...\n")
            cls.ensure_makefile(proj_dir, target_name)

        neorv32_home = cls.find_neorv32_home()
        riscv_prefix = cls.find_riscv_prefix()
        make_bin = cls.find_make()

        env = os.environ.copy()
        # Prepend bundled toolchain and local bin to PATH
        bundled_bin = os.path.join(REPO_DIR, "toolchain", "riscv", "bin")
        local_bin = os.path.expanduser("~/.local/bin")
        extra_paths = [p for p in [bundled_bin, local_bin] if os.path.isdir(p)]
        if extra_paths:
            env["PATH"] = os.pathsep.join(extra_paths) + os.pathsep + env.get("PATH", "")

        cmd = [
            make_bin,
            f"NEORV32_HOME={neorv32_home}",
            f"RISCV_PREFIX={riscv_prefix}",
            f"EFFORT={opt_level}",
            "clean_all",
            "exe"
        ]
        emit(f"[BUILD] Command: {' '.join(cmd)}\n")

        proc = subprocess.Popen(cmd, cwd=proj_dir, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        output_lines = []
        for line in proc.stdout:
            output_lines.append(line)
            emit(line)
        proc.wait()

        if proc.returncode != 0:
            return False, {
                "error": f"Make failed with exit code {proc.returncode}.",
                "exit_code": proc.returncode,
                "log": "".join(output_lines)
            }

        # Check generated executable binary
        candidates = [
            os.path.join(proj_dir, "neorv32_exe.bin"),
            os.path.join(proj_dir, "build", "neorv32_exe.bin"),
            os.path.join(proj_dir, f"{target_name}.bin"),
        ]
        built_bin = None
        for c in candidates:
            if os.path.exists(c):
                built_bin = c
                break

        if not built_bin:
            # Fallback to objcopy if standard image_gen wasn't invoked
            main_elf = os.path.join(proj_dir, "main.elf")
            if not os.path.exists(main_elf):
                main_elf = os.path.join(proj_dir, "build", "main.elf")
            if not os.path.exists(main_elf):
                return False, {"error": "Compilation failed: neither neorv32_exe.bin nor main.elf found."}

            objcopy_bin = f"{riscv_prefix}objcopy"
            elf_bin = os.path.join(proj_dir, "elf.bin")
            subprocess.run([objcopy_bin, "-I", "elf32-littleriscv", "-O", "binary", main_elf, elf_bin],
                           cwd=proj_dir, env=env, check=True)
            with open(elf_bin, "rb") as f:
                raw_data = f.read()
            padded_size = (len(raw_data) + 3) & ~3
            raw_data = raw_data.ljust(padded_size, b"\x00")
            checksum = 0
            for i in range(0, padded_size, 4):
                val = struct.unpack("<I", raw_data[i:i+4])[0]
                checksum = (checksum + val) & 0xFFFFFFFF
            checksum = (~checksum) & 0xFFFFFFFF
            header = struct.pack("<IIII", 0x214F454E, 0x00000000, padded_size, checksum)
            with open(out_bin_path, "wb") as f:
                f.write(header + raw_data)
        else:
            shutil.copy2(built_bin, out_bin_path)

        file_size = os.path.getsize(out_bin_path)
        emit(f"[BUILD SUCCESS] Target ready: {out_bin_path} ({file_size} bytes)\n")

        # Parse memory utilization from ELF if available
        mem_info = cls.parse_elf_memory(proj_dir, riscv_prefix, env)
        return True, {
            "bin_path": out_bin_path,
            "total_size": file_size,
            "memory": mem_info
        }

    @classmethod
    def parse_elf_memory(cls, proj_dir, riscv_prefix, env=None):
        """Extracts text, data, bss sizes using riscv-size."""
        main_elf = os.path.join(proj_dir, "main.elf")
        if not os.path.exists(main_elf):
            main_elf = os.path.join(proj_dir, "build", "main.elf")
        if not os.path.exists(main_elf):
            return {}

        size_bin = f"{riscv_prefix}size"
        try:
            res = subprocess.run([size_bin, main_elf], stdout=subprocess.PIPE, text=True, check=True, env=env)
            lines = res.stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                text = int(parts[0])
                data = int(parts[1])
                bss = int(parts[2])
                imem_used = text + data
                dmem_used = data + bss
                return {
                    "text": text,
                    "data": data,
                    "bss": bss,
                    "imem_used": imem_used,
                    "dmem_used": dmem_used,
                    "rom_limit": 128 * 1024,
                    "ram_limit": 64 * 1024,
                    "bram_total_budget": 225 * 1024
                }
        except Exception:
            pass
        return {}

    # -------------------------------------------------------------
    # Serial Port Management
    # -------------------------------------------------------------
    @staticmethod
    def list_ports(filter_virtual=True):
        ports = serial.tools.list_ports.comports()
        results = []
        for p in ports:
            dev = p.device
            # Filter out virtual Linux /dev/ttyS* ports
            if filter_virtual and sys.platform.startswith("linux"):
                if dev.startswith("/dev/ttyS") and not dev.startswith("/dev/ttyUSB") and not dev.startswith("/dev/ttyACM"):
                    continue
            results.append({
                "device": dev,
                "description": p.description,
                "hwid": p.hwid or ""
            })
        return results

    def open_serial(self):
        self.close_serial()
        self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
        return self.ser

    def close_serial(self):
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None

    # -------------------------------------------------------------
    # Bootloader Flasher
    # -------------------------------------------------------------
    def flash_binary(self, bin_path, log_cb=None, timeout_sec=12.0):
        """
        Flashes firmware bin_path using NEORV32 serial bootloader.
        """
        if not os.path.exists(bin_path):
            return False, {"error": f"Binary file not found: {bin_path}"}

        def emit(msg):
            if log_cb:
                log_cb(msg)

        emit(f"[FLASHER] Connecting to {self.port} @ {self.baud} baud...\n")
        self.close_serial()
        time.sleep(0.3)

        flash_ser = None
        for _ in range(5):
            try:
                flash_ser = serial.Serial(self.port, self.baud, timeout=0.2)
                break
            except Exception:
                time.sleep(0.3)

        if not flash_ser or not flash_ser.is_open:
            return False, {"error": f"Could not open serial port {self.port}."}

        bootloader_ready = False
        emit("[FLASHER] Probing NEORV32 bootloader prompt...\n")
        # Send space and newlines (aborts auto-boot countdown)
        for _ in range(3):
            flash_ser.write(b" \r\n")
            flash_ser.flush()
            time.sleep(0.15)
            resp = flash_ser.read(flash_ser.in_waiting).decode('latin-1', errors='ignore')
            if "CMD:>" in resp or "bootloader" in resp.lower() or "Available CMDs" in resp:
                bootloader_ready = True
                break

        if not bootloader_ready:
            flash_ser.write(b"h\r\n")
            flash_ser.flush()
            time.sleep(0.2)
            resp = flash_ser.read(flash_ser.in_waiting).decode('latin-1', errors='ignore')
            if "CMD:>" in resp or "Available CMDs" in resp:
                bootloader_ready = True

        if not bootloader_ready:
            emit("[FLASHER] Waiting for bootloader prompt / target reset...\n")
            t_end = time.time() + timeout_sec
            while time.time() < t_end:
                flash_ser.write(b" \r\n")
                time.sleep(0.2)
                resp = flash_ser.read(flash_ser.in_waiting).decode('latin-1', errors='ignore')
                if "CMD:>" in resp or "Available CMDs" in resp:
                    bootloader_ready = True
                    break

        if not bootloader_ready:
            flash_ser.close()
            return False, {"error": "Bootloader prompt CMD:> did not respond. Reset GEMMrv32 target and retry."}

        emit("[FLASHER] Bootloader CMD:> ready. Uploading image...\n")
        with open(bin_path, "rb") as f:
            data = f.read()

        file_size = len(data)
        checksum_calc = 0
        for i in range(0, file_size, 4):
            chunk = data[i:i+4]
            if len(chunk) < 4:
                chunk += b"\x00" * (4 - len(chunk))
            checksum_calc = (checksum_calc + struct.unpack("<I", chunk)[0]) & 0xFFFFFFFF

        flash_ser.reset_input_buffer()
        flash_ser.write(b"u")  # upload command
        time.sleep(0.3)
        flash_ser.read(flash_ser.in_waiting)

        # High-throughput streaming
        chunk_size = 1024
        for offset in range(0, file_size, chunk_size):
            chunk = data[offset:offset+chunk_size]
            flash_ser.write(chunk)
            flash_ser.flush()
            time.sleep(0.005)

        time.sleep(0.6)
        boot_resp = flash_ser.read(flash_ser.in_waiting).decode('latin-1', errors='ignore')

        if "ERROR" in boot_resp.upper() or "CHECKSUM" in boot_resp.upper():
            flash_ser.close()
            return False, {"error": f"Bootloader rejected firmware: {boot_resp.strip()}"}

        # Boot application
        flash_ser.write(b"e")
        time.sleep(0.3)
        self.ser = flash_ser

        emit(f"[FLASHER SUCCESS] Uploaded {file_size} bytes (Checksum 0x{checksum_calc:08X}). Application booted.\n")
        return True, {
            "bytes_flashed": file_size,
            "checksum": f"0x{checksum_calc:08X}",
            "response": boot_resp.strip()
        }

    # -------------------------------------------------------------
    # Neural Network Image Preprocessing & Transmission
    # -------------------------------------------------------------
    def send_image(self, img_path, dataset_type="digits", log_cb=None):
        if not os.path.exists(img_path):
            return False, {"error": f"Image file not found: {img_path}"}

        if not self.ser or not self.ser.is_open:
            self.open_serial()

        def emit(msg):
            if log_cb:
                log_cb(msg)

        is_digit = "digit" in dataset_type.lower()
        if is_digit:
            emit(f"[INFERENCE] Preprocessing {os.path.basename(img_path)} (28x28 Grayscale)...\n")
            im = Image.open(img_path).convert("L")
            im = im.resize((28, 28), Image.Resampling.BILINEAR)
            arr = np.array(im, dtype=np.float32)
            norm = np.clip(np.round(arr / 255.0 * 127.0), 0, 127).astype(np.int8)
            payload = norm.tobytes()

            self.ser.reset_input_buffer()
            self.ser.write(b"!IMG" + payload)
            self.ser.flush()
            emit(f"[INFERENCE] Sent !IMG + 784 bytes over {self.port}.\n")
        else:
            emit(f"[INFERENCE] Preprocessing {os.path.basename(img_path)} (32x32 RGB)...\n")
            im = Image.open(img_path).convert("RGB")
            im = im.resize((32, 32), Image.Resampling.BILINEAR)
            arr = np.array(im, dtype=np.float32)
            arr = np.transpose(arr, (2, 0, 1)) # CHW format
            norm = (arr - 128.0).astype(np.int8)
            payload = norm.tobytes()

            self.ser.reset_input_buffer()
            self.ser.write(payload)
            self.ser.flush()
            emit(f"[INFERENCE] Sent 3072 bytes (32x32 RGB) over {self.port}.\n")

        return True, {"bytes_sent": len(payload), "dataset": dataset_type}

    # -------------------------------------------------------------
    # CFS Telemetry / Systolic Register Parser
    # -------------------------------------------------------------
    @staticmethod
    def parse_cfs_telemetry(raw_line):
        """
        Parses systolic array telemetry frames formatted as:
        !CFS:REG0=0x80001000,REG1=0x80002000,REG2=0x80003000,SA=8,SB=8,SC=8,CMD=0x05,DIM=0x080808,CYC=1420
        """
        if "!CFS:" not in raw_line:
            return None

        line = raw_line.split("!CFS:")[1].strip()
        parts = line.split(",")
        fields = {}
        for p in parts:
            if "=" in p:
                k, v = p.split("=", 1)
                k = k.strip()
                v = v.strip()
                try:
                    if v.startswith("0x") or v.startswith("0X"):
                        fields[k] = int(v, 16)
                    else:
                        fields[k] = int(v)
                except ValueError:
                    fields[k] = v

        reg6 = fields.get("CMD", fields.get("REG6", 0))
        reg7 = fields.get("DIM", fields.get("REG7", 0))

        start = bool(reg6 & 1)
        clear = bool(reg6 & 2)
        done = bool(reg6 & 4)

        tile_m = (reg7 >> 16) & 0xFF
        tile_k = (reg7 >> 8) & 0xFF
        tile_n = reg7 & 0xFF

        return {
            "A_ptr": f"0x{fields.get('REG0', 0):08X}",
            "B_ptr": f"0x{fields.get('REG1', 0):08X}",
            "C_ptr": f"0x{fields.get('REG2', 0):08X}",
            "stride_A": fields.get("SA", fields.get("REG3", 0)),
            "stride_B": fields.get("SB", fields.get("REG4", 0)),
            "stride_C": fields.get("SC", fields.get("REG5", 0)),
            "cmd_reg": f"0x{reg6:02X}",
            "cmd_start": start,
            "clear_acc": clear,
            "status_done": done,
            "tile_M": tile_m,
            "tile_K": tile_k,
            "tile_N": tile_n,
            "cycles": fields.get("CYC", 0),
            "macs": tile_m * tile_k * tile_n,
            "raw": raw_line.strip()
        }


# -----------------------------------------------------------------
# Autonomous CLI Interface
# -----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GEMMrv32 Headless CLI & Engine")
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # Subcommand: list-ports
    subparsers.add_parser("list-ports", help="List available serial/COM ports")

    # Subcommand: build
    p_build = subparsers.add_parser("build", help="Build RISC-V source project into .bin")
    p_build.add_argument("--project", "-p", required=True, help="Project name or directory path")
    p_build.add_argument("--opt", "-O", default="-O2", help="Optimization level (-O0, -O1, -O2, -Os)")
    p_build.add_argument("--output", "-o", default=None, help="Output .bin file path")
    p_build.add_argument("--json", action="store_true", help="Output results in JSON format")

    # Subcommand: flash
    p_flash = subparsers.add_parser("flash", help="Flash binary firmware onto NEORV32 board")
    p_flash.add_argument("binary", nargs="?", default=None, help="Path to .bin file (positional)")
    p_flash.add_argument("--bin", "-b", dest="bin_opt", default=None, help="Path to .bin file (flag)")
    p_flash.add_argument("--port", "-p", default=None, help="Serial port (e.g., /dev/ttyUSB1)")
    p_flash.add_argument("--baud", "-baude", "--baude", type=int, default=19200, help="Baud rate (default 19200)")
    p_flash.add_argument("--console", "--open-console", "-c", action="store_true", help="Launch interactive TUI console immediately after flashing")
    p_flash.add_argument("--json", action="store_true", help="Output results in JSON format")

    # Subcommand: infer
    p_infer = subparsers.add_parser("infer", help="Send test image to GEMMrv32 SoC for neural network inference")
    p_infer.add_argument("--image", "-i", required=True, help="Path to image file (.png/.jpg)")
    p_infer.add_argument("--dataset", "-d", default="digits", choices=["digits", "vehicles"], help="Dataset type")
    p_infer.add_argument("--port", default=None, help="Serial port")
    p_infer.add_argument("--baud", type=int, default=19200, help="Baud rate")
    p_infer.add_argument("--timeout", type=float, default=3.0, help="Wait timeout for prediction response")
    p_infer.add_argument("--json", action="store_true", help="Output results in JSON format")

    # Subcommand: console / monitor
    p_console = subparsers.add_parser("console", aliases=["monitor"], help="Launch interactive full-screen TUI hardware console")
    p_console.add_argument("--port", "-p", default=None, help="Serial port (e.g., /dev/ttyUSB1)")
    p_console.add_argument("--baud", "-b", "--baude", type=int, default=19200, help="Baud rate (default 19200)")

    args = parser.parse_args()

    if args.subcommand in ("console", "monitor"):
        from gemm_tui import launch_console
        launch_console(port=args.port, baud=args.baud)
        sys.exit(0)

    elif args.subcommand == "list-ports":
        ports = GEMMEngine.list_ports()
        print(json.dumps(ports, indent=2))
        sys.exit(0)

    elif args.subcommand == "build":
        proj_dir = args.project
        if not os.path.isabs(proj_dir):
            candidate = os.path.join(DEFAULT_PROJECTS_DIR, proj_dir)
            if os.path.isdir(candidate):
                proj_dir = candidate

        def log_handler(msg):
            if not args.json:
                sys.stdout.write(msg)
                sys.stdout.flush()

        success, res = GEMMEngine.build_project(proj_dir, opt_level=args.opt, out_bin_path=args.output, log_cb=log_handler)
        if args.json:
            res["success"] = success
            print(json.dumps(res, indent=2))
        sys.exit(0 if success else 1)

    elif args.subcommand == "flash":
        bin_path = args.binary or args.bin_opt
        if not bin_path:
            sys.stderr.write("error: please provide binary file as argument or via --bin/-b\n")
            sys.exit(2)

        engine = GEMMEngine(port=args.port, baud=args.baud)
        def log_handler(msg):
            if not args.json:
                sys.stdout.write(msg)
                sys.stdout.flush()

        success, res = engine.flash_binary(bin_path, log_cb=log_handler)
        if args.json:
            res["success"] = success
            print(json.dumps(res, indent=2))

        if success and args.console:
            from gemm_tui import launch_console
            launch_console(port=engine.port, baud=engine.baud)

        sys.exit(0 if success else 1)

    elif args.subcommand == "infer":
        engine = GEMMEngine(port=args.port, baud=args.baud)
        def log_handler(msg):
            if not args.json:
                sys.stdout.write(msg)
                sys.stdout.flush()

        success, res = engine.send_image(args.image, dataset_type=args.dataset, log_cb=log_handler)
        if success and engine.ser:
            t_end = time.time() + args.timeout
            lines = []
            while time.time() < t_end:
                if engine.ser.in_waiting:
                    line = engine.ser.readline().decode('latin-1', errors='replace').strip()
                    if line:
                        lines.append(line)
                        if not args.json:
                            print(f"[RECV] {line}")
                time.sleep(0.05)
            res["responses"] = lines
            engine.close_serial()

        if args.json:
            res["success"] = success
            print(json.dumps(res, indent=2))
        sys.exit(0 if success else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
