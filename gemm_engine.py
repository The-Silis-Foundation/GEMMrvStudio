"""
GEMMrv32 Core Engine & Headless CLI
Provides programmatic & command-line APIs for:
  - Compiling RISC-V projects to NEORV32-compatible raw binaries
  - Bootloader handshake and firmware flashing
  - UART terminal I/O & command sending
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
import serial
import serial.tools.list_ports
import numpy as np
from PIL import Image

# Global defaults
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BINS_DIR = os.path.join(REPO_DIR, "example", "bin") if os.path.isdir(os.path.join(REPO_DIR, "example", "bin")) else r"C:\Users\Lenovo\Desktop\bins"
DEFAULT_SCRIPTS_DIR = os.path.join(REPO_DIR, "example", "src") if os.path.isdir(os.path.join(REPO_DIR, "example", "src")) else r"C:\Users\Lenovo\Desktop\scripts"
DEFAULT_CAR_IMAGES_DIR = r"C:\Users\Lenovo\Desktop\car_images"
DEFAULT_DIGIT_IMAGES_DIR = r"C:\Users\Lenovo\Desktop\digit_images"
DEFAULT_PROJECTS_DIR = os.path.join(REPO_DIR, "example", "src") if os.path.isdir(os.path.join(REPO_DIR, "example", "src")) else r"C:\Users\Lenovo\Desktop\GEMMrv32\GEMMrv32-main\sw\example"
NEORV32_HOME_DIR = r"C:/Users/Lenovo/Desktop/GEMMrv32"
DEFAULT_MAKE_PATH = r"D:\vivado\2025.2\Vivado\gnuwin\bin\make.exe"
DEFAULT_GCC_DIR = r"D:\vivado\2025.2\gnu\riscv\nt\bin"

class GEMMEngine:
    def __init__(self, port="COM10", baud=19200):
        self.port = port
        self.baud = baud
        self.ser = None

    # -------------------------------------------------------------
    # Toolchain: Project Build & Packaging
    # -------------------------------------------------------------
    @staticmethod
    def ensure_makefile(proj_dir, target_name):
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

        makefile_path = os.path.join(proj_dir, "makefile")
        gemm_inc_dir = r"C:/Users/Lenovo/Desktop/scripts/gemm_lib"

        content = f"""# Auto-generated NEORV32 GEMMrv32 makefile
APP_SRC ?= {c_sources}
NEORV32_HOME ?= {NEORV32_HOME_DIR}
EFFORT ?= -O2
USER_FLAGS ?= -I. -I"{gemm_inc_dir}"

include $(NEORV32_HOME)/sw/common/common.mk
"""
        with open(makefile_path, "w", encoding="utf-8") as f:
            f.write(content)
        return makefile_path

    @classmethod
    def build_project(cls, proj_dir, opt_level="-O2", out_bin_path=None, log_cb=None):
        """
        Builds a RISC-V project in proj_dir using Vivado GNU make/gcc.
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
        makefile_path = os.path.join(proj_dir, "makefile")
        if not os.path.exists(makefile_path):
            makefile_path = os.path.join(proj_dir, "Makefile")
        if not os.path.exists(makefile_path):
            emit(f"[INFO] Auto-generating makefile for {target_name}...\n")
            cls.ensure_makefile(proj_dir, target_name)

        env = os.environ.copy()
        env["PATH"] = DEFAULT_GCC_DIR + ";" + os.path.dirname(DEFAULT_MAKE_PATH) + ";" + env.get("PATH", "")

        cmd = [
            DEFAULT_MAKE_PATH,
            f"NEORV32_HOME={NEORV32_HOME_DIR}",
            "RISCV_PREFIX=riscv64-unknown-elf-",
            f"EFFORT={opt_level}",
            "clean_all",
            "elf"
        ]
        emit(f"[BUILD] Running: {' '.join(cmd)}\n")

        proc = subprocess.Popen(cmd, cwd=proj_dir, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        output_lines = []
        for line in proc.stdout:
            output_lines.append(line)
            emit(line)
        proc.wait()

        main_elf = os.path.join(proj_dir, "main.elf")
        if not os.path.exists(main_elf):
            main_elf = os.path.join(proj_dir, "build", "main.elf")

        if not os.path.exists(main_elf):
            return False, {
                "error": "Compilation failed; main.elf not generated.",
                "exit_code": proc.returncode,
                "log": "".join(output_lines)
            }

        # Convert to binary via objcopy
        elf_bin = os.path.join(proj_dir, "elf.bin")
        objcopy_exe = os.path.join(DEFAULT_GCC_DIR, "riscv64-unknown-elf-objcopy.exe")
        obj_cmd = [objcopy_exe, "-I", "elf32-littleriscv", "-O", "binary", main_elf, elf_bin]
        emit(f"[BUILD] Extracting flat binary with objcopy...\n")
        subprocess.run(obj_cmd, cwd=proj_dir, env=env, check=True)

        if not os.path.exists(elf_bin):
            return False, {"error": "objcopy failed to produce elf.bin."}

        # Pack NEORV32 bootloader image (16-byte header: signature, base_addr, size, checksum)
        with open(elf_bin, "rb") as f:
            raw_data = f.read()

        # Word-align / pad binary
        padded_size = (len(raw_data) + 3) & ~3
        raw_data = raw_data.ljust(padded_size, b"\x00")

        # Sum 32-bit words
        checksum = 0
        for i in range(0, padded_size, 4):
            val = struct.unpack("<I", raw_data[i:i+4])[0]
            checksum = (checksum + val) & 0xFFFFFFFF
        checksum = (~checksum) & 0xFFFFFFFF

        base_addr = 0x00000000  # Default IMEM base address
        header = struct.pack("<IIII", 0x214F454E, base_addr, padded_size, checksum)
        with open(out_bin_path, "wb") as f:
            f.write(header + raw_data)

        total_size = len(header) + len(raw_data)
        emit(f"[BUILD SUCCESS] Generated {out_bin_path} ({total_size} bytes, raw={padded_size})\n")
        return True, {
            "bin_path": out_bin_path,
            "raw_size": padded_size,
            "total_size": total_size,
            "checksum": f"0x{checksum:08X}"
        }

    # -------------------------------------------------------------
    # Serial Port Management
    # -------------------------------------------------------------
    @staticmethod
    def list_ports():
        ports = serial.tools.list_ports.comports()
        return [{"device": p.device, "description": p.description, "hwid": p.hwid} for p in ports]

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

        emit(f"[FLASHER] Connecting to {self.port} @ {self.baud}...\n")
        self.close_serial()
        time.sleep(0.3)

        flash_ser = None
        for _ in range(4):
            try:
                flash_ser = serial.Serial(self.port, self.baud, timeout=0.2)
                break
            except Exception:
                time.sleep(0.3)

        if not flash_ser or not flash_ser.is_open:
            return False, {"error": f"Could not open {self.port}."}

        bootloader_ready = False
        emit("[FLASHER] Probing NEORV32 bootloader prompt...\n")
        for _ in range(5):
            flash_ser.write(b"\r\n")
            flash_ser.flush()
            time.sleep(0.1)
            resp = flash_ser.read(flash_ser.in_waiting).decode('latin-1', errors='ignore')
            if "CMD:>" in resp or "bootloader" in resp.lower():
                bootloader_ready = True
                break

        if not bootloader_ready:
            emit("[FLASHER] Waiting for user / FPGA reset (press PROG/RESET on board)...\n")
            t_end = time.time() + timeout_sec
            while time.time() < t_end:
                flash_ser.write(b"\r\n")
                time.sleep(0.2)
                resp = flash_ser.read(flash_ser.in_waiting).decode('latin-1', errors='ignore')
                if "CMD:>" in resp:
                    bootloader_ready = True
                    break

        if not bootloader_ready:
            flash_ser.close()
            return False, {"error": "Bootloader prompt CMD:> did not respond. Reset FPGA and retry."}

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
        flash_ser.write(b"u") # upload command
        time.sleep(0.3)
        flash_ser.read(flash_ser.in_waiting)

        chunk_size = 256
        for offset in range(0, file_size, chunk_size):
            chunk = data[offset:offset+chunk_size]
            flash_ser.write(chunk)
            flash_ser.flush()
            time.sleep(0.015)

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

        # Decode REG6 bitfields
        start = bool(reg6 & 1)
        clear = bool(reg6 & 2)
        done = bool(reg6 & 4)

        # Decode REG7 tile dims (M, K, N)
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
    p_flash.add_argument("--bin", "-b", required=True, help="Path to .bin file")
    p_flash.add_argument("--port", default="COM10", help="Serial port (e.g., COM10)")
    p_flash.add_argument("--baud", type=int, default=19200, help="Baud rate (default 19200)")
    p_flash.add_argument("--json", action="store_true", help="Output results in JSON format")

    # Subcommand: infer
    p_infer = subparsers.add_parser("infer", help="Send test image to FPGA for neural network inference")
    p_infer.add_argument("--image", "-i", required=True, help="Path to image file (.png/.jpg)")
    p_infer.add_argument("--dataset", "-d", default="digits", choices=["digits", "vehicles"], help="Dataset type")
    p_infer.add_argument("--port", default="COM10", help="Serial port")
    p_infer.add_argument("--baud", type=int, default=19200, help="Baud rate")
    p_infer.add_argument("--timeout", type=float, default=3.0, help="Wait timeout for prediction response")
    p_infer.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    if args.subcommand == "list-ports":
        ports = GEMMEngine.list_ports()
        print(json.dumps(ports, indent=2))
        sys.exit(0)

    elif args.subcommand == "build":
        proj_dir = args.project
        if not os.path.isabs(proj_dir):
            candidate = os.path.join(DEFAULT_SCRIPTS_DIR, proj_dir)
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
        engine = GEMMEngine(port=args.port, baud=args.baud)
        def log_handler(msg):
            if not args.json:
                sys.stdout.write(msg)
                sys.stdout.flush()

        success, res = engine.flash_binary(args.bin, log_cb=log_handler)
        if args.json:
            res["success"] = success
            print(json.dumps(res, indent=2))
        sys.exit(0 if success else 1)

    elif args.subcommand == "infer":
        engine = GEMMEngine(port=args.port, baud=args.baud)
        def log_handler(msg):
            if not args.json:
                sys.stdout.write(msg)
                sys.stdout.flush()

        success, res = engine.send_image(args.image, dataset_type=args.dataset, log_cb=log_handler)
        if success and engine.ser:
            # Read response up to timeout
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
