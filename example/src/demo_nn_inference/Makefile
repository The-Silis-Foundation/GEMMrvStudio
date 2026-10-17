# Auto-generated NEORV32 GEMMrv32 Makefile
MARCH ?= rv32im_zicsr_zifencei
RISCV_PREFIX ?= riscv-none-elf-
EFFORT ?= -O2

APP_SRC ?= main.c  ../gemm_lib/gemmrv_hw.c ../gemm_lib/gemmrv_nn.c
APP_INC += -I . -I /home/econsystems/random/GEMMrvStudio/example/src/gemm_lib

# GEMMrv32 On-Chip SRAM Configuration (128 KB IMEM, 64 KB DMEM)
USER_FLAGS += -Wl,--defsym,__neorv32_rom_size=128k
USER_FLAGS += -Wl,--defsym,__neorv32_ram_size=64k

NEORV32_HOME ?= /home/econsystems/random/GEMMrv32
include $(NEORV32_HOME)/sw/common/common.mk
