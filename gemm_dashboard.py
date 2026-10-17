"""
GEMMrvStudio - Industrial RISC-V IDE & Systolic Accelerator Workbench
The Silis Foundation - Open-Source RISC-V SoC with Systolic GEMM Accelerator
Architecture: Arduino IDE 2.x Industrial Style + QScintilla + Dual Themes (Dark/Light)
"""

import sys
import os
import glob
import math
import time
import re
import serial
import serial.tools.list_ports
from PyQt5 import QtCore, QtGui, QtWidgets, QtSvg
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.Qsci import (
    QsciScintilla,
    QsciLexerCPP
)

from gemm_engine import (
    GEMMEngine,
    DEFAULT_BINS_DIR,
    DEFAULT_PROJECTS_DIR,
    DEFAULT_GEMM_LIB_DIR,
    REPO_DIR
)

# ---------------------------------------------------------------------------
# Industrial Color Palettes (Arduino 2.x Inspired: Dark & Light)
# ---------------------------------------------------------------------------
THEMES = {
    "dark": {
        "name": "Industrial Dark",
        "bg_primary": "#1e1e1e",
        "bg_secondary": "#252526",
        "bg_activity": "#333333",
        "bg_header": "#2d2d2d",
        "bg_editor": "#1e1e1e",
        "bg_drawer": "#181818",
        "bg_card": "#2a2a2a",
        "fg_primary": "#cccccc",
        "fg_secondary": "#888888",
        "fg_active": "#ffffff",
        "accent": "#00979c",       # Arduino teal
        "accent_hover": "#008184",
        "accent_active": "#00676a",
        "border": "#3c3c3c",
        "border_light": "#444444",
        "tab_active_bg": "#1e1e1e",
        "tab_inactive_bg": "#2d2d2d",
        "terminal_bg": "#0c0d0e",
        "terminal_fg": "#4af626",
        "caret": "#ffffff",
        "selection_bg": "#264f78",
        "line_number_bg": "#1e1e1e",
        "line_number_fg": "#5a5a5a",
    },
    "light": {
        "name": "Industrial Light",
        "bg_primary": "#ffffff",
        "bg_secondary": "#f3f3f3",
        "bg_activity": "#e8e8e8",
        "bg_header": "#f8f8f8",
        "bg_editor": "#ffffff",
        "bg_drawer": "#fafafa",
        "bg_card": "#f0f0f0",
        "fg_primary": "#333333",
        "fg_secondary": "#777777",
        "fg_active": "#000000",
        "accent": "#00979c",
        "accent_hover": "#008184",
        "accent_active": "#00676a",
        "border": "#cccccc",
        "border_light": "#e0e0e0",
        "tab_active_bg": "#ffffff",
        "tab_inactive_bg": "#ececec",
        "terminal_bg": "#1c1c1c",
        "terminal_fg": "#80ff80",
        "caret": "#000000",
        "selection_bg": "#add6ff",
        "line_number_bg": "#ffffff",
        "line_number_fg": "#999999",
    }
}

ICONS_DIR = os.path.join(REPO_DIR, "resources", "icons")


def load_svg_icon(name, color="#ffffff", size=20):
    svg_path = os.path.join(ICONS_DIR, f"{name}.svg")
    if not os.path.exists(svg_path):
        return QtGui.QIcon()
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_content = f.read()
    # Replace stroke or currentColor with given color
    svg_content = svg_content.replace('stroke="currentColor"', f'stroke="{color}"')
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg_content.encode("utf-8")))
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return QtGui.QIcon(pixmap)


# ---------------------------------------------------------------------------
# Background Worker Threads
# ---------------------------------------------------------------------------
class BuildWorker(QThread):
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(bool, dict)

    def __init__(self, proj_dir, opt_level="-O2", out_bin=None):
        super().__init__()
        self.proj_dir = proj_dir
        self.opt_level = opt_level
        self.out_bin = out_bin

    def run(self):
        success, res = GEMMEngine.build_project(
            self.proj_dir,
            opt_level=self.opt_level,
            out_bin_path=self.out_bin,
            log_cb=self.log_signal.emit
        )
        self.done_signal.emit(success, res)


class FlashWorker(QThread):
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(bool, dict)

    def __init__(self, port, baud, bin_path):
        super().__init__()
        self.port = port
        self.baud = baud
        self.bin_path = bin_path

    def run(self):
        engine = GEMMEngine(port=self.port, baud=self.baud)
        success, res = engine.flash_binary(self.bin_path, log_cb=self.log_signal.emit)
        self.done_signal.emit(success, res)


class SerialMonitorWorker(QThread):
    rx_signal = pyqtSignal(str)

    def __init__(self, port, baud):
        super().__init__()
        self.port = port
        self.baud = baud
        self.running = True
        self.ser = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
        except Exception as e:
            self.rx_signal.emit(f"\n[ERROR] Cannot open {self.port}: {e}\n")
            return

        while self.running and self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting:
                    data = self.ser.read(self.ser.in_waiting).decode("latin-1", errors="replace")
                    if data:
                        self.rx_signal.emit(data)
            except Exception:
                break
            time.sleep(0.01)

    def send_data(self, data_bytes):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(data_bytes)
            except Exception:
                pass

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.wait(1000)


# ---------------------------------------------------------------------------
# GEMM Coprocessor Practical Developer Workbench
# ---------------------------------------------------------------------------
class GEMMDevWorkbench(QtWidgets.QWidget):
    """
    Practical GEMM hardware tools for firmware developers:
      1. Memory Zero-Padding & Array Header Scaffolder
      2. Tiling & SRAM Scratchpad Budget Planner
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Title
        lbl_title = QtWidgets.QLabel("GEMM Coprocessor Assistant")
        lbl_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #00979c;")
        layout.addWidget(lbl_title)

        # Tabs for Workbench Tools
        self.tabs = QtWidgets.QTabWidget()
        self.tab_pad = QtWidgets.QWidget()
        self.tab_tiling = QtWidgets.QWidget()

        self.tabs.addTab(self.tab_pad, "Stride Padding & Arrays")
        self.tabs.addTab(self.tab_tiling, "Tiling & SRAM Planner")
        layout.addWidget(self.tabs)

        self.setup_pad_tab()
        self.setup_tiling_tab()

    def setup_pad_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_pad)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        desc = QtWidgets.QLabel(
            "Hardware requires strides as multiples of 8 and 4-byte aligned pointers.\n"
            "Calculates alignment and generates C struct/array definitions."
        )
        desc.setStyleSheet("font-size: 10px; color: #888888;")
        layout.addWidget(desc)

        # Inputs
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(6)

        grid.addWidget(QtWidgets.QLabel("Matrix Name:"), 0, 0)
        self.txt_mat_name = QtWidgets.QLineEdit("layer1_weights")
        grid.addWidget(self.txt_mat_name, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Rows (M):"), 1, 0)
        self.spn_m = QtWidgets.QSpinBox()
        self.spn_m.setRange(1, 255)
        self.spn_m.setValue(10)
        grid.addWidget(self.spn_m, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Cols (K/N):"), 2, 0)
        self.spn_k = QtWidgets.QSpinBox()
        self.spn_k.setRange(1, 255)
        self.spn_k.setValue(14)
        grid.addWidget(self.spn_k, 2, 1)

        grid.addWidget(QtWidgets.QLabel("Data Type:"), 3, 0)
        self.cmb_dtype = QtWidgets.QComboBox()
        self.cmb_dtype.addItems(["int8_t (Input)", "int32_t (Accumulator)"])
        grid.addWidget(self.cmb_dtype, 3, 1)

        layout.addLayout(grid)

        btn_calc = QtWidgets.QPushButton("Compute Padded Stride & Generate C Code")
        btn_calc.setStyleSheet(
            "background-color: #00979c; color: #ffffff; font-weight: bold; padding: 6px; border-radius: 3px;"
        )
        btn_calc.clicked.connect(self.generate_c_code)
        layout.addWidget(btn_calc)

        self.lbl_pad_stats = QtWidgets.QLabel("Raw Size: 140 B | Padded Size (Stride 16): 160 B (+20 B padding)")
        self.lbl_pad_stats.setStyleSheet("font-weight: bold; font-size: 10px; color: #00979c;")
        layout.addWidget(self.lbl_pad_stats)

        self.txt_output_code = QtWidgets.QPlainTextEdit()
        self.txt_output_code.setReadOnly(True)
        self.txt_output_code.setFont(QtGui.QFont("Monospace", 9))
        layout.addWidget(self.txt_output_code)

        self.generate_c_code()

    def generate_c_code(self):
        name = self.txt_mat_name.text().strip() or "matrix"
        m = self.spn_m.value()
        k = self.spn_k.value()
        is_acc = (self.cmb_dtype.currentIndex() == 1)

        elem_size = 4 if is_acc else 1
        stride = (k + 7) & ~7
        raw_bytes = m * k * elem_size
        padded_bytes = m * stride * elem_size
        diff = padded_bytes - raw_bytes

        self.lbl_pad_stats.setText(
            f"Raw: {raw_bytes} B | Stride: {stride} (aligned to 8) | Padded: {padded_bytes} B (+{diff} B padding)"
        )

        dtype_str = "int32_t" if is_acc else "int8_t"
        struct_type = "gemmrv_mat_out" if is_acc else "gemmrv_mat"
        macro_init = "GEMMRV_MAT_OUT" if is_acc else "GEMMRV_MAT"

        code = f"""// GEMMrv32 Memory Alignment Wrapper
// Stride must be multiple of 8; Buffer 4-byte aligned
static {dtype_str} {name}_data[{m} * {stride}] __attribute__((aligned(4))) = {{
    // Elements padded with 0s along columns up to stride {stride}
}};

{struct_type} {name} = {macro_init}({name}_data, {m}, {k}, {stride});
"""
        self.txt_output_code.setPlainText(code)

    def setup_tiling_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_tiling)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        desc = QtWidgets.QLabel(
            "Basys 3 has 64 KB DMEM. Plan systolic tiles to ensure buffer sizes\n"
            "do not overflow SRAM budget during multi-layer inference."
        )
        desc.setStyleSheet("font-size: 10px; color: #888888;")
        layout.addWidget(desc)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(6)

        grid.addWidget(QtWidgets.QLabel("M (Rows A):"), 0, 0)
        self.spn_tile_m = QtWidgets.QSpinBox()
        self.spn_tile_m.setRange(1, 1024)
        self.spn_tile_m.setValue(64)
        grid.addWidget(self.spn_tile_m, 0, 1)

        grid.addWidget(QtWidgets.QLabel("K (Cols A / Rows B):"), 1, 0)
        self.spn_tile_k = QtWidgets.QSpinBox()
        self.spn_tile_k.setRange(1, 1024)
        self.spn_tile_k.setValue(128)
        grid.addWidget(self.spn_tile_k, 1, 1)

        grid.addWidget(QtWidgets.QLabel("N (Cols B):"), 2, 0)
        self.spn_tile_n = QtWidgets.QSpinBox()
        self.spn_tile_n.setRange(1, 1024)
        self.spn_tile_n.setValue(64)
        grid.addWidget(self.spn_tile_n, 2, 1)

        layout.addLayout(grid)

        btn_plan = QtWidgets.QPushButton("Calculate Tiling & Buffer Footprint")
        btn_plan.setStyleSheet(
            "background-color: #00979c; color: #ffffff; font-weight: bold; padding: 6px; border-radius: 3px;"
        )
        btn_plan.clicked.connect(self.calculate_tiling)
        layout.addWidget(btn_plan)

        self.txt_tiling_report = QtWidgets.QPlainTextEdit()
        self.txt_tiling_report.setReadOnly(True)
        self.txt_tiling_report.setFont(QtGui.QFont("Monospace", 9))
        layout.addWidget(self.txt_tiling_report)

        self.calculate_tiling()

    def calculate_tiling(self):
        m = self.spn_tile_m.value()
        k = self.spn_tile_k.value()
        n = self.spn_tile_n.value()

        stride_a = (k + 7) & ~7
        stride_b = (n + 7) & ~7
        stride_c = stride_b

        tiles_m = math.ceil(m / 8)
        tiles_k = math.ceil(k / 8)
        tiles_n = math.ceil(n / 8)
        total_tiles = tiles_m * tiles_k * tiles_n

        size_a = m * stride_a * 1   # int8
        size_b = k * stride_b * 1   # int8
        size_c = m * stride_c * 4   # int32
        total_sram = size_a + size_b + size_c

        dmem_limit = 64 * 1024
        sram_pct = (total_sram / dmem_limit) * 100

        status = "FITS IN SRAM" if total_sram < dmem_limit else "EXCEEDS 64 KB SRAM (Requires Layer Tiling!)"

        report = f"""=== GEMMrv32 Systolic Tiling Report ===
Target Matrix Mult: C[{m}x{n}] = A[{m}x{k}] * B[{k}x{n}]

Strides:
  - Stride A: {stride_a} bytes
  - Stride B: {stride_b} bytes
  - Stride C: {stride_c * 4} bytes (int32 output)

Hardware 8x8 Systolic Tile Grid:
  - M-tiles: {tiles_m}
  - K-tiles: {tiles_k}
  - N-tiles: {tiles_n}
  - Total Dispatched Tiles: {total_tiles}

Memory Footprint:
  - Buffer A (int8):   {size_a:,} bytes
  - Buffer B (int8):   {size_b:,} bytes
  - Buffer C (int32):  {size_c:,} bytes
  - Total Matrices:    {total_sram:,} bytes ({sram_pct:.1f}% of 64 KB DMEM)
  - Allocation Status: {status}
"""
        self.txt_tiling_report.setPlainText(report)


# ---------------------------------------------------------------------------
# Main Industrial IDE Window (PyQt5 + QScintilla + Dual Theme)
# ---------------------------------------------------------------------------
class ArduinoGEMMStudio(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_theme_key = "light"
        self.current_proj_dir = os.path.join(DEFAULT_PROJECTS_DIR, "demo_gemm")
        self.current_bin_file = os.path.join(DEFAULT_BINS_DIR, "demo_gemm.bin")
        self.serial_worker = None

        self.setWindowTitle("GEMMrvStudio - The Silis Foundation")
        self.resize(1300, 840)
        self.setMinimumSize(960, 600)

        # Initialize UI Components
        self.setup_central_layout()
        self.apply_theme(self.current_theme_key)
        self.refresh_ports()
        self.load_project(self.current_proj_dir)

    # -------------------------------------------------------------
    # UI Layout Construction (Arduino 2.x Architecture)
    # -------------------------------------------------------------
    def setup_central_layout(self):
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QtWidgets.QVBoxLayout(main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # 1. Top Industrial Toolbar (Arduino Action Strip)
        self.setup_top_toolbar()

        # 2. Body Area (Activity Bar + Left Drawer + Editor + Bottom Drawer)
        body_container = QtWidgets.QWidget()
        self.body_layout = QtWidgets.QHBoxLayout(body_container)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(0)
        self.main_layout.addWidget(body_container, stretch=1)

        # 2a. Activity Rail (Left Vertical Bar with SVG icons)
        self.setup_activity_rail()

        # 2b. Central Splitter (Left Sidebar / Editor / Bottom Drawer)
        self.setup_workspace_splitters()

        # 3. Bottom Status Bar
        self.setup_status_bar()

    def setup_top_toolbar(self):
        self.toolbar_widget = QtWidgets.QWidget()
        self.toolbar_widget.setFixedHeight(48)
        tb_layout = QtWidgets.QHBoxLayout(self.toolbar_widget)
        tb_layout.setContentsMargins(12, 0, 12, 0)
        tb_layout.setSpacing(10)

        # Verify / Compile Button (Round teal circle with SVG checkmark)
        self.btn_verify = QtWidgets.QPushButton()
        self.btn_verify.setFixedSize(32, 32)
        self.btn_verify.setToolTip("Verify / Compile RISC-V Firmware (Ctrl+R)")
        self.btn_verify.setIcon(load_svg_icon("verify", color="#ffffff", size=18))
        self.btn_verify.clicked.connect(self.action_verify)
        tb_layout.addWidget(self.btn_verify)

        # Upload Button (Round teal circle with SVG arrow)
        self.btn_upload = QtWidgets.QPushButton()
        self.btn_upload.setFixedSize(32, 32)
        self.btn_upload.setToolTip("Upload to GEMMrv32 Board (Ctrl+U)")
        self.btn_upload.setIcon(load_svg_icon("upload", color="#ffffff", size=18))
        self.btn_upload.clicked.connect(self.action_upload)
        tb_layout.addWidget(self.btn_upload)

        # Board & Port Pill Dropdowns (Arduino signature pill selector)
        tb_layout.addSpacing(12)

        lbl_board_tag = QtWidgets.QLabel("Board:")
        lbl_board_tag.setStyleSheet("font-size: 11px; font-weight: bold; color: #888888;")
        tb_layout.addWidget(lbl_board_tag)

        self.cmb_board = QtWidgets.QComboBox()
        self.cmb_board.addItems(["GEMMrv32 Basys 3 (Artix-7 35T)"])
        self.cmb_board.setFixedWidth(230)
        tb_layout.addWidget(self.cmb_board)

        lbl_port_tag = QtWidgets.QLabel("Port:")
        lbl_port_tag.setStyleSheet("font-size: 11px; font-weight: bold; color: #888888;")
        tb_layout.addWidget(lbl_port_tag)

        self.cmb_port = QtWidgets.QComboBox()
        self.cmb_port.setFixedWidth(160)
        tb_layout.addWidget(self.cmb_port)

        btn_rescan = QtWidgets.QPushButton("↻")
        btn_rescan.setFixedSize(26, 26)
        btn_rescan.setToolTip("Rescan Serial Ports")
        btn_rescan.clicked.connect(self.refresh_ports)
        tb_layout.addWidget(btn_rescan)

        lbl_opt_tag = QtWidgets.QLabel("Opt:")
        lbl_opt_tag.setStyleSheet("font-size: 11px; font-weight: bold; color: #888888;")
        tb_layout.addWidget(lbl_opt_tag)

        self.cmb_opt = QtWidgets.QComboBox()
        self.cmb_opt.addItems(["-Os (Size / DOOM)", "-O2 (Speed)", "-O3 (Max Speed)", "-O0 (Debug)"])
        self.cmb_opt.setFixedWidth(135)
        tb_layout.addWidget(self.cmb_opt)

        tb_layout.addStretch()

        # Theme toggle button (Dark / Light)
        self.btn_theme = QtWidgets.QPushButton()
        self.btn_theme.setFixedSize(30, 30)
        self.btn_theme.setToolTip("Toggle Dark / Light Theme")
        self.btn_theme.setIcon(load_svg_icon("theme", color="#00979c", size=18))
        self.btn_theme.clicked.connect(self.toggle_theme)
        tb_layout.addWidget(self.btn_theme)

        # Serial Monitor Toggle Button (Top Right)
        self.btn_term_toggle = QtWidgets.QPushButton()
        self.btn_term_toggle.setFixedSize(30, 30)
        self.btn_term_toggle.setToolTip("Toggle Serial Monitor / Output")
        self.btn_term_toggle.setIcon(load_svg_icon("terminal", color="#00979c", size=18))
        self.btn_term_toggle.clicked.connect(self.toggle_bottom_drawer)
        tb_layout.addWidget(self.btn_term_toggle)

        self.main_layout.addWidget(self.toolbar_widget)

    def setup_activity_rail(self):
        self.activity_rail = QtWidgets.QWidget()
        self.activity_rail.setFixedWidth(46)
        rail_layout = QtWidgets.QVBoxLayout(self.activity_rail)
        rail_layout.setContentsMargins(0, 8, 0, 8)
        rail_layout.setSpacing(6)

        # Explorer Icon Button
        self.btn_act_explorer = QtWidgets.QPushButton()
        self.btn_act_explorer.setFixedSize(46, 38)
        self.btn_act_explorer.setIcon(load_svg_icon("folder", color="#00979c", size=20))
        self.btn_act_explorer.setToolTip("Sketchbook / Files")
        self.btn_act_explorer.clicked.connect(lambda: self.switch_sidebar(0))
        rail_layout.addWidget(self.btn_act_explorer)

        # GEMM Workbench Icon Button
        self.btn_act_gemm = QtWidgets.QPushButton()
        self.btn_act_gemm.setFixedSize(46, 38)
        self.btn_act_gemm.setIcon(load_svg_icon("gemm", color="#00979c", size=20))
        self.btn_act_gemm.setToolTip("GEMM Coprocessor Workbench")
        self.btn_act_gemm.clicked.connect(lambda: self.switch_sidebar(1))
        rail_layout.addWidget(self.btn_act_gemm)

        rail_layout.addStretch()

        self.body_layout.addWidget(self.activity_rail)

    def setup_workspace_splitters(self):
        # Horizontal Splitter: Left Sidebar (Tree/GEMM) vs. Center Workspace
        self.h_splitter = QtWidgets.QSplitter(Qt.Horizontal)
        self.body_layout.addWidget(self.h_splitter, stretch=1)

        # Left Sidebar Stack
        self.sidebar_stack = QtWidgets.QStackedWidget()
        self.sidebar_stack.setMinimumWidth(240)
        self.sidebar_stack.setMaximumWidth(400)

        # Page 0: File Explorer
        self.page_explorer = QtWidgets.QWidget()
        exp_layout = QtWidgets.QVBoxLayout(self.page_explorer)
        exp_layout.setContentsMargins(6, 6, 6, 6)
        exp_layout.setSpacing(4)

        exp_header = QtWidgets.QHBoxLayout()
        lbl_sk = QtWidgets.QLabel("SKETCHBOOK / FILES")
        lbl_sk.setStyleSheet("font-size: 11px; font-weight: bold; color: #888888;")
        exp_header.addWidget(lbl_sk)

        btn_open = QtWidgets.QPushButton("Open...")
        btn_open.setFixedHeight(22)
        btn_open.clicked.connect(self.action_open_project)
        exp_header.addWidget(btn_open)
        exp_layout.addLayout(exp_header)

        self.tree_files = QtWidgets.QListWidget()
        self.tree_files.itemDoubleClicked.connect(self.on_file_double_clicked)
        exp_layout.addWidget(self.tree_files)

        self.sidebar_stack.addWidget(self.page_explorer)

        # Page 1: GEMM Workbench
        self.gemm_workbench = GEMMDevWorkbench()
        self.sidebar_stack.addWidget(self.gemm_workbench)

        self.h_splitter.addWidget(self.sidebar_stack)

        # Right Splitter (Vertical): Code Editor on Top, Terminal / Output on Bottom
        self.v_splitter = QtWidgets.QSplitter(Qt.Vertical)
        self.h_splitter.addWidget(self.v_splitter)
        self.h_splitter.setStretchFactor(1, 4)

        # Code Editor Tabs (QScintilla tabs)
        self.editor_tabs = QtWidgets.QTabWidget()
        self.editor_tabs.setTabsClosable(True)
        self.editor_tabs.tabCloseRequested.connect(self.close_editor_tab)
        self.v_splitter.addWidget(self.editor_tabs)
        self.v_splitter.setStretchFactor(0, 7)

        # Bottom Drawer: Industrial Output & Seamless Serial Monitor
        self.setup_bottom_drawer()
        self.v_splitter.addWidget(self.bottom_drawer)
        self.v_splitter.setStretchFactor(1, 3)

    def setup_bottom_drawer(self):
        self.bottom_drawer = QtWidgets.QWidget()
        drawer_layout = QtWidgets.QVBoxLayout(self.bottom_drawer)
        drawer_layout.setContentsMargins(0, 0, 0, 0)
        drawer_layout.setSpacing(0)

        # Drawer Tab Bar (Output vs. Serial Monitor)
        drawer_bar = QtWidgets.QHBoxLayout()
        drawer_bar.setContentsMargins(8, 4, 8, 4)
        drawer_bar.setSpacing(6)

        self.btn_tab_output = QtWidgets.QPushButton("OUTPUT")
        self.btn_tab_output.setFixedHeight(24)
        self.btn_tab_output.clicked.connect(lambda: self.drawer_stack.setCurrentIndex(0))
        drawer_bar.addWidget(self.btn_tab_output)

        self.btn_tab_serial = QtWidgets.QPushButton("SERIAL MONITOR")
        self.btn_tab_serial.setFixedHeight(24)
        self.btn_tab_serial.clicked.connect(lambda: self.drawer_stack.setCurrentIndex(1))
        drawer_bar.addWidget(self.btn_tab_serial)

        drawer_bar.addSpacing(16)

        self.chk_autoscroll = QtWidgets.QCheckBox("Auto-scroll")
        self.chk_autoscroll.setChecked(True)
        drawer_bar.addWidget(self.chk_autoscroll)

        btn_clear = QtWidgets.QPushButton("Clear")
        btn_clear.setFixedHeight(24)
        btn_clear.clicked.connect(self.clear_current_terminal)
        drawer_bar.addWidget(btn_clear)

        drawer_bar.addStretch()

        # Serial monitor input field
        self.txt_serial_input = QtWidgets.QLineEdit()
        self.txt_serial_input.setPlaceholderText("Send command to board...")
        self.txt_serial_input.returnPressed.connect(self.send_serial_input)
        self.txt_serial_input.setFixedWidth(240)
        drawer_bar.addWidget(self.txt_serial_input)

        btn_send = QtWidgets.QPushButton("Send")
        btn_send.setFixedHeight(24)
        btn_send.clicked.connect(self.send_serial_input)
        drawer_bar.addWidget(btn_send)

        drawer_layout.addLayout(drawer_bar)

        # Drawer Stack
        self.drawer_stack = QtWidgets.QStackedWidget()
        drawer_layout.addWidget(self.drawer_stack)

        # Page 0: Build & Diagnostic Output
        self.txt_output = QtWidgets.QPlainTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setFont(QtGui.QFont("Monospace", 9))
        self.drawer_stack.addWidget(self.txt_output)

        # Page 1: Seamless Terminal Serial Monitor
        self.txt_terminal = QtWidgets.QPlainTextEdit()
        self.txt_terminal.setFont(QtGui.QFont("Monospace", 10))
        self.txt_terminal.setReadOnly(False)  # Allows keystroke capture
        self.txt_terminal.installEventFilter(self)
        self.drawer_stack.addWidget(self.txt_terminal)

    def setup_status_bar(self):
        status = QtWidgets.QStatusBar()
        self.setStatusBar(status)
        status.setFixedHeight(24)

        self.lbl_status = QtWidgets.QLabel("Ready")
        status.addWidget(self.lbl_status, 1)

        self.lbl_toolchain = QtWidgets.QLabel(f"Toolchain: {GEMMEngine.find_riscv_prefix()} | Board: GEMMrv32 Basys 3")
        status.addPermanentWidget(self.lbl_toolchain)

    # -------------------------------------------------------------
    # Theme Application (Industrial Dark / Light)
    # -------------------------------------------------------------
    def apply_theme(self, theme_key):
        self.current_theme_key = theme_key
        t = THEMES[theme_key]

        style = f"""
        QMainWindow, QWidget {{
            background-color: {t["bg_primary"]};
            color: {t["fg_primary"]};
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 11px;
        }}
        /* Top Toolbar */
        QWidget#toolbar_widget {{
            background-color: {t["bg_header"]};
            border-bottom: 1px solid {t["border"]};
        }}
        QPushButton {{
            background-color: {t["bg_secondary"]};
            color: {t["fg_primary"]};
            border: 1px solid {t["border"]};
            border-radius: 3px;
            padding: 3px 8px;
        }}
        QPushButton:hover {{
            background-color: {t["accent"]};
            color: #ffffff;
            border-color: {t["accent_hover"]};
        }}
        /* Verify / Upload Circular Arduino Buttons */
        QPushButton#btn_verify, QPushButton#btn_upload {{
            background-color: {t["accent"]};
            border: none;
            border-radius: 16px;
        }}
        QPushButton#btn_verify:hover, QPushButton#btn_upload:hover {{
            background-color: {t["accent_hover"]};
        }}
        /* Activity Rail */
        QWidget#activity_rail {{
            background-color: {t["bg_activity"]};
            border-right: 1px solid {t["border"]};
        }}
        /* ComboBoxes */
        QComboBox {{
            background-color: {t["bg_secondary"]};
            color: {t["fg_primary"]};
            border: 1px solid {t["border"]};
            border-radius: 14px;
            padding: 4px 10px;
        }}
        QComboBox::drop-down {{
            border: none;
        }}
        /* Tabs */
        QTabBar::tab {{
            background: {t["tab_inactive_bg"]};
            color: {t["fg_secondary"]};
            padding: 6px 14px;
            border: 1px solid {t["border"]};
            border-bottom: none;
        }}
        QTabBar::tab:selected {{
            background: {t["tab_active_bg"]};
            color: {t["fg_active"]};
            font-weight: bold;
            border-top: 2px solid {t["accent"]};
        }}
        /* Terminal & Diagnostics */
        QPlainTextEdit {{
            background-color: {t["terminal_bg"]};
            color: {t["terminal_fg"]};
            border: 1px solid {t["border"]};
        }}
        QListWidget {{
            background-color: {t["bg_secondary"]};
            border: 1px solid {t["border"]};
            color: {t["fg_primary"]};
        }}
        QStatusBar {{
            background-color: {t["bg_secondary"]};
            color: {t["fg_secondary"]};
            border-top: 1px solid {t["border"]};
        }}
        """
        self.setStyleSheet(style)
        self.btn_verify.setObjectName("btn_verify")
        self.btn_upload.setObjectName("btn_upload")
        self.toolbar_widget.setObjectName("toolbar_widget")
        self.activity_rail.setObjectName("activity_rail")

        # Update QScintilla open editors
        for i in range(self.editor_tabs.count()):
            sci = self.editor_tabs.widget(i)
            if isinstance(sci, QsciScintilla):
                self.style_qscintilla(sci, t)

    def style_qscintilla(self, sci, t):
        sci.setPaper(QtGui.QColor(t["bg_editor"]))
        sci.setColor(QtGui.QColor(t["fg_primary"]))
        sci.setCaretForegroundColor(QtGui.QColor(t["caret"]))
        sci.setSelectionBackgroundColor(QtGui.QColor(t["selection_bg"]))

        # Margins & Line numbers
        sci.setMarginsBackgroundColor(QtGui.QColor(t["line_number_bg"]))
        sci.setMarginsForegroundColor(QtGui.QColor(t["line_number_fg"]))
        sci.setMarginLineNumbers(1, True)
        sci.setMarginWidth(1, "00000")

        # Lexer configuration
        lexer = sci.lexer()
        if lexer:
            is_dark = (self.current_theme_key == "dark")
            lexer.setDefaultPaper(QtGui.QColor(t["bg_editor"]))
            lexer.setDefaultColor(QtGui.QColor(t["fg_primary"]))
            lexer.setColor(QtGui.QColor("#89b4fa" if is_dark else "#0000cc"), QsciLexerCPP.Keyword)
            lexer.setColor(QtGui.QColor("#94e2d5" if is_dark else "#008080"), QsciLexerCPP.PreProcessor)
            lexer.setColor(QtGui.QColor("#a6e3a1" if is_dark else "#a31515"), QsciLexerCPP.DoubleQuotedString)
            lexer.setColor(QtGui.QColor("#fab387" if is_dark else "#098658"), QsciLexerCPP.Number)
            lexer.setColor(QtGui.QColor("#6c7086" if is_dark else "#008000"), QsciLexerCPP.CommentLine)
            lexer.setColor(QtGui.QColor("#6c7086" if is_dark else "#008000"), QsciLexerCPP.Comment)

    def toggle_theme(self):
        new_theme = "light" if self.current_theme_key == "dark" else "dark"
        self.apply_theme(new_theme)

    # -------------------------------------------------------------
    # Project & QScintilla Editor Management
    # -------------------------------------------------------------
    def create_qscintilla_editor(self, file_path=None):
        sci = QsciScintilla()
        sci.setUtf8(True)
        sci.setFont(QtGui.QFont("DejaVu Sans Mono" if sys.platform != "win32" else "Consolas", 10))
        sci.setTabWidth(4)
        sci.setIndentationsUseTabs(False)
        sci.setAutoIndent(True)
        sci.setBraceMatching(QsciScintilla.StrictBraceMatch)

        # Attach C/C++ lexer
        lexer = QsciLexerCPP(sci)
        sci.setLexer(lexer)

        self.style_qscintilla(sci, THEMES[self.current_theme_key])

        if file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                sci.setText(f.read())
            sci.file_path = file_path
        else:
            sci.file_path = None

        return sci

    def action_open_project(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open RISC-V Project", self.current_proj_dir
        )
        if folder:
            self.load_project(folder)

    def load_project(self, folder_path):
        if not os.path.isdir(folder_path):
            return
        self.current_proj_dir = folder_path
        self.tree_files.clear()

        files = sorted(glob.glob(os.path.join(folder_path, "*")))
        for f in files:
            if os.path.isfile(f):
                self.tree_files.addItem(os.path.basename(f))

        # Open main.c or primary source in editor tab
        c_files = [f for f in files if f.endswith(".c") or f.endswith(".cpp")]
        if c_files:
            main_f = next((f for f in c_files if os.path.basename(f) == "main.c"), c_files[0])
            self.open_file_in_tab(main_f)

    def on_file_double_clicked(self, item):
        fname = item.text()
        full_path = os.path.join(self.current_proj_dir, fname)
        self.open_file_in_tab(full_path)

    def open_file_in_tab(self, file_path):
        # Check if already open
        for i in range(self.editor_tabs.count()):
            w = self.editor_tabs.widget(i)
            if getattr(w, "file_path", None) == file_path:
                self.editor_tabs.setCurrentIndex(i)
                return

        sci = self.create_qscintilla_editor(file_path)
        tab_name = os.path.basename(file_path)
        idx = self.editor_tabs.addTab(sci, tab_name)
        self.editor_tabs.setCurrentIndex(idx)

    def close_editor_tab(self, index):
        self.editor_tabs.removeTab(index)

    def save_current_file(self):
        idx = self.editor_tabs.currentIndex()
        if idx < 0:
            return
        sci = self.editor_tabs.widget(idx)
        if hasattr(sci, "file_path") and sci.file_path:
            with open(sci.file_path, "w", encoding="utf-8") as f:
                f.write(sci.text())
            self.lbl_status.setText(f"Saved: {os.path.basename(sci.file_path)}")

    # -------------------------------------------------------------
    # Navigation & Drawers
    # -------------------------------------------------------------
    def switch_sidebar(self, index):
        if self.sidebar_stack.currentIndex() == index and self.sidebar_stack.isVisible():
            self.sidebar_stack.hide()
        else:
            self.sidebar_stack.show()
            self.sidebar_stack.setCurrentIndex(index)

    def toggle_bottom_drawer(self):
        if self.bottom_drawer.isVisible():
            self.bottom_drawer.hide()
        else:
            self.bottom_drawer.show()

    def clear_current_terminal(self):
        if self.drawer_stack.currentIndex() == 0:
            self.txt_output.clear()
        else:
            self.txt_terminal.clear()

    # -------------------------------------------------------------
    # Build & Upload Pipeline (Arduino 1-Click Flow)
    # -------------------------------------------------------------
    def action_verify(self):
        self.save_current_file()
        self.bottom_drawer.show()
        self.drawer_stack.setCurrentIndex(0)
        self.txt_output.clear()
        self.lbl_status.setText("Compiling RISC-V firmware...")

        target_name = os.path.basename(os.path.normpath(self.current_proj_dir))
        out_bin = os.path.join(DEFAULT_BINS_DIR, f"{target_name}.bin")
        opt = self.cmb_opt.currentText().split()[0] if hasattr(self, "cmb_opt") else "-Os"

        self.build_thread = BuildWorker(self.current_proj_dir, opt_level=opt, out_bin=out_bin)
        self.build_thread.log_signal.connect(self.append_output_log)
        self.build_thread.done_signal.connect(self.on_build_done)
        self.build_thread.start()

    def on_build_done(self, success, res):
        if success:
            self.current_bin_file = res.get("bin_path", self.current_bin_file)
            size = res.get("total_size", 0)
            self.lbl_status.setText(f"Compilation Successful: {size} bytes")
        else:
            self.lbl_status.setText("Compilation Failed (Check Output)")

    def action_upload(self):
        self.save_current_file()
        self.bottom_drawer.show()
        self.drawer_stack.setCurrentIndex(0)
        self.lbl_status.setText("Compiling before upload...")

        # Stop serial monitor if active on same port
        if self.serial_worker:
            self.serial_worker.stop()
            self.serial_worker = None

        target_name = os.path.basename(os.path.normpath(self.current_proj_dir))
        out_bin = os.path.join(DEFAULT_BINS_DIR, f"{target_name}.bin")
        opt = self.cmb_opt.currentText().split()[0] if hasattr(self, "cmb_opt") else "-Os"

        self.build_thread = BuildWorker(self.current_proj_dir, opt_level=opt, out_bin=out_bin)
        self.build_thread.log_signal.connect(self.append_output_log)
        self.build_thread.done_signal.connect(lambda s, r: self.on_upload_build_done(s, r, out_bin))
        self.build_thread.start()

    def on_upload_build_done(self, success, res, out_bin):
        if not success:
            self.lbl_status.setText("Upload aborted: compilation failed.")
            return

        port = self.cmb_port.currentText()
        if not port:
            self.append_output_log("[ERROR] No serial port selected for upload.\n")
            return

        self.lbl_status.setText(f"Uploading {os.path.basename(out_bin)} to {port}...")
        self.flash_thread = FlashWorker(port, 19200, out_bin)
        self.flash_thread.log_signal.connect(self.append_output_log)
        self.flash_thread.done_signal.connect(self.on_flash_done)
        self.flash_thread.start()

    def on_flash_done(self, success, res):
        if success:
            self.lbl_status.setText("Firmware uploaded and executed.")
            # Switch seamlessly to Serial Monitor
            self.drawer_stack.setCurrentIndex(1)
            self.start_serial_monitor()
        else:
            self.lbl_status.setText("Upload Failed.")

    def append_output_log(self, text):
        self.txt_output.insertPlainText(text)
        if self.chk_autoscroll.isChecked():
            self.txt_output.verticalScrollBar().setValue(
                self.txt_output.verticalScrollBar().maximum()
            )

    # -------------------------------------------------------------
    # Seamless Serial Monitor & Keyboard Streamer
    # -------------------------------------------------------------
    def refresh_ports(self):
        self.cmb_port.clear()
        ports = GEMMEngine.list_ports(filter_virtual=True)
        for p in ports:
            self.cmb_port.addItem(p["device"])

        # Auto-select default port (/dev/ttyUSB1)
        default_p = GEMMEngine.get_default_port()
        idx = self.cmb_port.findText(default_p)
        if idx >= 0:
            self.cmb_port.setCurrentIndex(idx)

    def start_serial_monitor(self):
        port = self.cmb_port.currentText()
        if not port:
            return
        if self.serial_worker:
            self.serial_worker.stop()

        self.serial_worker = SerialMonitorWorker(port, 19200)
        self.serial_worker.rx_signal.connect(self.append_terminal_rx)
        self.serial_worker.start()

    def append_terminal_rx(self, text):
        self.txt_terminal.insertPlainText(text)
        if self.chk_autoscroll.isChecked():
            self.txt_terminal.verticalScrollBar().setValue(
                self.txt_terminal.verticalScrollBar().maximum()
            )

    def send_serial_input(self):
        text = self.txt_serial_input.text()
        if not text:
            return
        self.txt_serial_input.clear()
        if self.serial_worker:
            self.serial_worker.send_data((text + "\r\n").encode("latin-1"))
            self.txt_terminal.insertPlainText(f"> {text}\n")

    def eventFilter(self, obj, event):
        # Keystroke forwarding directly to UART for interactive games/apps (DOOM)
        if obj == self.txt_terminal and event.type() == QtCore.QEvent.KeyPress:
            if self.serial_worker:
                key_text = event.text()
                if key_text:
                    self.serial_worker.send_data(key_text.encode("latin-1"))
                    return True
                elif event.key() == Qt.Key_Return:
                    self.serial_worker.send_data(b"\r\n")
                    return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        if self.serial_worker:
            self.serial_worker.stop()
        event.accept()


# ---------------------------------------------------------------------------
# Main Application Launcher
# ---------------------------------------------------------------------------
def main():
    # Handle --open-console / console command directly
    if "--open-console" in sys.argv or (len(sys.argv) > 1 and sys.argv[1] in ("console", "monitor")):
        import argparse
        p = argparse.ArgumentParser(description="GEMMrvStudio TUI Console")
        p.add_argument("--open-console", action="store_true", help="Open TUI serial console")
        p.add_argument("cmd", nargs="?", default=None)
        p.add_argument("--port", "-p", default=None, help="Serial port")
        p.add_argument("--baud", "-b", "--baude", type=int, default=19200, help="Baud rate")
        args, _ = p.parse_known_args()
        from gemm_tui import launch_console
        launch_console(port=args.port, baud=args.baud)
        return

    # Handle other headless CLI commands if passed
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--gui"):
        from gemm_engine import main as engine_cli
        engine_cli()
        return

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("GEMMrvStudio")
    window = ArduinoGEMMStudio()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
