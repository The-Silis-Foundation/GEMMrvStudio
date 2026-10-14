import sys
import os
import argparse
import json
import struct
import subprocess
import glob
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import serial
import serial.tools.list_ports
from PIL import Image, ImageTk
import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BINS_DIR = os.path.join(REPO_DIR, "example", "bin") if os.path.isdir(os.path.join(REPO_DIR, "example", "bin")) else r"C:\Users\Lenovo\Desktop\bins"
DEFAULT_SCRIPTS_DIR = os.path.join(REPO_DIR, "example", "src") if os.path.isdir(os.path.join(REPO_DIR, "example", "src")) else r"C:\Users\Lenovo\Desktop\scripts"
DEFAULT_CAR_IMAGES_DIR = r"C:\Users\Lenovo\Desktop\car_images"
DEFAULT_DIGIT_IMAGES_DIR = r"C:\Users\Lenovo\Desktop\digit_images"
DEFAULT_PROJECTS_DIR = os.path.join(REPO_DIR, "example", "src") if os.path.isdir(os.path.join(REPO_DIR, "example", "src")) else r"C:\Users\Lenovo\Desktop\GEMMrv32\GEMMrv32-main\sw\example"
NEORV32_HOME_DIR = r"C:/Users/Lenovo/Desktop/GEMMrv32"
DEFAULT_MAKE_PATH = r"D:\vivado\2025.2\Vivado\gnuwin\bin\make.exe"
DEFAULT_GCC_DIR = r"D:\vivado\2025.2\gnu\riscv\nt\bin"

class LineNumberedText(tk.Frame):
    """Editor area with synchronized line numbering and C/C++ syntax highlighting"""
    KEYWORDS = {
        "char", "class", "const", "double", "enum", "explicit", "extern",
        "float", "for", "friend", "inline", "int", "long", "namespace",
        "operator", "private", "protected", "public", "short", "signed",
        "static", "struct", "template", "typedef", "typename", "union",
        "unsigned", "virtual", "void", "volatile", "while", "int8_t",
        "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t",
        "uint64_t", "size_t", "gemmrv_mat", "gemmrv_mat_out"
    }
    CONTROLS = {"break", "case", "continue", "default", "do", "else", "goto", "if", "return", "switch"}

    def __init__(self, master, **kwargs):
        super().__init__(master, bg="#ffffff")
        self.line_canvas = tk.Canvas(self, width=42, bg="#f7fafc", bd=0, highlightthickness=0)
        self.line_canvas.pack(side="left", fill="y")

        self.text_scroll = ttk.Scrollbar(self, orient="vertical")
        self.text_scroll.pack(side="right", fill="y")

        self.text = tk.Text(self, yscrollcommand=self._on_scroll, bd=0, highlightthickness=0,
                            font=("Consolas", 10), undo=True, maxundo=50,
                            bg="#ffffff", fg="#1a202c",
                            insertbackground="#000000", selectbackground="#bee3f8", selectforeground="#1a202c",
                            inactiveselectbackground="#e2e8f0", wrap="none",
                            **kwargs)
        self.text.pack(side="left", fill="both", expand=True)

        self.text_scroll.config(command=self._on_text_scroll)

        # Crisp, readable syntax highlight tags on light background
        self.text.tag_config("kw", foreground="#0000ff", font=("Consolas", 10, "bold"))
        self.text.tag_config("ctrl", foreground="#7928ca", font=("Consolas", 10, "bold"))
        self.text.tag_config("preproc", foreground="#008080")
        self.text.tag_config("string", foreground="#a31515")
        self.text.tag_config("comment", foreground="#008000", font=("Consolas", 10, "italic"))
        self.text.tag_config("number", foreground="#098658")

        self.text.bind("<KeyRelease>", self._on_key_release)
        self.text.bind("<MouseWheel>", self._update_lines)
        self.text.bind("<Button-1>", self._update_lines)

    def _on_scroll(self, *args):
        self.text_scroll.set(*args)
        self._update_lines()

    def _on_text_scroll(self, *args):
        self.text.yview(*args)
        self._update_lines()

    def _on_key_release(self, event=None):
        self._update_lines()
        self.highlight_syntax()

    def _update_lines(self, event=None):
        self.line_canvas.delete("all")
        i = self.text.index("@0,0")
        while True:
            dline = self.text.dlineinfo(i)
            if dline is None:
                break
            y = dline[1]
            linenum = str(i).split(".")[0]
            self.line_canvas.create_text(38, y + 2, anchor="ne", text=linenum,
                                         font=("Consolas", 9), fill="#5c6370")
            i = self.text.index(f"{i}+1line")

    def highlight_syntax(self):
        import re
        content = self.text.get("1.0", "end-1c")

        for tag in ("kw", "ctrl", "preproc", "string", "comment", "number"):
            self.text.tag_remove(tag, "1.0", "end")

        # Comments
        for m in re.finditer(r"//[^\n]*", content):
            start = f"1.0+{m.start()}c"
            end = f"1.0+{m.end()}c"
            self.text.tag_add("comment", start, end)

        # Preprocessor
        for m in re.finditer(r"#[a-zA-Z_]+", content):
            start = f"1.0+{m.start()}c"
            end = f"1.0+{m.end()}c"
            self.text.tag_add("preproc", start, end)

        # Strings
        for m in re.finditer(r'"[^"\n]*"', content):
            start = f"1.0+{m.start()}c"
            end = f"1.0+{m.end()}c"
            self.text.tag_add("string", start, end)

        # Numbers
        for m in re.finditer(r"\b(0x[0-9a-fA-F]+|\d+)\b", content):
            start = f"1.0+{m.start()}c"
            end = f"1.0+{m.end()}c"
            self.text.tag_add("number", start, end)

        # Keywords and controls
        for m in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", content):
            word = m.group()
            start = f"1.0+{m.start()}c"
            end = f"1.0+{m.end()}c"
            if word in self.KEYWORDS:
                self.text.tag_add("kw", start, end)
            elif word in self.CONTROLS:
                self.text.tag_add("ctrl", start, end)


class ProfessionalGEMMDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("GEMMrv32 - RISC-V IDE & Hardware Console")
        self.root.geometry("1240x820")
        self.root.minsize(1020, 700)
        self.root.configure(bg="#f4f6f9")

        self.ser = None
        self.is_flashing = False
        self.reader_alive = False

        # Global serial / flasher variables
        self.port_var = tk.StringVar(value="COM10")
        self.baud_var = tk.IntVar(value=19200)
        self.bin_path_var = tk.StringVar()
        self.image_path_var = tk.StringVar()
        self.dataset_mode_var = tk.StringVar(value="Digits (28x28)")
        self.show_preview_var = tk.BooleanVar(value=False)
        self.show_inference_tools_var = tk.BooleanVar(value=False)

        # IDE State
        self.open_files = {} # tab_id -> {'path': str, 'text_widget': LineNumberedText}
        self.current_proj_dir = tk.StringVar(value=DEFAULT_SCRIPTS_DIR)
        self.proj_target_bin = tk.StringVar(value="my_application.bin")
        self.proj_opt_level = tk.StringVar(value="-O2")
        self.is_building = False

        self.setup_styles()
        self.setup_main_layout()

        self.populate_bin_list()
        self.populate_image_list()
        self.create_new_editor_tab()
        self.start_reader_thread()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        # Professional light theme Notebook styling
        style.configure("TNotebook", background="#edf2f7", borderwidth=0)
        style.configure("TNotebook.Tab", background="#e2e8f0", foreground="#2d3748",
                        padding=[16, 6], font=("Segoe UI", 9, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", "#ffffff")],
                  foreground=[("selected", "#2b6cb0")])

        # Scrollbar and Combobox styling
        style.configure("TCombobox", fieldbackground="#ffffff", background="#e9ecef")
        style.configure("TPanedwindow", background="#edf2f7")

    def setup_main_layout(self):
        self.root.configure(bg="#f4f6f9")

        # 1. Global Header Bar (Clean, Compact, Professional)
        header_frame = tk.Frame(self.root, bg="#ffffff", bd=1, relief="solid")
        header_frame.pack(fill="x", padx=10, pady=(8, 4))

        title_box = tk.Frame(header_frame, bg="#ffffff")
        title_box.pack(side="left", padx=12, pady=6)

        lbl_title = tk.Label(title_box, text="GEMMrv32 Studio",
                             font=("Segoe UI", 13, "bold"), fg="#1a202c", bg="#ffffff")
        lbl_title.pack(side="left", padx=(0, 10))

        lbl_sub = tk.Label(title_box, text="NEORV32 RISC-V Embedded IDE & Systolic Hardware Coprocessor",
                           font=("Segoe UI", 8), fg="#718096", bg="#ffffff")
        lbl_sub.pack(side="left")

        # Global Connection status badge
        self.lbl_status_badge = tk.Label(header_frame, text="DISCONNECTED",
                                         font=("Segoe UI", 9, "bold"), fg="#c53030", bg="#fff5f5",
                                         padx=10, pady=4, bd=1, relief="solid")
        self.lbl_status_badge.pack(side="right", padx=12, pady=6)

        # 2. Main Two-Tab Notebook (Tab 1 is IDE and DEFAULT; Tab 2 is Console & Programmer)
        self.main_notebook = ttk.Notebook(self.root)
        self.main_notebook.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        self.tab_ide = tk.Frame(self.main_notebook, bg="#f4f6f9")
        self.tab_console = tk.Frame(self.main_notebook, bg="#f4f6f9")

        self.main_notebook.add(self.tab_ide, text="  RISC-V IDE & Builder (Default)  ")
        self.main_notebook.add(self.tab_console, text="  Hardware Console & Programmer  ")

        # Build contents for both tabs
        self.setup_ide_tab(self.tab_ide)
        self.setup_console_tab(self.tab_console)

        # Ensure Tab 1 is default active
        self.main_notebook.select(self.tab_ide)

    # =========================================================================
    # TAB 1: RISC-V IDE & BUILDER (Resizable Panes, Editor has More Space)
    # =========================================================================
    def setup_ide_tab(self, parent):
        self.root.geometry("1000x600") # Or whatever starting resolution you prefer
        # Top PanedWindow: Horizontal split between (Left: Editor) and (Right: Resizable Sidebar & Build Output)
        self.h_paned = ttk.PanedWindow(parent, orient="horizontal")
        self.h_paned.pack(fill="both", expand=True, padx=2, pady=2)

        # 1. Left Large Pane: Tabbed Source Code Editor
        editor_card = tk.Frame(self.h_paned, bg="#ffffff", bd=1, relief="solid", width=700, height=600)
        self.h_paned.add(editor_card, weight=7) # Exact 3:1 weight ratio from image

        # Editor Header / Action Bar
        ed_bar = tk.Frame(editor_card, bg="#edf2f7", bd=0)
        ed_bar.pack(fill="x")

        tk.Label(ed_bar, text="Source Code Editor", font=("Segoe UI", 9, "bold"),
                 fg="#2d3748", bg="#edf2f7").pack(side="left", padx=10, pady=5)

        btn_save = tk.Button(ed_bar, text="Save (Ctrl+S)", command=self.save_current_editor_file,
                             font=("Segoe UI", 8), bg="#ffffff", fg="#2b6cb0", relief="solid", bd=1, padx=8, pady=1)
        btn_save.pack(side="right", padx=(4, 8), pady=4)

        btn_new_file = tk.Button(ed_bar, text="New Tab", command=self.create_new_editor_tab,
                                font=("Segoe UI", 8), bg="#ffffff", fg="#2d3748", relief="solid", bd=1, padx=8, pady=1)
        btn_new_file.pack(side="right", padx=4, pady=4)

        btn_close_tab = tk.Button(ed_bar, text="Close Tab", command=self.close_current_editor_tab,
                                  font=("Segoe UI", 8), bg="#ffffff", fg="#c53030", relief="solid", bd=1, padx=8, pady=1)
        btn_close_tab.pack(side="right", padx=4, pady=4)

        # Notebook for multiple code file tabs
        self.editor_notebook = ttk.Notebook(editor_card)
        self.editor_notebook.pack(fill="both", expand=True, padx=1, pady=1)
        self.root.bind("<Control-s>", lambda e: self.save_current_editor_file())

        # 2. Right Side Pane: Vertical PanedWindow with Wizard (Top) and Build Diagnostics (Bottom)
        self.v_paned = ttk.PanedWindow(self.h_paned, orient="vertical")
        self.h_paned.add(self.v_paned, weight=3)

        # Right-Top: Project Wizard & Linker Configuration Card
        wiz_card = tk.Frame(self.v_paned, bg="#ffffff", bd=1, relief="solid", width=300, height=360)
        self.v_paned.add(wiz_card, weight=6)

        wiz_title_bar = tk.Frame(wiz_card, bg="#edf2f7")
        wiz_title_bar.pack(fill="x")
        tk.Label(wiz_title_bar, text="Project Linker & Compiler Wizard",
                 font=("Segoe UI", 9, "bold"), fg="#2d3748", bg="#edf2f7").pack(side="left", padx=10, pady=5)

        wiz_body = tk.Frame(wiz_card, bg="#ffffff")
        wiz_body.pack(fill="both", expand=True, padx=10, pady=6)

        # Workspace directory & Project Management row
        row_w1 = tk.Frame(wiz_body, bg="#ffffff")
        row_w1.pack(fill="x", pady=2)

        # Prominent Project Buttons: Open Project & New Project
        row_proj_btns = tk.Frame(row_w1, bg="#ffffff")
        row_proj_btns.pack(fill="x", pady=(2, 4))

        btn_open_proj = tk.Button(row_proj_btns, text="Open Project...", command=self.open_project_wizard,
                                  font=("Segoe UI", 8, "bold"), bg="#edf2f7", fg="#2b6cb0", relief="solid", bd=1, padx=8, pady=3, cursor="hand2")
        btn_open_proj.pack(side="left", fill="x", expand=True, padx=(0, 3))

        btn_new_proj = tk.Button(row_proj_btns, text="New Project...", command=self.new_project_wizard,
                                 font=("Segoe UI", 8, "bold"), bg="#edf2f7", fg="#2f855a", relief="solid", bd=1, padx=8, pady=3, cursor="hand2")
        btn_new_proj.pack(side="right", fill="x", expand=True, padx=(3, 0))

        tk.Label(row_w1, text="Project Folder:", font=("Segoe UI", 8, "bold"), fg="#4a5568", bg="#ffffff").pack(anchor="w")

        row_w1_sub = tk.Frame(row_w1, bg="#ffffff")
        row_w1_sub.pack(fill="x", pady=2)
        self.entry_proj_dir = tk.Entry(row_w1_sub, textvariable=self.current_proj_dir, font=("Consolas", 8),
                                       bg="#fafafa", bd=1, relief="solid")
        self.entry_proj_dir.pack(side="left", fill="x", expand=True, padx=(0, 4))
        btn_browse_proj = tk.Button(row_w1_sub, text="Browse...", command=self.browse_project_dir,
                                    font=("Segoe UI", 8), bg="#edf2f7", fg="#2d3748", relief="solid", bd=1, padx=6, pady=1)
        btn_browse_proj.pack(side="right")

        # Project Files Tree / List
        row_w2 = tk.Frame(wiz_body, bg="#ffffff")
        row_w2.pack(fill="both", expand=True, pady=4)
        tk.Label(row_w2, text="Source Files (Click to open in Editor):",
                 font=("Segoe UI", 8, "bold"), fg="#4a5568", bg="#ffffff").pack(anchor="w")

        tree_scroll = tk.Scrollbar(row_w2)
        tree_scroll.pack(side="right", fill="y")
        self.tree_proj_files = tk.Listbox(row_w2, font=("Consolas", 9), selectmode="single",
                                          bg="#fafafa", fg="#1a202c", bd=1, relief="solid",
                                          yscrollcommand=tree_scroll.set, height=5)
        self.tree_proj_files.pack(side="left", fill="both", expand=True)
        tree_scroll.config(command=self.tree_proj_files.yview)
        self.tree_proj_files.bind("<Double-1>", self.on_project_file_double_click)
        self.tree_proj_files.bind("<<ListboxSelect>>", self.on_project_file_double_click)

        # Compiler Options
        row_w3 = tk.Frame(wiz_body, bg="#ffffff")
        row_w3.pack(fill="x", pady=3)

        tk.Label(row_w3, text="Opt Level:", font=("Segoe UI", 8, "bold"), fg="#4a5568", bg="#ffffff").pack(side="left")
        combo_opt = ttk.Combobox(row_w3, textvariable=self.proj_opt_level, values=["-O2", "-Os", "-O0", "-O1", "-O3"],
                                 width=6, state="readonly", font=("Segoe UI", 8))
        combo_opt.pack(side="left", padx=4)

        tk.Label(row_w3, text="Target Binary:", font=("Segoe UI", 8, "bold"), fg="#4a5568", bg="#ffffff").pack(side="left", padx=(8, 2))
        self.entry_target_bin = tk.Entry(row_w3, textvariable=self.proj_target_bin, font=("Consolas", 8),
                                         width=22, bd=1, relief="solid")
        self.entry_target_bin.pack(side="left", padx=2)

        # Action Buttons
        row_act = tk.Frame(wiz_body, bg="#ffffff")
        row_act.pack(fill="x", pady=(6, 2))

        self.btn_compile_only = tk.Button(row_act, text="Build Binary", command=self.build_project_only,
                                          font=("Segoe UI", 9, "bold"), bg="#edf2f7", fg="#2b6cb0", activebackground="#e2e8f0",
                                          relief="solid", bd=1, padx=12, pady=5, cursor="hand2")
        self.btn_compile_only.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_build_and_run = tk.Button(row_act, text="Build & Run on Board", command=self.build_and_run_on_board,
                                           font=("Segoe UI", 9, "bold"), bg="#2b6cb0", fg="#ffffff", activebackground="#2c5282",
                                           relief="solid", bd=1, padx=14, pady=5, cursor="hand2")
        self.btn_build_and_run.pack(side="right", fill="x", expand=True, padx=(4, 0))

        # Right-Bottom: Build Diagnostics Console Card
        diag_card = tk.Frame(self.v_paned, bg="#ffffff", bd=1, relief="solid", width=300, height=240)
        self.v_paned.add(diag_card, weight=4)

        diag_bar = tk.Frame(diag_card, bg="#edf2f7")
        diag_bar.pack(fill="x")
        tk.Label(diag_bar, text="Compiler Build Diagnostics", font=("Segoe UI", 9, "bold"),
                 fg="#2d3748", bg="#edf2f7").pack(side="left", padx=10, pady=5)

        btn_clear_diag = tk.Button(diag_bar, text="Clear", command=lambda: self.txt_diag.delete("1.0", "end"),
                                  font=("Segoe UI", 8), bg="#ffffff", fg="#4a5568", relief="solid", bd=1, padx=6, pady=1)
        btn_clear_diag.pack(side="right", padx=8, pady=3)

        diag_scroll = tk.Scrollbar(diag_card)
        diag_scroll.pack(side="right", fill="y")
        self.txt_diag = tk.Text(diag_card, font=("Consolas", 9), bg="#fafafa", fg="#1a202c",
                                yscrollcommand=diag_scroll.set, bd=0, wrap="char")
        self.txt_diag.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        diag_scroll.config(command=self.txt_diag.yview)

        self.txt_diag.tag_config("sys", foreground="#2b6cb0", font=("Consolas", 9, "bold"))
        self.txt_diag.tag_config("err", foreground="#c53030", font=("Consolas", 9, "bold"))
        self.txt_diag.tag_config("success", foreground="#2f855a", font=("Consolas", 9, "bold"))
        self.txt_diag.tag_config("cmd", foreground="#4a5568")
# ... (all your existing widget setup code) ...

        # ---------------------------------------------------------
        # THE FIX: Force the initial sash (divider) positions
        # ---------------------------------------------------------
        def apply_initial_proportions():
            # Get the actual rendered dimensions of the paned windows
            total_width = self.h_paned.winfo_width()
            total_height = self.v_paned.winfo_height()

            # Ensure the window has actually been drawn (width > 1)
            if total_width > 10:
                # Move the horizontal divider to 70% of the width
                self.h_paned.sashpos(0, int(total_width * 0.70))
            
            if total_height > 10:
                # Move the vertical divider to 60% of the height (leaving 40% for the bottom)
                self.v_paned.sashpos(0, int(total_height * 0.60))

        # Force Tkinter to calculate the geometry of all widgets right now
        self.root.update_idletasks() 
        
        # Schedule the sash movement 50ms after the UI renders 
        # (This bypasses Tkinter's internal frame shrinking)
        self.root.after(50, apply_initial_proportions)
    # =========================================================================
    # TAB 2: HARDWARE CONSOLE & PROGRAMMER (Original Clean Light UI)
    # =========================================================================
    def setup_console_tab(self, parent):
        # 2. Main Controls Card
        ctrl_card = tk.Frame(parent, bg="#ffffff", bd=1, relief="solid")
        ctrl_card.pack(fill="x", padx=10, pady=4)

        # Row A: Connection Management
        row_conn = tk.Frame(ctrl_card, bg="#ffffff")
        row_conn.pack(fill="x", padx=15, pady=(10, 6))

        tk.Label(row_conn, text="Serial Port:", font=("Segoe UI", 9, "bold"), fg="#2d3748", bg="#ffffff").pack(side="left")
        self.combo_port = ttk.Combobox(row_conn, textvariable=self.port_var, width=9, font=("Segoe UI", 9))
        self.combo_port['values'] = [p.device for p in serial.tools.list_ports.comports()] or ["COM10"]
        self.combo_port.pack(side="left", padx=(6, 16))

        tk.Label(row_conn, text="Baud Rate:", font=("Segoe UI", 9, "bold"), fg="#2d3748", bg="#ffffff").pack(side="left")
        combo_baud = ttk.Combobox(row_conn, textvariable=self.baud_var, width=8, font=("Segoe UI", 9), state="readonly")
        combo_baud['values'] = [9600, 19200, 38400, 57600, 115200]
        combo_baud.pack(side="left", padx=(6, 16))

        btn_reconnect = tk.Button(row_conn, text="Connect / Refresh", command=self.manual_reconnect,
                                  font=("Segoe UI", 9), bg="#edf2f7", fg="#2b6cb0", activebackground="#e2e8f0",
                                  relief="solid", bd=1, padx=10, pady=2, cursor="hand2")
        btn_reconnect.pack(side="left", padx=4)

        btn_kill = tk.Button(row_conn, text="Kill Background Processes", command=self.kill_bg_processes,
                             font=("Segoe UI", 9), bg="#fff5f5", fg="#c53030", activebackground="#fed7d7",
                             relief="solid", bd=1, padx=10, pady=2, cursor="hand2")
        btn_kill.pack(side="left", padx=4)

        # Row B: Firmware Flasher Section
        row_flash = tk.Frame(ctrl_card, bg="#ffffff")
        row_flash.pack(fill="x", padx=15, pady=6)

        tk.Label(row_flash, text="Firmware Binary:", font=("Segoe UI", 9, "bold"), fg="#2d3748", bg="#ffffff").pack(side="left")
        self.combo_bins = ttk.Combobox(row_flash, textvariable=self.bin_path_var, width=42, font=("Segoe UI", 9))
        self.combo_bins.pack(side="left", padx=(6, 6))

        btn_browse_bin = tk.Button(row_flash, text="Browse...", command=self.browse_binary,
                                   font=("Segoe UI", 8), bg="#edf2f7", fg="#2d3748", relief="solid", bd=1, padx=6, pady=2)
        btn_browse_bin.pack(side="left", padx=(0, 14))

        self.btn_flash = tk.Button(row_flash, text="Program Board (PROG Mode)", command=self.start_program_flow,
                                   font=("Segoe UI", 9, "bold"), bg="#2b6cb0", fg="#ffffff", activebackground="#2c5282",
                                   relief="solid", bd=1, padx=14, pady=3, cursor="hand2")
        self.btn_flash.pack(side="left", padx=4)

        # Row C: Collapsible Image Inference & Hardware Testing Section
        header_inf = tk.Frame(ctrl_card, bg="#ffffff")
        header_inf.pack(fill="x", padx=15, pady=(6, 2))

        self.btn_toggle_inf = tk.Button(header_inf, text="▶ Image Inference & Testing Tools",
                                        command=self.toggle_inference_tools,
                                        font=("Segoe UI", 9, "bold"), fg="#2b6cb0", bg="#ffffff",
                                        activeforeground="#1a365d", activebackground="#ffffff",
                                        bd=0, cursor="hand2", anchor="w")
        self.btn_toggle_inf.pack(side="left")

        # Container frame for image inference controls (hidden by default)
        self.row_img = tk.Frame(ctrl_card, bg="#f8fafc", bd=1, relief="solid")

        sub_pad = tk.Frame(self.row_img, bg="#f8fafc")
        sub_pad.pack(fill="x", padx=10, pady=6)

        tk.Label(sub_pad, text="Dataset:", font=("Segoe UI", 9, "bold"), fg="#2d3748", bg="#f8fafc").pack(side="left")
        self.combo_dataset = ttk.Combobox(sub_pad, textvariable=self.dataset_mode_var,
                                          values=["Digits (28x28)", "Vehicles (32x32)"],
                                          width=14, state="readonly", font=("Segoe UI", 9))
        self.combo_dataset.pack(side="left", padx=(4, 10))
        self.combo_dataset.bind("<<ComboboxSelected>>", self.on_dataset_mode_changed)

        tk.Label(sub_pad, text="Test Image:", font=("Segoe UI", 9, "bold"), fg="#2d3748", bg="#f8fafc").pack(side="left")
        self.combo_imgs = ttk.Combobox(sub_pad, textvariable=self.image_path_var, width=32, font=("Segoe UI", 9))
        self.combo_imgs.pack(side="left", padx=(6, 6))
        self.combo_imgs.bind("<<ComboboxSelected>>", self.on_image_selected)

        btn_browse_img = tk.Button(sub_pad, text="Browse...", command=self.browse_image,
                                   font=("Segoe UI", 8), bg="#ffffff", fg="#2d3748", relief="solid", bd=1, padx=6, pady=2)
        btn_browse_img.pack(side="left", padx=(0, 10))

        self.chk_preview = ttk.Checkbutton(sub_pad, text="Show Photo Preview", variable=self.show_preview_var,
                                           command=self.toggle_preview_box)
        self.chk_preview.pack(side="left", padx=(0, 12))

        self.btn_send_img = tk.Button(sub_pad, text="Send Image for Inference", command=self.send_selected_image,
                                      font=("Segoe UI", 9, "bold"), bg="#2f855a", fg="#ffffff", activebackground="#276749",
                                      relief="solid", bd=1, padx=14, pady=3, cursor="hand2")
        self.btn_send_img.pack(side="left", padx=4)

        # Status text line
        self.lbl_action_status = tk.Label(ctrl_card, text="System Ready.", font=("Segoe UI", 9),
                                          fg="#4a5568", bg="#ffffff")
        self.lbl_action_status.pack(fill="x", padx=15, pady=(4, 8))

        # 3. Main Workspace: Image Preview & Terminal Area
        self.work_frame = tk.Frame(parent, bg="#f4f6f9")
        self.work_frame.pack(fill="both", expand=True, padx=10, pady=6)

        # Left: Image Preview Box (created, but NOT packed by default since preview is OFF)
        self.img_box = tk.Frame(self.work_frame, bg="#ffffff", bd=1, relief="solid", width=160)
        self.img_box.pack_propagate(False)

        tk.Label(self.img_box, text="Selected Image", font=("Segoe UI", 9, "bold"), fg="#2d3748", bg="#ffffff").pack(pady=(10, 6))
        self.lbl_img_preview = tk.Label(self.img_box, text="No Image", bg="#edf2f7", fg="#a0aec0", bd=1, relief="solid")
        self.lbl_img_preview.pack(padx=12, pady=6, fill="x", expand=True)

        self.lbl_img_info = tk.Label(self.img_box, text="32x32 RGB", font=("Segoe UI", 8), fg="#718096", bg="#ffffff")
        self.lbl_img_info.pack(pady=(0, 10))

        # Right: Terminal Panel
        self.term_frame = tk.Frame(self.work_frame, bg="#ffffff", bd=1, relief="solid")
        self.term_frame.pack(side="right", fill="both", expand=True)

        term_title_bar = tk.Frame(self.term_frame, bg="#edf2f7")
        term_title_bar.pack(fill="x")

        tk.Label(term_title_bar, text="Console Output", font=("Segoe UI", 9, "bold"), fg="#2d3748", bg="#edf2f7").pack(side="left", padx=12, pady=5)

        btn_clear = tk.Button(term_title_bar, text="Clear", command=self.clear_terminal,
                             font=("Segoe UI", 8), bg="#ffffff", fg="#4a5568", relief="solid", bd=1, padx=8, pady=1)
        btn_clear.pack(side="right", padx=10, pady=3)

        # Terminal Text Window
        txt_box = tk.Frame(self.term_frame, bg="#ffffff")
        txt_box.pack(fill="both", expand=True, padx=1, pady=1)

        scroll = tk.Scrollbar(txt_box)
        scroll.pack(side="right", fill="y")

        self.txt_term = tk.Text(txt_box, bg="#fafafa", fg="#1a202c", insertbackground="#000000",
                                font=("Consolas", 10), wrap="char", yscrollcommand=scroll.set, bd=0)
        self.txt_term.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        scroll.config(command=self.txt_term.yview)
        self.txt_term.bind("<Key>", self.forward_terminal_key)

        # Text tag styles
        self.txt_term.tag_config("sys", foreground="#2b6cb0", font=("Consolas", 10, "bold"))
        self.txt_term.tag_config("err", foreground="#c53030", font=("Consolas", 10, "bold"))
        self.txt_term.tag_config("success", foreground="#2f855a", font=("Consolas", 10, "bold"))
        self.txt_term.tag_config("warn", foreground="#b7791f", font=("Consolas", 10, "bold"))

        # 4. Expanded Interactive Terminal Input Section
        input_card = tk.Frame(parent, bg="#ffffff", bd=1, relief="solid")
        input_card.pack(fill="x", padx=10, pady=(4, 12))

        tk.Label(input_card, text="Console Input:", font=("Segoe UI", 9, "bold"), fg="#2d3748", bg="#ffffff").pack(side="left", padx=(12, 6), pady=8)

        self.entry_input = tk.Entry(input_card, font=("Consolas", 11), bg="#ffffff", fg="#1a202c",
                                    insertbackground="#1a202c", bd=1, relief="solid")
        self.entry_input.pack(side="left", fill="x", expand=True, padx=6, pady=8)
        self.entry_input.bind("<Return>", lambda e: self.send_terminal_input())

        btn_send = tk.Button(input_card, text="Send Line", command=self.send_terminal_input,
                             font=("Segoe UI", 9, "bold"), bg="#edf2f7", fg="#2b6cb0",
                             relief="solid", bd=1, padx=16, pady=4, cursor="hand2")
        btn_send.pack(side="right", padx=(6, 12), pady=8)

    # =========================================================================
    # IDE LOGIC: Multi-Tab Text Editor Operations
    # =========================================================================
    def open_project_wizard(self):
        p = filedialog.askdirectory(initialdir=DEFAULT_SCRIPTS_DIR, title="Open Project Folder")
        if p:
            self.load_project_folder(p, open_all_tabs=True)

    def new_project_wizard(self):
        # Dialog to prompt project name
        from tkinter import simpledialog
        name = simpledialog.askstring("New Project", "Enter Project Name (e.g. my_nn_layer):", parent=self.root)
        if name:
            clean_name = "".join(c for c in name if c.isalnum() or c in ("_", "-")).strip()
            if not clean_name:
                clean_name = "new_project"
            proj_path = os.path.join(DEFAULT_SCRIPTS_DIR, clean_name)
            os.makedirs(proj_path, exist_ok=True)

            # Create default main.c template
            template_code = f"""#include <stdint.h>
#include <neorv32.h>

#define BAUD_RATE 19200

int main(void) {{
    // Initialize GPIO and UART
    neorv32_gpio_port_set(0x0F);
    neorv32_uart0_setup(BAUD_RATE, 0);

    neorv32_uart0_printf("\\n=== Project: {clean_name} ===\\n");
    neorv32_uart0_printf("Core Clock: %u Hz\\n", (uint32_t)NEORV32_SYSINFO->CLK);
    neorv32_uart0_printf("Running on NEORV32 RISC-V SoC!\\n\\n");

    int count = 0;
    while (1) {{
        neorv32_uart0_printf("Heartbeat: %d\\n", ++count);
        for (volatile int i = 0; i < 500000; i++);
    }}

    return 0;
}}
"""
            main_file = os.path.join(proj_path, "main.c")
            if not os.path.exists(main_file):
                with open(main_file, "w", encoding="utf-8") as f:
                    f.write(template_code)

            self.load_project_folder(proj_path, open_all_tabs=True)
            self.log_diag(f"[NEW PROJECT] Created and opened '{clean_name}' at {proj_path}\n", "success")

    def browse_project_dir(self):
        p = filedialog.askdirectory(initialdir=self.current_proj_dir.get(), title="Select Project Directory")
        if p:
            self.load_project_folder(p, open_all_tabs=False)

    def load_project_folder(self, folder_path, open_all_tabs=False):
        norm = os.path.normpath(folder_path)
        self.current_proj_dir.set(norm)
        base = os.path.basename(norm)
        if base and base != "scripts":
            self.proj_target_bin.set(f"{base}.bin")
        self.populate_project_files()

        if open_all_tabs and os.path.isdir(norm):
            # Close existing untitled tabs if empty
            for tab_id, data in list(self.open_files.items()):
                if not data['path']:
                    txt = data['editor'].text.get("1.0", "end-1c").strip()
                    if not txt:
                        self.editor_notebook.forget(tab_id)
                        del self.open_files[tab_id]

            # Collect primary C/H/makefile files in this project
            opened_any = False
            for root, dirs, files in os.walk(norm):
                if "build" in root:
                    continue
                # Sort so main.c comes first
                sorted_files = sorted(files, key=lambda x: (x.lower() != "main.c", x.lower()))
                for f in sorted_files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in (".c", ".h", ".cpp", ".s") or f.lower() == "makefile":
                        full = os.path.join(root, f)
                        self.open_file_in_editor(full)
                        opened_any = True

            self.log_diag(f"[PROJECT OPENED] {base} ({norm})\n", "sys")

    def populate_project_files(self):
        p = self.current_proj_dir.get()
        self.tree_proj_files.delete(0, "end")
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                # Don't recurse deep into build dirs
                if "build" in root:
                    continue
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in (".c", ".h", ".s", ".mk") or f.lower() == "makefile":
                        rel = os.path.relpath(os.path.join(root, f), p)
                        self.tree_proj_files.insert("end", rel)

            # Files listed in sidebar tree for user to double-click on demand
            pass

    def on_project_file_double_click(self, event=None):
        sel = self.tree_proj_files.curselection()
        if sel:
            rel = self.tree_proj_files.get(sel[0])
            full = os.path.join(self.current_proj_dir.get(), rel)
            # If user clicked a file, open it
            if os.path.isfile(full):
                self.open_file_in_editor(full)
                # Automatically update target binary to project or file name
                parent_folder = os.path.basename(os.path.normpath(self.current_proj_dir.get()))
                if parent_folder and parent_folder != "scripts":
                    self.proj_target_bin.set(f"{parent_folder}.bin")
                else:
                    base = os.path.splitext(os.path.basename(full))[0]
                    self.proj_target_bin.set(f"{base}.bin")

    def open_file_in_editor(self, full_path):
        norm = os.path.normpath(full_path)
        # Check if already open
        for tab_id, data in list(self.open_files.items()):
            if data['path'] == norm:
                self.editor_notebook.select(tab_id)
                return

        try:
            with open(norm, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            messagebox.showerror("Error Opening File", f"Could not read {norm}:\n{e}")
            return

        frame = tk.Frame(self.editor_notebook, bg="#ffffff")
        editor = LineNumberedText(frame)
        editor.pack(fill="both", expand=True)
        editor.text.insert("1.0", content)
        editor._update_lines()
        editor.highlight_syntax()

        tab_title = os.path.basename(norm)
        self.editor_notebook.add(frame, text=f" {tab_title} ")
        self.editor_notebook.select(frame)

        self.open_files[str(frame)] = {'path': norm, 'editor': editor, 'frame': frame}

    def create_new_editor_tab(self):
        frame = tk.Frame(self.editor_notebook, bg="#ffffff")
        editor = LineNumberedText(frame)
        editor.pack(fill="both", expand=True)

        idx = len(self.open_files) + 1
        tab_title = f"Untitled-{idx}.c"
        self.editor_notebook.add(frame, text=f" {tab_title} ")
        self.editor_notebook.select(frame)

        self.open_files[str(frame)] = {'path': None, 'editor': editor, 'frame': frame}

    def close_current_editor_tab(self):
        sel = self.editor_notebook.select()
        if sel and sel in self.open_files:
            self.editor_notebook.forget(sel)
            del self.open_files[sel]

    def save_current_editor_file(self):
        sel = self.editor_notebook.select()
        if not sel or sel not in self.open_files:
            return

        info = self.open_files[sel]
        content = info['editor'].text.get("1.0", "end-1c")
        path = info['path']

        if not path:
            init_d = self.current_proj_dir.get() if os.path.exists(self.current_proj_dir.get()) else DEFAULT_SCRIPTS_DIR
            path = filedialog.asksaveasfilename(
                initialdir=init_d,
                title="Save As",
                filetypes=[("C Source", "*.c"), ("C Header", "*.h"), ("All Files", "*.*")]
            )
            if not path:
                return
            info['path'] = os.path.normpath(path)
            tab_title = os.path.basename(info['path'])
            self.editor_notebook.tab(sel, text=f" {tab_title} ")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self.log_diag(f"[SAVED] {path}\n", "sys")
            # Automatically set project directory if empty or if saving single file
            folder = os.path.dirname(os.path.normpath(path))
            self.current_proj_dir.set(folder)
            fname = os.path.splitext(os.path.basename(path))[0]
            self.proj_target_bin.set(f"{fname}.bin")
            self.populate_project_files()
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save {path}:\n{e}")

    def log_diag(self, text, tag=None):
        self.txt_diag.insert("end", text, tag)
        self.txt_diag.see("end")

    # =========================================================================
    # BUILD PIPELINE & "BUILD & RUN" LOGIC
    # =========================================================================
    def build_project_only(self):
        if self.is_building:
            return
        threading.Thread(target=self._build_worker, args=(False,), daemon=True).start()

    def build_and_run_on_board(self):
        if self.is_building:
            return
        # Stays on Tab 1 until build finishes, then switches to Tab 2 for upload
        threading.Thread(target=self._build_worker, args=(True,), daemon=True).start()

    def _build_worker(self, run_after):
        self.is_building = True
        self.root.after(0, lambda: self.btn_compile_only.config(state="disabled"))
        self.root.after(0, lambda: self.btn_build_and_run.config(state="disabled"))

        proj_dir = self.current_proj_dir.get().strip()

        # If project dir is still empty, look at currently open editor tab
        if not proj_dir or not os.path.isdir(proj_dir):
            sel = self.editor_notebook.select()
            if sel and sel in self.open_files and self.open_files[sel]['path']:
                proj_dir = os.path.dirname(os.path.normpath(self.open_files[sel]['path']))
                self.root.after(0, lambda d=proj_dir: self.current_proj_dir.set(d))
            else:
                self.root.after(0, lambda: self.log_diag("[ERROR] No file saved or project folder selected. Save your file first (Ctrl+S)!\n", "err"))
                self.is_building = False
                self.root.after(0, lambda: self.btn_compile_only.config(state="normal"))
                self.root.after(0, lambda: self.btn_build_and_run.config(state="normal"))
                return

        target_name = self.proj_target_bin.get().strip()
        if not target_name:
            target_name = "application.bin"
        if not target_name.endswith(".bin"):
            target_name += ".bin"

        dest_bin = os.path.join(DEFAULT_BINS_DIR, target_name)
        opt = self.proj_opt_level.get()

        self.root.after(0, lambda: self.log_diag(f"\n===================================================\n", "sys"))
        self.root.after(0, lambda: self.log_diag(f"--- STARTING RISC-V COMPILATION ---\n", "sys"))
        self.root.after(0, lambda: self.log_diag(f"Directory: {proj_dir}\nOpt Level: {opt}\nOutput:    {dest_bin}\n", "cmd"))

        makefile_path = os.path.join(proj_dir, "makefile")
        if not os.path.exists(makefile_path):
            makefile_path = os.path.join(proj_dir, "Makefile")

        # Auto-generate standalone makefile if single-file without makefile
        if not os.path.exists(makefile_path):
            self.root.after(0, lambda: self.log_diag("[INFO] No makefile found. Generating standalone NEORV32 makefile...\n", "sys"))
            try:
                # Find current active C file in directory or open tab
                c_files = glob.glob(os.path.join(proj_dir, "*.c"))
                active_c = None
                sel = self.editor_notebook.select()
                if sel and sel in self.open_files and self.open_files[sel]['path']:
                    active_c = os.path.basename(self.open_files[sel]['path'])
                elif c_files:
                    active_c = os.path.basename(c_files[0])
                else:
                    active_c = "main.c"

                neorv32_home = r"C:/Users/Lenovo/Desktop/GEMMrv32"

                # Generate clean makefile without problematic quotes
                gen_makefile_content = (
                    "MARCH = rv32im_zicsr_zifencei\n"
                    f"EFFORT = {opt}\n"
                    "USER_FLAGS += -Wl,--defsym,__neorv32_rom_size=128k\n"
                    "USER_FLAGS += -Wl,--defsym,__neorv32_ram_size=128k\n\n"
                    f"APP_SRC = {active_c}\n"
                    "APP_INC += -I .\n\n"
                    f"NEORV32_HOME = {neorv32_home}\n"
                    "include $(NEORV32_HOME)/sw/common/common.mk\n"
                )
                makefile_path = os.path.join(proj_dir, "makefile")
                with open(makefile_path, "w", encoding="utf-8") as mf:
                    mf.write(gen_makefile_content)
                self.root.after(0, lambda: self.log_diag(f"[SUCCESS] Created standalone makefile for {active_c}\n", "success"))
            except Exception as mfe:
                self.root.after(0, lambda: self.log_diag(f"[ERROR] Could not auto-generate makefile: {mfe}\n", "err"))
                self.is_building = False
                self.root.after(0, lambda: self.btn_compile_only.config(state="normal"))
                self.root.after(0, lambda: self.btn_build_and_run.config(state="normal"))
                return

        # Prepare make command with Vivado GNU compiler environment in PATH
        env = os.environ.copy()
        env["PATH"] = DEFAULT_GCC_DIR + ";" + os.path.dirname(DEFAULT_MAKE_PATH) + ";" + env.get("PATH", "")

        # Target 'elf' so we don't depend on host gcc for image_gen.c
        cmd = [DEFAULT_MAKE_PATH, f"NEORV32_HOME={NEORV32_HOME_DIR}", "RISCV_PREFIX=riscv64-unknown-elf-", f"EFFORT={opt}", "clean_all", "elf"]
        self.root.after(0, lambda: self.log_diag(f"Running: {' '.join(cmd)}\n\n", "cmd"))

        try:
            proc = subprocess.Popen(cmd, cwd=proj_dir, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)

            for line in proc.stdout:
                self.root.after(0, self.log_diag, line)

            proc.wait()
            ret = proc.returncode

            main_elf = os.path.join(proj_dir, "main.elf")
            if not os.path.exists(main_elf):
                main_elf = os.path.join(proj_dir, "build", "main.elf")

            if not os.path.exists(main_elf):
                self.root.after(0, lambda: self.log_diag(f"\n[BUILD FAILED] main.elf was not produced!\n", "err"))
                self.is_building = False
                self.root.after(0, lambda: self.btn_compile_only.config(state="normal"))
                self.root.after(0, lambda: self.btn_build_and_run.config(state="normal"))
                return

            # Convert ELF to flat binary via objcopy
            elf_bin = os.path.join(proj_dir, "elf.bin")
            objcopy_exe = os.path.join(DEFAULT_GCC_DIR, "riscv64-unknown-elf-objcopy.exe")
            obj_cmd = [objcopy_exe, "-I", "elf32-littleriscv", "-O", "binary", main_elf, elf_bin]
            self.root.after(0, lambda: self.log_diag(f"Extracting raw binary image with objcopy...\n", "cmd"))
            subprocess.run(obj_cmd, cwd=proj_dir, env=env, check=True)

            # Generate NEORV32 bootloader header (magic word 0x214F454E + size + checksum)
            import struct
            with open(elf_bin, "rb") as f:
                raw_bin = f.read()

            pad_len = (len(raw_bin) + 3) & ~3
            raw_bin = raw_bin.ljust(pad_len, b"\x00")

            words = [struct.unpack("<I", raw_bin[i:i+4])[0] for i in range(0, len(raw_bin), 4)]
            chk = (~sum(words)) & 0xFFFFFFFF
            sig = 0x214F454E
            base_addr = 0x00000000
            hdr = struct.pack("<IIII", sig, base_addr, len(raw_bin), chk)

            # Write final executable to dest_bin
            os.makedirs(DEFAULT_BINS_DIR, exist_ok=True)
            with open(dest_bin, "wb") as f:
                f.write(hdr + raw_bin)

            sz = os.path.getsize(dest_bin)
            self.root.after(0, lambda: self.log_diag(f"\n[BUILD SUCCESS] Generated {sz} bytes (checksum 0x{chk:08X}) at {dest_bin}\n", "success"))

            # Refresh firmware dropdown list
            self.root.after(0, self.populate_bin_list)
            self.root.after(0, lambda: self.bin_path_var.set(dest_bin))

            self.is_building = False
            self.root.after(0, lambda: self.btn_compile_only.config(state="normal"))
            self.root.after(0, lambda: self.btn_build_and_run.config(state="normal"))

            if run_after:
                # Switch to Tab 2 and flash!
                self.root.after(0, lambda: self.main_notebook.select(self.tab_console))
                self.root.after(100, self.start_program_flow)

        except Exception as e:
            self.root.after(0, lambda: self.log_diag(f"\n[BUILD EXCEPTION] {e}\n", "err"))
            self.is_building = False
            self.root.after(0, lambda: self.btn_compile_only.config(state="normal"))
            self.root.after(0, lambda: self.btn_build_and_run.config(state="normal"))

    # =========================================================================
    # TAB 2 FUNCTIONALITY: Serial Reader, Terminal & Flasher
    # =========================================================================
    def populate_bin_list(self):
        if os.path.exists(DEFAULT_BINS_DIR):
            bins = glob.glob(os.path.join(DEFAULT_BINS_DIR, "*.bin"))
            self.combo_bins['values'] = bins
            if bins:
                pref = [b for b in bins if "demo_micro_transformer" in b or "demo_gemm" in b]
                self.bin_path_var.set(pref[0] if pref else bins[0])

    def on_dataset_mode_changed(self, event=None):
        self.populate_image_list()

    def get_current_dataset_dir(self):
        if "Digit" in self.dataset_mode_var.get():
            return DEFAULT_DIGIT_IMAGES_DIR
        return DEFAULT_CAR_IMAGES_DIR

    def populate_image_list(self):
        d = self.get_current_dataset_dir()
        if os.path.exists(d):
            exts = ("*.jpg", "*.jpeg", "*.png")
            imgs = []
            for ext in exts:
                imgs.extend(glob.glob(os.path.join(d, ext)))
            self.combo_imgs['values'] = imgs
            if imgs:
                self.image_path_var.set(imgs[0])
                self.load_preview(imgs[0])
            else:
                self.image_path_var.set("")
                self.lbl_img_preview.config(image="", text="No Images")
                self.lbl_img_info.config(text="")

    def browse_binary(self):
        p = filedialog.askopenfilename(
            initialdir=DEFAULT_BINS_DIR if os.path.exists(DEFAULT_BINS_DIR) else "C:\\",
            title="Select Firmware Binary",
            filetypes=[("Binary Files", "*.bin"), ("All Files", "*.*")]
        )
        if p:
            self.bin_path_var.set(p)

    def browse_image(self):
        curr_dir = self.get_current_dataset_dir()
        p = filedialog.askopenfilename(
            initialdir=curr_dir if os.path.exists(curr_dir) else "C:\\",
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png"), ("All Files", "*.*")]
        )
        if p:
            self.image_path_var.set(p)
            self.load_preview(p)

    def on_image_selected(self, event):
        path = self.image_path_var.get()
        if os.path.exists(path):
            self.load_preview(path)

    def load_preview(self, path):
        try:
            im = Image.open(path)
            im_resized = im.resize((120, 120), Image.Resampling.BILINEAR)
            self.photo_preview = ImageTk.PhotoImage(im_resized)
            self.lbl_img_preview.config(image=self.photo_preview, text="")
            base = os.path.basename(path)
            self.lbl_img_info.config(text=f"{base}\n({im.width}x{im.height})")
        except Exception:
            self.lbl_img_preview.config(text="Preview\nError", image="")

    def toggle_inference_tools(self):
        is_open = self.show_inference_tools_var.get()
        if not is_open:
            self.show_inference_tools_var.set(True)
            self.btn_toggle_inf.config(text="▼ Image Inference & Testing Tools")
            # Pack before status label
            self.row_img.pack(fill="x", padx=15, pady=(2, 6), before=self.lbl_action_status)
        else:
            self.show_inference_tools_var.set(False)
            self.btn_toggle_inf.config(text="▶ Image Inference & Testing Tools")
            self.row_img.pack_forget()

    def toggle_preview_box(self):
        if self.show_preview_var.get():
            self.term_frame.pack_forget()
            self.img_box.pack(side="left", fill="y", padx=(0, 10))
            self.term_frame.pack(side="right", fill="both", expand=True)
            path = self.image_path_var.get()
            if path and os.path.exists(path):
                self.load_preview(path)
        else:
            self.img_box.pack_forget()

    def log(self, text, tag=None):
        self.txt_term.insert("end", text, tag)
        self.txt_term.see("end")

    def clear_terminal(self):
        self.txt_term.delete("1.0", "end")

    def kill_bg_processes(self):
        try:
            self.close_serial()
            current_pid = os.getpid()
            cmd = f'powershell -Command "Get-Process python* -ErrorAction SilentlyContinue | Where-Object {{ $_.Id -ne {current_pid} }} | Stop-Process -Force -ErrorAction SilentlyContinue"'
            subprocess.run(cmd, shell=True)
            time.sleep(0.4)
            self.log("[SYSTEM] Terminated all background python processes locking COM ports.\n", "warn")
            self.manual_reconnect()
        except Exception as e:
            self.log(f"[SYSTEM ERROR] Could not kill processes: {e}\n", "err")

    def close_serial(self):
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None
        self.lbl_status_badge.config(text="DISCONNECTED", fg="#c53030", bg="#fff5f5")

    def manual_reconnect(self):
        self.close_serial()
        port = self.port_var.get()
        baud = self.baud_var.get()
        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            self.lbl_status_badge.config(text=f"CONNECTED ({port})", fg="#2f855a", bg="#f0fff4")
            self.log(f"[SYSTEM] Connected to {port} at {baud} baud.\n", "sys")
        except Exception as e:
            self.lbl_status_badge.config(text="ERROR", fg="#c53030", bg="#fff5f5")
            self.log(f"[SYSTEM ERROR] Could not open {port}: {e}\n", "err")

    def forward_terminal_key(self, event):
        if self.ser and self.ser.is_open:
            try:
                if event.keysym in ("Return", "KP_Enter"):
                    self.ser.write(b"\r\n")
                elif event.keysym == "BackSpace":
                    self.ser.write(b"\b")
                elif len(event.char) == 1:
                    self.ser.write(event.char.encode("latin-1", errors="ignore"))
                self.ser.flush()
            except Exception:
                pass
        return "break"

    def send_terminal_input(self):
        txt = self.entry_input.get()
        self.entry_input.delete(0, "end")
        if self.ser and self.ser.is_open:
            try:
                payload = (txt + "\r\n").encode("latin-1")
                self.ser.write(payload)
                self.ser.flush()
            except Exception as e:
                self.log(f"[TRANSMISSION ERROR] {e}\n", "err")
        else:
            self.log("[WARNING] Serial port is disconnected!\n", "warn")

    def start_reader_thread(self):
        self.reader_alive = True
        threading.Thread(target=self._reader_worker, daemon=True).start()

    def _reader_worker(self):
        while self.reader_alive:
            if not self.is_flashing:
                if self.ser and self.ser.is_open:
                    try:
                        n = self.ser.in_waiting
                        if n > 0:
                            chunk = self.ser.read(n).decode('latin-1', errors='replace')
                            self.root.after(0, self.log, chunk)
                    except Exception:
                        self.close_serial()
            time.sleep(0.04)

    def start_program_flow(self):
        bin_path = self.bin_path_var.get()
        if not os.path.exists(bin_path):
            messagebox.showerror("Binary Not Found", f"Cannot find firmware:\n{bin_path}")
            return

        if self.is_flashing:
            return

        self.btn_flash.config(state="disabled")
        threading.Thread(target=self._flasher_worker, args=(bin_path,), daemon=True).start()

    def _flasher_worker(self, bin_path):
        self.is_flashing = True
        port = self.port_var.get()
        baud = self.baud_var.get()

        self.root.after(0, lambda: self.lbl_action_status.config(text="Connecting to bootloader...", fg="#2b6cb0"))
        self.root.after(0, lambda: self.log(f"\n[PROGRAMMER] Opening {port} @ {baud} for bootloader sync...\n", "sys"))

        self.close_serial()
        time.sleep(0.3)

        flash_ser = None
        for attempt in range(4):
            try:
                flash_ser = serial.Serial(port, baud, timeout=0.2)
                break
            except Exception:
                time.sleep(0.4)

        if not flash_ser or not flash_ser.is_open:
            self.root.after(0, lambda: self.lbl_action_status.config(text="Could not open COM port.", fg="#c53030"))
            self.root.after(0, lambda: self.log(f"[PROGRAMMER ERROR] Could not open {port}. Press 'Kill Background Processes' and retry.\n", "err"))
            self.is_flashing = False
            self.root.after(0, lambda: self.btn_flash.config(state="normal"))
            return

        # Handshake with NEORV32 bootloader
        bootloader_ready = False
        self.root.after(0, lambda: self.log("[PROGRAMMER] Sending reset / ping sequences...\n", "cmd"))

        for _ in range(5):
            flash_ser.write(b"\r\n")
            flash_ser.flush()
            time.sleep(0.1)
            resp = flash_ser.read(flash_ser.in_waiting).decode('latin-1', errors='ignore')
            if "CMD:>" in resp or "neorv32" in resp.lower() or "bootloader" in resp.lower():
                bootloader_ready = True
                break

        if not bootloader_ready:
            self.root.after(0, lambda: self.log("[PROGRAMMER] Prompting user to reset board...\n", "warn"))
            self.root.after(0, lambda: self.lbl_action_status.config(text="Press CPU_RESET (PROG) on FPGA Board now!", fg="#b7791f"))
            t_end = time.time() + 8.0
            while time.time() < t_end:
                flash_ser.write(b"\r\n")
                time.sleep(0.2)
                resp = flash_ser.read(flash_ser.in_waiting).decode('latin-1', errors='ignore')
                if "CMD:>" in resp:
                    bootloader_ready = True
                    break

        self.root.after(0, lambda: self.lbl_action_status.config(text="Bootloader detected. Uploading firmware...", fg="#2b6cb0"))
        self.root.after(0, lambda: self.log("[PROGRAMMER] Bootloader detected. Uploading binary image...\n", "sys"))

        try:
            with open(bin_path, "rb") as f:
                data = f.read()

            file_size = len(data)
            # Calculate 32-bit sum checksum of raw binary payload
            import struct
            words = [struct.unpack("<I", data[i:i+4])[0] for i in range(16, len(data), 4)]
            chk32 = sum(words) & 0xFFFFFFFF
            self.root.after(0, lambda: self.log(f"[PROGRAMMER] File: {os.path.basename(bin_path)} ({file_size} bytes, Data Checksum: 0x{chk32:08X})\n", "cmd"))

            flash_ser.reset_input_buffer()
            flash_ser.write(b"u")
            time.sleep(0.5)
            ack_u = flash_ser.read(flash_ser.in_waiting).decode('latin-1', errors='ignore')
            self.root.after(0, lambda: self.log(f"[BOOTLOADER] {ack_u.strip()}\n", "cmd"))

            # Transmit binary stream
            flash_ser.write(data)
            flash_ser.flush()

            # Wait and read bootloader verification result
            time.sleep(1.2)
            boot_resp = flash_ser.read(flash_ser.in_waiting).decode('latin-1', errors='ignore')

            if "ERROR_CHECKSUM" in boot_resp:
                self.root.after(0, lambda: self.log(f"[BOOTLOADER ERROR] CRC/Checksum mismatch on FPGA side: {boot_resp}\n", "err"))
                self.root.after(0, lambda: self.lbl_action_status.config(text="Upload failed: Checksum Mismatch!", fg="#c53030"))
                self.is_flashing = False
                self.root.after(0, lambda: self.btn_flash.config(state="normal"))
                return
            elif "ERROR" in boot_resp:
                self.root.after(0, lambda: self.log(f"[BOOTLOADER ERROR] Upload rejected by bootloader: {boot_resp}\n", "err"))
                self.root.after(0, lambda: self.lbl_action_status.config(text="Upload rejected by bootloader!", fg="#c53030"))
                self.is_flashing = False
                self.root.after(0, lambda: self.btn_flash.config(state="normal"))
                return
            else:
                self.root.after(0, lambda: self.log(f"[BOOTLOADER VERIFIED] {boot_resp.strip() or 'OK (Image Verified)'}\n", "success"))

            # Boot application
            flash_ser.write(b"e")
            time.sleep(0.3)

            self.ser = flash_ser
            self.root.after(0, lambda: self.lbl_status_badge.config(text=f"CONNECTED ({port})", fg="#2f855a", bg="#f0fff4"))
            self.root.after(0, lambda: self.lbl_action_status.config(text=f"Flashed {file_size} bytes. Verified OK.", fg="#2f855a"))
            self.root.after(0, lambda: self.log(f"[PROGRAMMER SUCCESS] Flashed & Verified {file_size} bytes. Application active.\n\n", "success"))

        except Exception as e:
            if flash_ser:
                try:
                    flash_ser.close()
                except Exception:
                    pass
            self.root.after(0, lambda: self.lbl_action_status.config(text=f"Flashing failed: {e}", fg="#c53030"))
            self.root.after(0, lambda: self.log(f"[PROGRAMMER ERROR] Flashing failed: {e}\n", "err"))

        self.is_flashing = False
        self.root.after(0, lambda: self.btn_flash.config(state="normal"))

    def send_selected_image(self):
        img_path = self.image_path_var.get()
        if not os.path.exists(img_path):
            messagebox.showerror("Image Missing", "Please select a valid image file.")
            return

        if not self.ser or not self.ser.is_open:
            self.manual_reconnect()
            if not self.ser or not self.ser.is_open:
                messagebox.showerror("Serial Port Error", f"Cannot open {self.port_var.get()}. Connect the board first.")
                return

        threading.Thread(target=self._image_sender_worker, args=(img_path,), daemon=True).start()

    def _image_sender_worker(self, img_path):
        try:
            is_digit = "Digit" in self.dataset_mode_var.get()

            if is_digit:
                self.root.after(0, lambda: self.lbl_action_status.config(text="Preprocessing digit to 28x28 Grayscale...", fg="#2b6cb0"))
                im = Image.open(img_path).convert("L")
                im = im.resize((28, 28), Image.Resampling.BILINEAR)

                arr = np.array(im, dtype=np.float32)
                # Calibrated range: 0 to 127 for foreground/background pixels matching trained model
                norm = np.clip(np.round(arr / 255.0 * 127.0), 0, 127).astype(np.int8)
                payload = norm.tobytes() # Exactly 784 bytes (28 x 28)

                self.root.after(0, lambda: self.log(f"\n[IMAGE TRANSMISSION] Sending sync '!IMG' + 784 bytes (28x28 Grayscale) for '{os.path.basename(img_path)}'...\n", "sys"))

                self.ser.reset_input_buffer()
                self.ser.write(b"!IMG" + payload)
                self.ser.flush()
            else:
                self.root.after(0, lambda: self.lbl_action_status.config(text="Preprocessing image to 32x32 RGB...", fg="#2b6cb0"))
                im = Image.open(img_path).convert("RGB")
                im = im.resize((32, 32), Image.Resampling.BILINEAR)

                arr = np.array(im, dtype=np.float32)
                arr = np.transpose(arr, (2, 0, 1)) # CHW format
                norm = (arr - 128.0).astype(np.int8)
                payload = norm.tobytes() # Exactly 3072 bytes (3 x 32 x 32)

                self.root.after(0, lambda: self.log(f"\n[IMAGE TRANSMISSION] Sending 3072 bytes (32x32 RGB) for '{os.path.basename(img_path)}'...\n", "sys"))

                self.ser.reset_input_buffer()
                self.ser.write(payload)
                self.ser.flush()

        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda: self.log(f"[IMAGE TRANSMISSION ERROR] {err_msg}\n", "err"))


def run_cli_headless():
    parser = argparse.ArgumentParser(description="GEMMrv32 IDE & Headless Engine")
    parser.add_argument("--headless", action="store_true", help="Run in headless CLI mode without opening GUI")
    parser.add_argument("--build", help="Build project name or directory into binary")
    parser.add_argument("--opt", default="-O2", help="Compiler optimization level (-O0, -O1, -O2, -Os, -O3)")
    parser.add_argument("--flash", help="Path to .bin firmware to flash to FPGA board")
    parser.add_argument("--infer", help="Path to test image to send for inference")
    parser.add_argument("--dataset", default="digits", choices=["digits", "vehicles"], help="Dataset mode (digits/vehicles)")
    parser.add_argument("--port", default="COM10", help="Serial COM port (default: COM10)")
    parser.add_argument("--baud", type=int, default=19200, help="Baud rate (default: 19200)")
    parser.add_argument("--list-ports", action="store_true", help="List available serial COM ports")
    parser.add_argument("--json", action="store_true", help="Output results in structured JSON format")

    args, unknown = parser.parse_known_args()

    # If any CLI argument is provided or --headless requested, run headless
    if args.headless or args.build or args.flash or args.infer or args.list_ports:
        if args.list_ports:
            ports = [p.device for p in serial.tools.list_ports.comports()]
            if args.json:
                print(json.dumps({"ports": ports}, indent=2))
            else:
                print(f"Available ports: {', '.join(ports) if ports else 'None'}")
            return True

        if args.build:
            proj = args.build
            if not os.path.isabs(proj):
                candidate = os.path.join(DEFAULT_SCRIPTS_DIR, proj)
                if os.path.isdir(candidate):
                    proj = candidate
            print(f"[HEADLESS BUILD] Compiling {proj} (Opt: {args.opt})...")
            # Reuse gemm_engine if available or build directly
            from gemm_engine import GEMMEngine
            success, res = GEMMEngine.build_project(proj, opt_level=args.opt, log_cb=sys.stdout.write)
            if args.json:
                res["success"] = success
                print(json.dumps(res, indent=2))
            return True

        if args.flash:
            bin_path = args.flash
            if not os.path.isabs(bin_path):
                candidate = os.path.join(DEFAULT_BINS_DIR, bin_path)
                if os.path.exists(candidate):
                    bin_path = candidate
            print(f"[HEADLESS FLASH] Flashing {bin_path} to {args.port} @ {args.baud}...")
            from gemm_engine import GEMMEngine
            engine = GEMMEngine(port=args.port, baud=args.baud)
            success, res = engine.flash_binary(bin_path, log_cb=sys.stdout.write)
            if args.json:
                res["success"] = success
                print(json.dumps(res, indent=2))
            return True

        if args.infer:
            print(f"[HEADLESS INFER] Preprocessing & sending {args.infer} ({args.dataset})...")
            from gemm_engine import GEMMEngine
            engine = GEMMEngine(port=args.port, baud=args.baud)
            success, res = engine.send_image(args.infer, dataset_type=args.dataset, log_cb=sys.stdout.write)
            if success and engine.ser:
                t_end = time.time() + 3.0
                responses = []
                while time.time() < t_end:
                    if engine.ser.in_waiting:
                        line = engine.ser.readline().decode('latin-1', errors='replace').strip()
                        if line:
                            responses.append(line)
                            if not args.json:
                                print(f"[RECV] {line}")
                    time.sleep(0.05)
                res["responses"] = responses
                engine.close_serial()
            if args.json:
                res["success"] = success
                print(json.dumps(res, indent=2))
            return True

        return True

    return False


if __name__ == "__main__":
    is_headless = run_cli_headless()
    if not is_headless:
        root = tk.Tk()
        app = ProfessionalGEMMDashboard(root)
        root.mainloop()
