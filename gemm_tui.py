"""
GEMMrvStudio TUI Hardware Console & Telemetry Monitor
The Silis Foundation - High-Performance Industrial Serial Console
Features:
  - Clean, full-screen curses terminal with live streaming serial I/O
  - Universal character streaming: every key typed streams directly to the UART
  - Live hardware metrics banner (Baud, Traffic RX/TX, Connection status)
  - Sanitize embedded null bytes and clean ANSI display
  - Scroll history (PageUp / PageDown) and screen clear (F3)
"""

import sys
import os
import time
import threading
import curses
import serial

from gemm_engine import GEMMEngine


class GEMMConsoleTUI:
    def __init__(self, port=None, baud=19200):
        self.port = port or GEMMEngine.get_default_port()
        self.baud = baud
        self.ser = None
        self.running = True
        
        # Telemetry & Status
        self.stats = {
            "connected": False,
            "bytes_rx": 0,
            "bytes_tx": 0,
            "last_status": "Connecting..."
        }
        
        # Log buffers
        self.log_lines = []
        self.max_log_lines = 3000
        self.scroll_offset = 0
        self.auto_scroll = True
        self.last_key_sent = ""
        
        self.lock = threading.Lock()

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
            self.stats["connected"] = True
            self.stats["last_status"] = "Connected"
            self.add_log(f"[SYSTEM] Connected to {self.port} @ {self.baud} baud (8N1).", attr="cyan")
            return True
        except Exception as e:
            self.stats["connected"] = False
            self.stats["last_status"] = f"Error: {e}"
            self.add_log(f"[ERROR] Could not open {self.port}: {e}", attr="red")
            return False

    def add_log(self, text, attr="normal"):
        with self.lock:
            # Strip null bytes and normalize line endings
            cleaned = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
            lines = cleaned.split("\n")
            for line in lines:
                sanitized = "".join(ch for ch in line if ch >= " " or ch == "\t")
                if sanitized:
                    self.log_lines.append((sanitized, attr))
            if len(self.log_lines) > self.max_log_lines:
                self.log_lines = self.log_lines[-self.max_log_lines:]

    def serial_reader_thread(self):
        while self.running:
            if self.ser and self.ser.is_open:
                try:
                    in_wait = 0
                    try:
                        in_wait = self.ser.in_waiting
                    except (serial.SerialException, OSError):
                        pass

                    if in_wait:
                        raw = self.ser.read(in_wait)
                        self.stats["bytes_rx"] += len(raw)
                        filtered = raw.replace(b"\x00", b"")
                        if filtered:
                            text = filtered.decode("latin-1", errors="replace")
                            self.add_log(text, attr="rx")
                except Exception as e:
                    if hasattr(e, "errno") and e.errno == 4:
                        continue
                    self.stats["connected"] = False
                    self.stats["last_status"] = "Disconnected"
                    self.add_log(f"[ERROR] Serial read error: {e}", attr="red")
                    time.sleep(1)
                    # Auto-reconnect
                    try:
                        self.ser.close()
                        self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
                        self.stats["connected"] = True
                        self.stats["last_status"] = "Connected"
                        self.add_log(f"[SYSTEM] Reconnected to {self.port}.", attr="cyan")
                    except Exception:
                        pass
            time.sleep(0.01)

    def send_raw_bytes(self, b):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(b)
                self.stats["bytes_tx"] += len(b)
            except Exception as e:
                self.add_log(f"[ERROR] Send error: {e}", attr="red")

    def run(self):
        try:
            curses.wrapper(self._main_curses)
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            if self.ser and self.ser.is_open:
                try:
                    self.ser.close()
                except Exception:
                    pass
            print("[INFO] GEMMrvStudio console closed.")

    def _main_curses(self, stdscr):
        self.stdscr = stdscr
        curses.curs_set(0)  # Clean streaming cursor
        curses.use_default_colors()
        stdscr.nodelay(True)
        stdscr.keypad(True)
        
        # Color pairs
        # 1: Cyan / default (Title, headers)
        # 2: Green / default (Normal received data)
        # 3: Red / default (Errors)
        # 4: Yellow / default (Stats)
        # 5: Inverted (Top header banner)
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_YELLOW, -1)
        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_CYAN)

        # Start serial background worker
        self.connect()
        t = threading.Thread(target=self.serial_reader_thread, daemon=True)
        t.start()

        try:
            while self.running:
                self.draw_ui()
                self.handle_input()
                time.sleep(0.02)
        except KeyboardInterrupt:
            self.running = False

        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass

    def draw_ui(self):
        max_y, max_x = self.stdscr.getmaxyx()
        if max_y < 8 or max_x < 40:
            self.stdscr.erase()
            self.stdscr.addstr(0, 0, "Window too small (min 40x8)!", curses.color_pair(3))
            self.stdscr.refresh()
            return

        self.stdscr.erase()

        # -------------------------------------------------------------
        # 1. Top Industrial Header Bar
        # -------------------------------------------------------------
        header_text = f" GEMMrvStudio Serial Console "
        status_text = f" [{self.port} @ {self.baud} 8N1] [STATUS: {'ONLINE' if self.stats['connected'] else 'OFFLINE'}] "
        fill_spaces = max(0, max_x - len(header_text) - len(status_text) - 1)
        top_bar = header_text + (" " * fill_spaces) + status_text
        try:
            self.stdscr.attron(curses.color_pair(5) | curses.A_BOLD)
            self.stdscr.addstr(0, 0, top_bar[:max_x-1])
            self.stdscr.attroff(curses.color_pair(5) | curses.A_BOLD)
        except curses.error:
            pass

        # -------------------------------------------------------------
        # 2. Hardware Status / Traffic Ribbon
        # -------------------------------------------------------------
        last_k_disp = repr(self.last_key_sent) if self.last_key_sent else "None"
        info_str = f" Traffic: RX {self.stats['bytes_rx']:,} B │ TX {self.stats['bytes_tx']:,} B │ Last Sent: {last_k_disp} │ Mode: Live Stream (Type directly)"
        try:
            self.stdscr.attron(curses.color_pair(4))
            self.stdscr.addstr(1, 1, info_str[:max_x - 2])
            self.stdscr.attroff(curses.color_pair(4))
            
            # Divider line
            self.stdscr.attron(curses.color_pair(1))
            self.stdscr.addstr(2, 0, "─" * (max_x - 1))
            self.stdscr.attroff(curses.color_pair(1))
        except curses.error:
            pass

        # -------------------------------------------------------------
        # 3. Main Serial Streaming Log
        # -------------------------------------------------------------
        log_start_y = 3
        log_height = max_y - log_start_y - 2
        
        with self.lock:
            total_logs = len(self.log_lines)
            if self.auto_scroll:
                visible_logs = self.log_lines[-log_height:] if total_logs > log_height else self.log_lines
            else:
                end_idx = max(0, total_logs - self.scroll_offset)
                start_idx = max(0, end_idx - log_height)
                visible_logs = self.log_lines[start_idx:end_idx]

        for idx, (line, attr) in enumerate(visible_logs):
            row = log_start_y + idx
            if row >= max_y - 2:
                break
            
            c_pair = 2
            if attr == "red" or "ERROR" in line:
                c_pair = 3
            elif attr == "cyan" or "[SYSTEM]" in line:
                c_pair = 1

            try:
                safe_line = line.replace("\x00", "")[:max_x - 3]
                self.stdscr.attron(curses.color_pair(c_pair))
                self.stdscr.addstr(row, 1, safe_line)
                self.stdscr.attroff(curses.color_pair(c_pair))
            except (curses.error, ValueError):
                pass

        # -------------------------------------------------------------
        # 4. Bottom Footer & Controls Bar
        # -------------------------------------------------------------
        footer_y = max_y - 1
        legend = " [F3] Clear  │  [F5] Reconnect  │  [PgUp/PgDn] Scroll  │  [Ctrl+C] Exit  │  Streaming All Typed Input "
        try:
            self.stdscr.attron(curses.color_pair(1))
            self.stdscr.addstr(footer_y - 1, 0, "─" * (max_x - 1))
            self.stdscr.attroff(curses.color_pair(1))

            self.stdscr.attron(curses.color_pair(4) | curses.A_BOLD)
            self.stdscr.addstr(footer_y, 1, legend[:max_x - 2])
            self.stdscr.attroff(curses.color_pair(4) | curses.A_BOLD)
        except curses.error:
            pass

        self.stdscr.refresh()

    def handle_input(self):
        try:
            ch = self.stdscr.getch()
        except curses.error:
            return

        if ch == -1:
            return

        # Exit on Ctrl+C
        if ch == 3:
            self.running = False
            return

        # Special Action Keys
        if ch == curses.KEY_F3:
            with self.lock:
                self.log_lines.clear()
            self.add_log("[SYSTEM] Log cleared.", attr="cyan")
            return

        elif ch == curses.KEY_F5:
            self.add_log(f"[SYSTEM] Reconnecting to {self.port}...", attr="cyan")
            self.connect()
            return

        elif ch == curses.KEY_PPAGE:  # Page Up
            self.auto_scroll = False
            self.scroll_offset = min(len(self.log_lines), self.scroll_offset + 10)
            return

        elif ch == curses.KEY_NPAGE:  # Page Down
            self.scroll_offset = max(0, self.scroll_offset - 10)
            if self.scroll_offset == 0:
                self.auto_scroll = True
            return

        # Handle Enter / Carriage Return
        if ch in (10, 13, curses.KEY_ENTER):
            self.send_raw_bytes(b"\r\n")
            self.last_key_sent = "Enter"
            return

        # Handle Backspace
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            self.send_raw_bytes(b"\x08")
            self.last_key_sent = "Backspace"
            return

        # Universal Character Streaming: every typed character is sent directly to UART
        elif 0 <= ch <= 255:
            char_byte = bytes([ch])
            self.send_raw_bytes(char_byte)
            self.last_key_sent = chr(ch) if (32 <= ch <= 126) else f"0x{ch:02X}"


def launch_console(port=None, baud=19200):
    tui = GEMMConsoleTUI(port=port, baud=baud)
    tui.run()


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else None
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 19200
    launch_console(port=p, baud=b)
