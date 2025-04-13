import tkinter as tk
from tkinter import ttk, scrolledtext
import serial
import threading
from datetime import datetime

class IMUDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("IMU Dashboard - ESP32 Bluetooth")
        self.root.geometry("800x600")
        
        # Serial connection vars
        self.serial_conn = None
        self.connected = False
        
        # GUI Setup
        self.setup_ui()
        
        # Start serial read thread
        self.read_thread = threading.Thread(target=self.read_serial, daemon=True)
        self.read_thread.start()

    def setup_ui(self):
        # Connection Frame
        conn_frame = ttk.LabelFrame(self.root, text="Connection", padding=10)
        conn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(conn_frame, text="COM Port:").grid(row=0, column=0, sticky=tk.W)
        self.port_entry = ttk.Entry(conn_frame, width=15)
        self.port_entry.grid(row=0, column=1, padx=5)
        
        self.connect_btn = ttk.Button(conn_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=0, column=2, padx=5)
        
        # Status Indicator
        self.status_var = tk.StringVar(value="Disconnected")
        ttk.Label(conn_frame, textvariable=self.status_var, foreground="red").grid(row=0, column=3)
        
        # Control Buttons
        ctrl_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.calibrate_btn = ttk.Button(ctrl_frame, text="Calibrate IMU", 
                                      command=self.send_calibrate, state=tk.DISABLED)
        self.calibrate_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = ttk.Button(ctrl_frame, text="Reset", 
                                   command=self.send_reset, state=tk.DISABLED)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Log Display
        log_frame = ttk.LabelFrame(self.root, text="ESP32 Logs", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, 
                                                width=80, height=25,
                                                font=('Consolas', 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Clear button
        ttk.Button(log_frame, text="Clear Logs", command=self.clear_logs).pack(side=tk.RIGHT)
        
        # Configure tags for colored text
        self.log_text.tag_config('error', foreground='red')
        self.log_text.tag_config('status', foreground='blue')
        self.log_text.tag_config('data', foreground='green')
        self.log_text.tag_config('warning', foreground='orange')

    def toggle_connection(self):
        if not self.connected:
            self.connect_serial()
        else:
            self.disconnect_serial()

    def connect_serial(self):
        port = self.port_entry.get()
        try:
            self.serial_conn = serial.Serial(port, 115200, timeout=1)
            self.connected = True
            self.status_var.set("Connected")
            self.connect_btn.config(text="Disconnect")
            self.calibrate_btn.config(state=tk.NORMAL)
            self.reset_btn.config(state=tk.NORMAL)
            self.log_message("Connected to " + port + " at 115200 baud", 'status')
        except Exception as e:
            self.log_message(f"Connection failed: {str(e)}", 'error')

    def disconnect_serial(self):
        if self.serial_conn:
            self.serial_conn.close()
        self.connected = False
        self.status_var.set("Disconnected")
        self.connect_btn.config(text="Connect")
        self.calibrate_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.DISABLED)
        self.log_message("Disconnected", 'status')

    def send_calibrate(self):
        if self.connected:
            self.serial_conn.write(b"CALIBRATE\n")
            self.log_message("Sent: CALIBRATE command", 'status')

    def send_reset(self):
        if self.connected:
            self.serial_conn.write(b"RESET\n")
            self.log_message("Sent: RESET command", 'status')

    def clear_logs(self):
        self.log_text.delete(1.0, tk.END)

    def log_message(self, message, tag=None):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted_msg = f"[{timestamp}] {message}\n"
        
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, formatted_msg, tag)
        self.log_text.configure(state=tk.DISABLED)
        self.log_text.see(tk.END)

    def read_serial(self):
        while True:
            if self.connected and self.serial_conn:
                try:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    if line:
                        # Auto-detect message type for coloring
                        if line.startswith("ERROR:"):
                            self.log_message(line, 'error')
                        elif line.startswith("STATUS:"):
                            self.log_message(line, 'status')
                        elif line.startswith("YPR:") or line.startswith("OFFSETS:"):
                            self.log_message(line, 'data')
                        elif "warning" in line.lower():
                            self.log_message(line, 'warning')
                        else:
                            self.log_message(line)
                except UnicodeDecodeError:
                    self.log_message("Received non-UTF-8 data", 'error')
                except Exception as e:
                    self.log_message(f"Serial error: {str(e)}", 'error')
                    self.disconnect_serial()

if __name__ == "__main__":
    root = tk.Tk()
    app = IMUDashboard(root)
    root.mainloop()