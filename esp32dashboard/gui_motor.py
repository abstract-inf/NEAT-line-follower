import tkinter as tk
from tkinter import messagebox
import serial
import threading

class BluetoothApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bluetooth Serial Sender")

        self.serial_conn = None

        tk.Label(root, text="Bluetooth Port:").pack()
        self.port_entry = tk.Entry(root)
        self.port_entry.pack(pady=(0, 10))

        self.connect_button = tk.Button(root, text="Connect", command=self.connect)
        self.connect_button.pack(pady=(0, 10))

        tk.Label(root, text="Message:").pack()
        self.message_entry = tk.Entry(root)
        self.message_entry.pack(pady=(0, 10))
        self.message_entry.bind("<Return>", self.send_data)

        self.sent_label = tk.Label(root, text="Sent: ")
        self.sent_label.pack(pady=(0, 10))

        # Received messages display
        tk.Label(root, text="Received:").pack()
        self.received_text = tk.Text(root, height=10, state='disabled')
        self.received_text.pack(pady=(0, 10))

        # Custom command buttons
        tk.Label(root, text="Quick Commands:").pack()

        commands = [
            "-255,-255", "-200,-200", "-100,-100", "-70,-70",
            "0,0",
            "70,70", "100,100", "200,200", "255,255"
        ]

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=(0, 10))

        for cmd in commands:
            btn = tk.Button(btn_frame, text=cmd, width=10, command=lambda c=cmd: self.send_custom_command(c))
            btn.pack(side=tk.LEFT, padx=2)

        # Start read thread
        self.read_thread = threading.Thread(target=self.read_data, daemon=True)
        self.read_thread.start()

    def connect(self):
        port = self.port_entry.get()
        try:
            self.serial_conn = serial.Serial(port, 9600, timeout=1)
            messagebox.showinfo("Connected", f"Connected to {port}")
        except serial.SerialException as e:
            messagebox.showerror("Connection Error", str(e))

    def send_data(self, event=None):
        if self.serial_conn and self.serial_conn.is_open:
            data = self.message_entry.get()
            self.serial_conn.write(data.encode())
            self.sent_label.config(text=f"Sent: {data}")
            self.message_entry.delete(0, tk.END)
        else:
            messagebox.showwarning("Not Connected", "Please connect to a Bluetooth port first.")

    def send_custom_command(self, command):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.write(command.encode())
            self.sent_label.config(text=f"Sent: {command}")
        else:
            messagebox.showwarning("Not Connected", "Please connect to a Bluetooth port first.")

    def read_data(self):
        while True:
            if self.serial_conn and self.serial_conn.is_open:
                try:
                    data = self.serial_conn.readline().decode().strip()
                    if data:
                        self.received_text.configure(state='normal')
                        self.received_text.insert(tk.END, data + '\n')
                        self.received_text.configure(state='disabled')
                        self.received_text.see(tk.END)
                except:
                    pass

if __name__ == "__main__":
    root = tk.Tk()
    app = BluetoothApp(root)
    root.mainloop()