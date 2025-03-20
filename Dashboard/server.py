import socket
import threading
import tkinter as tk
from tkinter import scrolledtext

def start_server():
    global client_socket, connection_status_label
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', 12345))
    server_socket.listen(1)
    log_message("Server listening on port 12345")
    
    try:
        client_socket, client_address = server_socket.accept()
        log_message(f"Connection from {client_address} established!")
        update_connection_status(True)
        
        client_socket.settimeout(1)
        receive_thread = threading.Thread(target=receive_data, daemon=True)
        receive_thread.start()
    except:
        update_connection_status(False)

def receive_data():
    global client_socket
    while True:
        try:
            data = client_socket.recv(1024)
            if data:
                log_message(f"ESP32: {data.decode().strip()}")
        except socket.timeout:
            pass
        except:
            update_connection_status(False)
            break

def send_message():
    global client_socket
    message = message_entry.get()
    if message and client_socket:
        try:
            client_socket.sendall(message.encode())
            log_message(f"Sent: {message}")
            message_entry.delete(0, tk.END)
        except:
            update_connection_status(False)

def update_connection_status(connected):
    if connected:
        connection_status_label.config(text="Connected", fg="green")
    else:
        connection_status_label.config(text="Disconnected", fg="red")

def reconnect():
    log_message("Attempting to reconnect...")
    start_server()

def log_message(message):
    log_text.config(state=tk.NORMAL)
    log_text.insert(tk.END, message + '\n')
    log_text.config(state=tk.DISABLED)
    log_text.yview(tk.END)

# GUI Setup
root = tk.Tk()
root.title("ESP32 Server GUI")
root.geometry("400x450")

tk.Label(root, text="ESP32 Server Log").pack()
log_text = scrolledtext.ScrolledText(root, state=tk.DISABLED, height=15)
log_text.pack()

message_entry = tk.Entry(root, width=40)
message_entry.pack(pady=5)

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack()

# Connection Status
connection_status_label = tk.Label(root, text="Disconnected", fg="red")
connection_status_label.pack(side=tk.LEFT, padx=10, pady=5)

reconnect_button = tk.Button(root, text="Reconnect", command=reconnect)
reconnect_button.pack(side=tk.RIGHT, padx=10, pady=5)

# Start server in a separate thread
server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

root.mainloop()