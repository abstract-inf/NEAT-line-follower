import socket

esp_host = "esp32.local"  # Use mDNS instead of IP
port = 5000

# Try to resolve the IP first
try:
    ip_address = socket.gethostbyname(esp_host)
    print(f"Resolved IP: {ip_address}")
except socket.gaierror as e:
    print(f"Error resolving {esp_host}: {e}")
    ip_address = None

if ip_address:
    while True:
        # Get command from user input (Terminal)
        command = input("Enter command to send to ESP32 (e.g., 'LED_ON', 'LED_OFF'): ").strip()

        if command:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5)  # Avoid freezing if ESP32 is unavailable
                s.connect((ip_address, port))

                # Send the command to ESP32
                s.sendall(command.encode())  # Send command fully

                # Wait for and receive the response from ESP32
                data = s.recv(1024).decode()  # Receive response
                print("ESP32 says:", data)

                s.close()  # Close the connection after receiving the response
            except Exception as e:
                print("Connection failed:", e)
else:
    print("Failed to resolve ESP32 IP address.")
