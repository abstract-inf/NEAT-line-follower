import socket

class ESP32Client:
    """
    A TCP client for communicating with an ESP32 device.
    
    Args:
        host (str): Hostname or IP of the ESP32 (default: "esp32.local")
        port (int): TCP port number (default: 5000)
        timeout (int): Connection timeout in seconds (default: 5)
    """
    
    def __init__(self, host="esp32.local", port=5000, timeout=5):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.ip_address = None
        self.socket = None
        
    def resolve_host(self):
        """
        Resolve the hostname to an IP address using mDNS.
        
        Returns:
            str: The resolved IP address, or None if resolution failed
        """
        try:
            self.ip_address = socket.gethostbyname(self.host)
            print(f"Resolved IP: {self.ip_address}")
            return self.ip_address
        except socket.gaierror as e:
            print(f"Error resolving {self.host}: {e}")
            return None
    
    def connect(self):
        """
        Establish a connection to the ESP32.
        
        Returns:
            bool: True if connection succeeded, False otherwise
        """
        if not self.ip_address:
            if not self.resolve_host():
                return False
                
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.ip_address, self.port))
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def send_command(self, command):
        """
        Send a command to the ESP32 and receive the response.
        
        Args:
            command (str): The command to send (e.g., "LED_ON")
            
        Returns:
            str: The response from ESP32, or None if failed
        """
        if not command:
            return None
            
        try:
            if not self.socket or not self._is_connected():
                if not self.connect():
                    return None
                    
            self.socket.sendall(command.encode())
            response = self.socket.recv(1024).decode()
            return response
        except Exception as e:
            print(f"Command failed: {e}")
            return None
        finally:
            self.disconnect()
    
    def _is_connected(self):
        """Check if the socket is still connected."""
        try:
            # Try to get peer name to check connection
            self.socket.getpeername()
            return True
        except:
            return False
    
    def disconnect(self):
        """Close the connection if it exists."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            finally:
                self.socket = None
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.disconnect()