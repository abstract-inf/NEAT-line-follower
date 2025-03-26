from esp32_client import ESP32Client

client = ESP32Client()

ip = client.resolve_host()

print(ip)

client.disconnect()
client.disconnect()
client.connect()