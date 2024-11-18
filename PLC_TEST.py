import pyads

# Testverbinding met TwinCAT runtime
try:
    plc = pyads.Connection('127.0.110.1.1.1', 851)  # Vervang door je eigen AMS Net ID
    plc.open()
    print("Verbonden met TwinCAT!")
    plc.close()
except Exception as e:
    print(f"Verbindingsfout: {e}")
