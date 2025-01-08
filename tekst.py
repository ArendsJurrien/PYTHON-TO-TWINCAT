import pyads

def send_text_to_twincat(text, plc_address="39.231.85.117.1.1", port=851):
    """
    Schrijf een tekstbericht naar een string-variabele in TwinCAT.
    """
    try:
        # Verbind met de PLC
        plc = pyads.Connection(plc_address, port)
        plc.open()
        
        if plc.is_open:
            print(f"Verbonden met TwinCAT PLC op {plc_address}:{port}")
            
            # Schrijf het bericht naar de PLC
            plc.write_by_name("Main.status_message", text, pyads.PLCTYPE_STRING)
            print(f"Bericht '{text}' succesvol verzonden naar TwinCAT.")
        else:
            print("Kon geen verbinding maken met de PLC.")
        
        plc.close()
    except Exception as e:
        print(f"Fout bij verzenden: {e}")

# Voorbeeld van het verzenden van een bericht
send_text_to_twincat("Timeout >30 seconds. Herplaats tandwiel")