import serial
import time

# Initialize the serial connection
ser = serial.Serial(
    port='/dev/serial0',  # Use '/dev/ttyUSB0' if using USB interface
    baudrate=115200,
    timeout=10  # 1 second timeout for read/write operations
)

# ThingSpeak URL and API key
URL = "https://api.thingspeak.com/update?api_key=IYJKIB141WMLJ0GJ&"

def read_from_gsm():
    """Read response from the GSM module."""
    print("GSM says:")
    response = ""
    start_time = time.time()

    # Wait for data from GSM module
    while ser.in_waiting == 0:
        if time.time() - start_time > 10:  # Timeout after 10 seconds
            print("Timeout waiting for GSM response.")
            return

    # Read available data
    while ser.in_waiting > 0:
        response += ser.read().decode('utf-8', errors='ignore')

    print(response)
    return response

def send_command(command):
    """Send AT command to the GSM module."""
    ser.write((command + '\r\n').encode('utf-8'))
    time.sleep(0.2)  # Wait for the GSM module to process the command
    read_from_gsm()

def send_sensor_data_to_cloud(temperature, humidity, ph,risk):
    """Send sensor data to the cloud via HTTP GET request."""
    print("Sending sensor data to cloud...")

    # Prepare data payload
    data = "field1="+str(temperature)+"&field2="+str(humidity)+"&field3="+str(ph)+"&field4="+str(risk)
    full_url = URL + data

    # Initialize HTTP connection
    send_command("AT+HTTPTERM")  # Terminate any existing HTTP session
    send_command("AT+HTTPINIT")  # Initialize HTTP service

    # Set the URL for the HTTP GET request
    send_command(f'AT+HTTPPARA="URL","{full_url}"')

    # Execute the HTTP GET request
    send_command("AT+HTTPACTION=0")

    # Terminate the HTTP session
    send_command("AT+HTTPTERM")

def setup_gsm():
    """Set up the GSM module."""
    print("Setting up the GSM module...")
    time.sleep(30)  # Wait for the GSM module to boot
    send_command("AT")  # Check GSM module communication
    print("GSM initialized.")

# Main script
if __name__ == "__main__":
    try:
        setup_gsm()
        # Example: send dummy data to the cloud
        send_sensor_data_to_cloud(25.69,69.5,42.0)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        ser.close()

