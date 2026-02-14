import serial, time

try:
    arduino = serial.Serial('COM12', 9600, timeout=2)
    time.sleep(2)
    print("✅ COM12 opened successfully!")
    arduino.close()
except Exception as e:
    print("❌ COM12 failed:", e)
