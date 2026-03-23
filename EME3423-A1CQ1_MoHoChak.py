import cv2
import serial
import time

ARDUINO_PORT = 'COM8'
BAUD_RATE = 9600
CAMERA_ID = 0

try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE)
    time.sleep(2)   
    print(f"Connected Arduino ({ARDUINO_PORT})")
except:
    print(f"UnConnected Arduino {ARDUINO_PORT}，please check com")
    exit()

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("Can't open cam")
    exit()


while True:

    ret, frame = cap.read()
    if not ret:
        print("can't detect")
        break


    sensor_text = "Sensor:  "
    if arduino.in_waiting > 0:
        try:
            raw_data = arduino.readline().decode().strip()
            if raw_data:
                sensor_text = f"Sensor: {raw_data}"
        except:
            pass


    cv2.putText(frame, sensor_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, current_time, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)


    cv2.imshow('EME3423 - Sensor + Camera (AIR)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
print("Finish")

//--------------------------
void setup() {
  Serial.begin(9600); 
}

void loop() {
  int sensorValue = analogRead(A0);  
  Serial.println(sensorValue);        
  delay(100);                        
}
