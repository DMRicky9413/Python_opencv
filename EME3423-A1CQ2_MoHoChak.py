import cv2
import numpy as np
import serial
import time

ser = serial.Serial('COM7', baudrate=115200, timeout=1)
time.sleep(0.5)
pos = 90
pos1 = 90

def nothing (x):
    pass

cv2.namedWindow('Trackbars')
cv2.createTrackbar('HUELOW', 'Trackbars', 90, 255, nothing)
cv2.createTrackbar('HUEHIGH', 'Trackbars', 120, 255, nothing)
cv2.createTrackbar('SATLOW', 'Trackbars', 160, 255, nothing)
cv2.createTrackbar('SATHIGH', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('VALLOW', 'Trackbars', 85, 255, nothing)
cv2.createTrackbar('VALHIGH', 'Trackbars', 255, 255, nothing)

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    _, img = cam.read()
    cv2.imshow('ori', img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    huelow = cv2.getTrackbarPos('HUELOW', 'Trackbars')
    huehigh = cv2.getTrackbarPos('HUEHIGH', 'Trackbars')
    satlow = cv2.getTrackbarPos('SATLOW', 'Trackbars')
    sathigh = cv2.getTrackbarPos('SATHIGH', 'Trackbars')
    vallow = cv2.getTrackbarPos('VALLOW', 'Trackbars')
    valhigh = cv2.getTrackbarPos('VALHIGH', 'Trackbars')

    FGmask2 = cv2.inRange(hsv, (huelow, satlow, vallow), (huehigh, sathigh, valhigh))
    cv2.imshow('FGmask2', FGmask2)

    contours2, hierarchy2 = cv2.findContours(FGmask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = 0

    for cnt2 in contours2:
        area = cv2.contourArea(cnt2)
        # print(area)
        (x,y,w,h) = cv2.boundingRect(cnt2)

        if area >= 300:
            if area >=biggest_contour:
                biggest_contour = area
                (x, y ,w, h) = cv2.boundingRect(cnt2)
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 3)

                errorPan = (x + w/2) - 640/2

                if abs(errorPan) > 20:
                    pos1 = pos1 - errorPan/60

                if pos1 > 170:
                    pos1 = 170

                if pos1 < 10:
                    pos1 = 10

                seroPos = 'UD' + str(pos1) + '\r'
                ser.write(seroPos.encode('utf-8'))

                time.sleep(0.01)



    cv2.imshow('final', img)

    if cv2.waitKey(1) == 27:
        break

//----------------------------
#include <Servo.h>

Servo myServoRL;
Servo myServoUD;

int servoPinRL = 9;
int servoPinUD = 10;

String servoPos;
int pos;

void setup() {

  Serial.begin(115200);
  myServoRL.attach(servoPinRL);
  myServoUD.attach(servoPinUD);
  myServoRL.write(90);
  myServoUD.write(90);
}

void loop() {

  while (Serial.available() == 0){

  }
  servoPos = Serial.readStringUntil('\r');

  if (servoPos.substring(0,2) == "RL"){
    servoPos = servoPos.substring(2);
    pos = servoPos.toInt();
    myServoRL.write(pos);
    delay(10);
  }
   if (servoPos.substring(0,2) == "UD"){
    servoPos = servoPos.substring(2);
    pos = servoPos.toInt();
    myServoUD.write(pos);
    delay(10);
  }
}
