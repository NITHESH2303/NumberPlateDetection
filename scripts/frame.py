"""
Frame Extraction Utility
Author: Nithesh Kanna

Extracts frames from video streams when cars are detected.
"""
import cv2
import os
from random import randrange

car_tracker = cv2.CascadeClassifier('cars.xml')

video_images = 'img/frame'

# Video source - can be:
# - File path: 'video.mp4'
# - Webcam: 0
# - RTSP stream: 'rtsp://username:password@ip:port/path'
cam = cv2.VideoCapture(0)  # Default: webcam

frameFrequency = 30

total_frame = 0
id = 0
while True:

    frame_read, frame = cam.read()

    if frame_read is False:
        break
    total_frame += 1

    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_tracker.detectMultiScale(grayscale_img)

    found = False

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(255), randrange(256)), 2)
        found = True

    if found:
        if total_frame % frameFrequency == 0:
            id += 1
            image_name = video_images + str(id) + '.jpg'
            cv2.imwrite(image_name, frame)
            print(image_name)

    cv2.imshow('Car', frame)
    key = cv2.waitKey(1)
    #
    if key == 81 or key == 113:
        break

cam.release()

print("completed")
