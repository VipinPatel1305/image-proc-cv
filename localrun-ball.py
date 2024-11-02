from ultralytics import YOLO
import cv2
import math 
import threading
import sys
import datetime
import threading
import pygame

# start webcam
#http://192.168.0.123:7123/stream.mjpg
cap = cv2.VideoCapture("http://192.168.0.160:7123/stream.mjpg")
#cap = cv2.VideoCapture("../assets/v1.mp4")
cap.set(3, 640)
cap.set(4, 480)

model = YOLO('../models/tennis.pt') # pretrained YOLOv8n model
classNames = model.names
# object classes
pygame.mixer.init()
flag = False
frameCount = 0
image_buffer = []
startVideoWrite = False
FRAMES_IN_VIDEO = 100000

ball_detected = False
px1 = 0
py1 = 0
px2 = 0
py2 = 0
#x1: 207, y1: 305 || x2: 228, y2: 322
imgx1 = 380
imgy1 = 191
imgx2 = 394
imgy2 = 204
#cooridnates:: x1: 218, y1: 350, x2: 237, y2: 368
def play_audio(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    # Keep the thread alive while the audio is playing
    while pygame.mixer.music.get_busy():
        time.sleep(1)
    
def play_sound():
    print("playing sound")
    audio_thread = threading.Thread(target=play_audio, args=("t.mp3",))
    audio_thread.start()
    
def is_overlapping(x1, y1, x2, y2):
    global imgx1, imgy1, imgx2, imgy2
    # If one rectangle is to the left of the other
    if x2 < imgx1 or imgx2 < x1:
        return False
    
    # If one rectangle is above the other
    if y2 < imgy1 or imgy2 < y1:
        return False
    
    # Rectangles overlap
    return True
while True:
    success, img = cap.read()
    #results = model(img, stream=True, verbose=False)
    results = model.predict(img, stream=True, conf=0.65, max_det=2, show=False, verbose=False)
    personDetected = False
    # coordinates
    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            if ball_detected == True:
                if py1 > y1:
                    ball_detected = False
                    py1 = 0
                    if is_overlapping(x1, y1, x2, y2):
                        play_sound()
            else:
                ball_detected = True
                
            print(f" cooridnates:: x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")    
            px1 = x1
            px2 = x2
            py1 = y1
            py2 = y2    
            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # confidence
            confidence = math.ceil((box.conf[0]*100))/10
            class_name = classNames[int(box.cls[0])]
            print(f"Confidence: {confidence} | Class: {class_name}")
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, f"{class_name} {confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 0, 0), 2)
                        

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()
