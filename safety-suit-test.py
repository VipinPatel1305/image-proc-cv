from ultralytics import YOLO
import cv2
import math 
import threading
import sys
import datetime
# start webcam
#http://192.168.0.123:7123/stream.mjpg
#cap = cv2.VideoCapture("http://192.168.0.123:7123/stream.mjpg")
cap = cv2.VideoCapture("../assets/arc-suite-test1.mp4")

model = YOLO('/home/vipin/Downloads/safety-suits/suits-v5i-yolo8/runs/detect/safety-suit-b83/weights/best.pt') # pretrained YOLOv8n model
print(model.names)






classNames = [    'Hardhat', 
    'Mask', 
    'NO-Hardhat', 
    'NO-Mask', 
    'NO-Safety Vest', 
    'Person', 
    'Safety Cone', 
    'Safety Vest', 
    'machinery', 
    'vehicle']
   
classNames = ['CAPUZ', 'CAT4', 'luva']

classNames = [
    'arc_flash_suit', 'no_arc_flash_suit', 'partial_arc_flash_suit'
]
while True:
    success, img = cap.read()
    #results = model(img, stream=True, verbose=False)
    results = model.predict(img, stream=True, conf=0.85, max_det=10)
    # coordinates
    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            #print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            #print("Class name -->", classNames[cls])
            

                
            if True:   
            # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                        

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()                       

