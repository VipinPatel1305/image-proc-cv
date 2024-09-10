from ultralytics import YOLO
import cv2
import math 
import threading
import sys
import datetime
# start webcam
#http://192.168.0.123:7123/stream.mjpg
cap = cv2.VideoCapture("http://192.168.0.123:7123/stream.mjpg")
cap.set(3, 640)
cap.set(4, 480)

model = YOLO('../models/yolov8s-pose.pt') # pretrained YOLOv8n model
#results = model(source = "http://192.168.0.123:7123/stream.mjpg", stream=True, show=True)
print(model.names)



labels = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

while True:
    success, img = cap.read()
    results = model(source=img, stream=True, verbose=False)
    for result in results:
        if hasattr(result, 'keypoints'):
            for person in result.keypoints:
                    #print("==============tensor=========")
                    xy_cpu = person.xy.cpu()

                    # Iterate over all values
                    for batch_idx in range(xy_cpu.size(0)):  # Iterating over batch dimension
                        for i in range(xy_cpu.size(1)):  # Iterating over second dimension
                            x = xy_cpu[batch_idx, i, 0].item()  # Access x-coordinate and convert to Python float
                            y = xy_cpu[batch_idx, i, 1].item()  # Access y-coordinate and convert to Python float
                            #print(f"Index [{batch_idx}, {i}] - x: {x}, y: {y}")
                            cv2.circle(img, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)
                            
                        

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()
