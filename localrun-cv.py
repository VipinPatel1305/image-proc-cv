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

model = YOLO('../models/best.pt') # pretrained YOLOv8n model
print(model.names)
# object classes
classNames = ['Hardhat','Mask','NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person',  'Safety Cone',  'Safety Vest', 'machinery',  'vehicle']
#classNames = ['NO-Hardhat', 'NO-Mask', 'Person']
#model.set_classes(["hard hat", "gloves", "person"])

flag = False
frameCount = 0
image_buffer = []
startVideoWrite = False
FRAMES_IN_VIDEO = 100
def writeImgToVideo(imgBuffer):
    global cap
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for AVI files; you can use 'mp4v' for MP4 files
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_name = f"./output/output_{timestamp}.mp4"
    video = cv2.VideoWriter(output_file_name, fourcc, fps, (width, height))
    print("start writing video")
    # Write images to the video file
    counter = 0
    for image in imgBuffer:
        video.write(image)  # Write the frame to the video
        counter += 1
    video.release()
    print("video recorded")
    #follow up action: whatsapp/email

    
while True:
    success, img = cap.read()
    #results = model(img, stream=True, verbose=False)
    results = model.predict(img, stream=True, conf=0.85, max_det=10, show=False, verbose=False, classes=[2, 3, 5])
    personDetected = False
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
                
                if classNames[cls] == "Person":
                    personDetected = True
                    #print("person detected")
            
                if startVideoWrite == True and frameCount < FRAMES_IN_VIDEO:
                    image_buffer.append(img)
                    frameCount += 1
                elif frameCount == FRAMES_IN_VIDEO:
                    startVideoWrite = False
                    frameCount = 0
                    thread = threading.Thread(target=writeImgToVideo, args=(image_buffer,))
                    thread.start()
                    
                if classNames[cls] == "NO-Hardhat" and personDetected:
                    if startVideoWrite == False and frameCount == 0:
                        image_buffer = []
                        startVideoWrite = True
                        image_buffer.append(img)
                        print("create buffer")
                        

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()
