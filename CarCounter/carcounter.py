import numpy as np
from ultralytics import YOLO
import cv2
import math
import cvzone
from sort import *
# Video capture (replace with webcam or video file)
cap = cv2.VideoCapture(r"C:\Users\ravik\Downloads\cars.mp4")
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Load YOLO model
model = YOLO('../Weights/yolov8l.pt')

# Class names
classnames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

mask=cv2.imread('mask.png')

tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)

totalcount=[]
limits=[400,297,673,297]
while True:
    success, img = cap.read()
    imgregion=cv2.bitwise_and(img,mask)
    imggraphic=cv2.imread(r"C:\Users\ravik\PycharmProjects\ObjectDetection\download.jpeg",cv2.IMREAD_UNCHANGED)
    if imggraphic.shape[-1] != 4:  # If not RGBA
        b, g, r = cv2.split(imggraphic)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255  # Fully opaque
        imggraphic = cv2.merge((b, g, r, alpha))

    cvzone.overlayPNG(img,imggraphic,(0,0))

    if not success:
        break

    # Run YOLO inference
    results = model(imgregion,stream=True)
    detections = np.empty((0, 5))
    # Process each detection
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence value
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class name
            cls = int(box.cls[0])
            currentclass=classnames[cls]
            if currentclass=="car" or currentclass=="bus" or currentclass=="motorbike" or currentclass=="truck" and conf>0.3:
                # cvzone.putTextRect(img, f'{classnames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1,offset=5)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=15,rt=5)
                currentarrray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentarrray))


    resulttracker=tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),thickness=5)
    for result in resulttracker:
        x1,y1,x2,y2,id=result
        print(result)
        x1, y1, x2, y2,id = int(x1), int(y1), int(x2), int(y2),int(id)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=2,colorR=(255,0,255))
        cvzone.putTextRect(img, f' {id}', (max(0, x1), max(35, y1)), scale=2, thickness=3,
                           offset=10)
        cx,cy=x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limits[0]<cx<limits[2] and limits[1]-10<cy<limits[1]+    10:
            if totalcount.count(id)==0:
                totalcount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), thickness=5)

    # cvzone.putTextRect(img,f'Count:{len(totalcount)}',(50,50))
    cv2.putText(img,str(len(totalcount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(0,255,0),8)

    # Display the video frame
    cv2.imshow('Video', img)
    # cv2.imshow('Image region', imgregion)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
