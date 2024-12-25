from ultralytics import YOLO
import cv2
model=YOLO('../Weights/yolov8l.pt')
results=model(r"C:\Users\ravik\Downloads\WhatsApp Image 2024-12-17 at 13.43.09_76706d80.jpg",show=True)
cv2.waitKey(0)