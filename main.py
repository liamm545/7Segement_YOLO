import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from utils.functions import *

model = YOLO('./best.pt')

# 이미지 로드
img = './test_pic/4.png'
model_apply = model.predict(img)

# SegmentDetector Class 선언
segment_detector = SegmentDetector()

# Result of detection process
result = segment_detector.process_detection(model_apply)

print(result)
