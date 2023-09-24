import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from utils.functions import *

model = YOLO('./best.pt')

# 이미지 로드
img = './test_pic/6.png'
detect_result = model.predict(img)

# YOLO_Processor 선언
segment_detector = SegmentDetector()

# Result of detection process
result = segment_detector.process_detection(detect_result)

print(result)

# def main():
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     main()
#
# cv2.destroyAllWindows()
