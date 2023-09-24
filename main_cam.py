import cv2
import torch
from ultralytics import YOLO
from utils.functions import *

model = YOLO('./best.pt')
segment_detector = SegmentDetector()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 객체 탐지 수행
    model_apply = model.predict(frame)

    # 탐지된 객체를 처리
    result = segment_detector.process_detection(model_apply)

    # 결과를 프레임에 표시
    cv2.putText(frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('YOLO Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()