import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from utils.functions import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="7-segment detection on a given image!")
    parser.add_argument('--roi', nargs=2, default=None, help='Region of interest as top-left(좌상) and bottom-right(우하) coordinates. E.g. --roi "(1,2)" "(400,300)"')
    parser.add_argument("--path", type=str, required=True, help="Path to the image to process.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = YOLO('./best_4.pt')

    # 이미지 로드
    img_path = args.path
    img = cv2.imread(img_path)

    # ROI 처리
    if args.roi:
        roi_top_left = eval(args.roi[0])
        roi_bottom_right = eval(args.roi[1])
        img = img[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

    # Model prediction
    model_apply = model.predict(img)

    # SegmentDetector Class 선언
    segment_detector = SegmentDetector()

    # Result of detection process
    result = segment_detector.process_detection(model_apply)

    print(result)

if __name__ == "__main__":
    main()
