import cv2
import numpy as np

class YOLOProcessor:
    def __init__(self):
        self.cls_to_str = ['-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def enhance_dot_conf(self, items):
        for i in range(1, len(items) - 1):
            # Check if current item is a dot and if its neighbors are digits
            if int(items[i][2]) == 1 and 2 <= int(items[i-1][2]) <= 11 and 2 <= int(items[i+1][2]) <= 11:
                # Increase the confidence of the dot
                items[i] = (1.0, items[i][1], items[i][2])
        return items

    def _get_detection_data(self, detect_result):
        boxes = detect_result[0].boxes
        return boxes.conf.cpu().numpy().tolist(), boxes.xywh.cpu().numpy().tolist(), boxes.cls.cpu().numpy().tolist()

    def process_detection(self, detect_result, threshold=0.65):
        conf_list, coor_list, cls_list = self._get_detection_data(detect_result)

        # Filter, sort, and map results
        sorted_data = sorted(
            filter(lambda item: item[0] >= threshold, zip(conf_list, coor_list, cls_list)),
            key=lambda x: x[1][0]
        )
        result = ''.join([self.cls_to_str[int(item[2])] for item in sorted_data])

        return result