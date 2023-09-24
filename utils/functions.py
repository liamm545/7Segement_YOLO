import cv2
import numpy as np

class SegmentDetector:
    def __init__(self):
        self.cls_to_str = ['-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    @staticmethod
    def calculate_areas(coor_list):
        return [coor[2] * coor[3] for coor in coor_list]

    def filter_boxes(self, conf_list, coor_list, cls_list, areas, threshold, area_ratio):
        if not conf_list:
            return []
        max_conf_index = conf_list.index(max(conf_list))
        max_conf_area = areas[max_conf_index]

        # 허용 가능 최소 & 최대 넓이
        min_area = area_ratio[0] * max_conf_area
        max_area = area_ratio[1] * max_conf_area

        filtered_vals = []
        for conf, coor, cls, area in zip(conf_list, coor_list, cls_list, areas):
            # '.'일 경우
            if cls == 1.0:
                # 현재 '.'의 conf가 너무 안나와서 일단 예외 처리 함
                if conf >= 0.3:
                    filtered_vals.append((conf, coor, cls))
                continue
            # '-'일 경우
            if cls == 0.0:
                filtered_vals.append((conf, coor, cls))
                continue
            if min_area <= area <= max_area and conf >= threshold:
                filtered_vals.append((conf, coor, cls))
        return filtered_vals

    def process_detection(self, model_apply, threshold=0.6, area_ratio=(0.8, 1.2)):
        detect_result = model_apply[0].boxes

        conf_list = detect_result.conf.cpu().numpy().tolist()
        coor_list = detect_result.xywh.cpu().numpy().tolist()
        cls_list = detect_result.cls.cpu().numpy().tolist()

        # bounding box를 통한 1차 예외 처리
        areas = self.calculate_areas(coor_list)
        filtered_vals = self.filter_boxes(conf_list, coor_list, cls_list, areas, threshold, area_ratio)

        # 여기에 x값과 y값에 대한 예외 처리 추가 필요

        sorted_vals = sorted(filtered_vals, key=lambda x: x[1][0])
        result = ''.join([self.cls_to_str[int(item[2])] for item in sorted_vals])

        return result
