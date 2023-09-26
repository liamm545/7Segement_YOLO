import cv2
import numpy as np
import statistics

class SegmentDetector:
    def __init__(self):
        self.cls_to_str = ['-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    @staticmethod
    def calculate_areas(coor_list):
        return [coor[2] * coor[3] for coor in coor_list]

    def filter_by_area(self, conf_list, coor_list, cls_list, areas, threshold, area_ratio):
        if not conf_list:
            return [], []

        max_conf_index = conf_list.index(max(conf_list))
        max_conf_area = areas[max_conf_index]

        # 허용 가능 최소 & 최대 넓이
        min_area = area_ratio[0] * max_conf_area
        max_area = area_ratio[1] * max_conf_area

        filtered_vals = []
        minus = []  # '-'를 저장하는 리스트
        dot = [] # '.'을 저장하는 리스

        for conf, coor, cls, area in zip(conf_list, coor_list, cls_list, areas):
            if cls == 0.0: # '-' 일 경우
                minus.append((conf, coor, cls))
            elif cls == 1.0: # '.' 일 경우
                dot.append((conf, coor, cls))
            elif min_area <= area <= max_area and conf >= threshold:
                filtered_vals.append((conf, coor, cls))
        # print(filtered_vals)
        # print(dot)
        return filtered_vals, dot, minus

    def filter_minus_by_coordinates(self, minus_list, filtered_by_coordinates):
        if not minus_list:
            return []

        minus_item = minus_list[0]
        sorted_by_x = sorted(filtered_by_coordinates, key=lambda x: x[1][0])

        avg_x_gap = sum(sorted_by_x[i + 1][1][0] - sorted_by_x[i][1][0] for i in range(len(sorted_by_x) - 1)) / (
                len(sorted_by_x) - 1)
        avg_y_coord = sum(item[1][1] for item in sorted_by_x) / len(sorted_by_x)

        if abs(minus_item[1][0] - sorted_by_x[0][1][0] - avg_x_gap) < 0.1 * avg_x_gap and abs(
                minus_item[1][1] - avg_y_coord) < 0.1 * avg_y_coord:
            return [minus_item]
        else:
            return []

    def filter_dots_by_coordinates(self, dot_list, filtered_by_coordinates):
        sorted_numbers = sorted(filtered_by_coordinates, key=lambda x: x[1][0])
        valid_dots = []

        for dot in dot_list:
            conf, coor, cls = dot
            for i, (num_conf, num_coor, num_cls) in enumerate(sorted_numbers):
                if i == len(sorted_numbers) - 1:
                    break  # 마지막 숫자의 오른쪽에는 '.'이 오지 않

                # 현재의 '.'이 현재 숫자와 다음 숫자 사이에 있는지 확인
                if num_coor[0] + num_coor[2]/2 <= coor[0]+coor[3]/2 <= sorted_numbers[i + 1][1][0]:
                    # print(num_coor)
                    # print(num_coor[1] + num_coor[3] / 2)
                    # print(num_coor[1] + num_coor[3])
                    # print(coor)
                    # `.`의 y값이 현재 숫자의 아래 중앙 부분에 위치하는지 확인
                    if num_coor[1] <= coor[1] <= num_coor[1] + num_coor[3]:
                        valid_dots.append(dot)
                    break

        # 여러 개의 '.' 중에서 신뢰도가 가장 높은 '.'을 선택
        if valid_dots:
            valid_dots.sort(key=lambda x: x[0], reverse=True)
            return [valid_dots[0]]
        else:
            return []

    def filter_numbers_by_coordinates(self, filtered_by_area):
        if not filtered_by_area:
            return []

        # 세그먼트의 중심 y값과 h값 계산
        y_centers = [coor[1][1] + coor[1][3] / 2 for coor in filtered_by_area]
        h_values = [coor[1][3] for coor in filtered_by_area]

        avg_y_center = sum(y_centers) / len(y_centers)
        avg_h = sum(h_values) / len(h_values)

        # 중심 y값 기준으로 선별 (평균 세그먼트 높이의 반 정도로 필터링)
        # 따라서 실전에 사용할 때는 애초에 너무 기운 경우는 제외해야 됨
        filtered_by_y = [item for item in filtered_by_area if
                         abs(item[1][1] + item[1][3] / 2 - avg_y_center) < 0.5 * avg_h]

        # x값과 w값 추출
        x_coordinates = [coor[1][0] for coor in filtered_by_y]
        w_values = [coor[1][2] for coor in filtered_by_y]

        # x좌표를 기준으로 정렬하되, 각 세그먼트의 끝 x 좌표를 계산
        end_x_coordinates = [x + w for x, w in zip(x_coordinates, w_values)]
        sorted_end_x_coordinates = sorted(end_x_coordinates)

        # 이웃한 세그먼트들 간의 간격 계산
        gaps = [sorted_end_x_coordinates[i + 1] - sorted_end_x_coordinates[i] for i in
                range(len(sorted_end_x_coordinates) - 1)]

        if not gaps:  # Only one coordinate
            return filtered_by_y

        avg_gap = sum(gaps) / len(gaps)
        std_gap = (sum((gap - avg_gap) ** 2 for gap in gaps) / len(gaps)) ** 0.5

        # 간격이 평균 간격보다 표준편차의 2배 이상 차이나면 필터링
        threshold = avg_gap + 2 * std_gap

        final_filtered = [item for idx, item in enumerate(filtered_by_y[:-1]) if gaps[idx] < threshold]
        final_filtered.append(filtered_by_y[-1])  # 마지막 숫자는 항상 포함

        return final_filtered

    def process_detection(self, model_apply, threshold=0.6, area_ratio=(0.7, 1.5)):
        detect_result = model_apply[0].boxes

        # 자동으로 conf 순서대로 맞춰져있음
        conf_list = detect_result.conf.cpu().numpy().tolist()
        coor_list = detect_result.xywh.cpu().numpy().tolist()
        cls_list = detect_result.cls.cpu().numpy().tolist()

        # bounding box를 통한 1차 예외 처리
        # 넓이 계산
        areas = self.calculate_areas(coor_list)

        # filtered_by_area -> 널이로 필터링 된 숫자, dots_and_minus -> '.'과 '-'
        filtered_by_area, dot_list, minus_list = self.filter_by_area(conf_list, coor_list, cls_list, areas, threshold, area_ratio)

        # x좌표와 y좌표로 다시 숫자 필터링
        filtered_by_coordinates = self.filter_numbers_by_coordinates(filtered_by_area)

        # '.', '-' 필터링
        filtered_dot = self.filter_dots_by_coordinates(dot_list, filtered_by_coordinates)
        filtered_minus = self.filter_minus_by_coordinates(minus_list, filtered_by_coordinates)

        final_filtered = []

        for item in filtered_by_coordinates + filtered_dot + filtered_minus:
            if item not in final_filtered:
                final_filtered.append(item)

        result = ''.join([self.cls_to_str[int(item[2])] for item in sorted(final_filtered, key=lambda x: x[1][0])])

        return result
