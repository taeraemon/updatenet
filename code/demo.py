# --------------------------------------------------------
# Demo for UpdateNet
# Licensed under The MIT License
# Written by tykim (tykim512 at snu.ac.kr)
# Reference from Qiang Wang (DaSiamRPN)
# --------------------------------------------------------
#!/usr/bin/python

import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join

from net_upd import SiamRPNBIG
from run_SiamRPN_upd import SiamRPN_init_upd, SiamRPN_track_upd

from updatenet import UpdateResNet

from utils import get_axis_aligned_bbox, cxy_wh_2_rect

import xml.etree.ElementTree as ET

# IoU 계산 함수
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # 겹치는 영역의 너비와 높이 계산
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # 각 박스의 면적 계산
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # IoU 계산
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Ground Truth를 XML 파일에서 읽어오는 함수
def read_ground_truth(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bndbox = root.find(".//bndbox")
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)

    # [x, y, width, height] 형식으로 변환
    return [xmin, ymin, xmax - xmin, ymax - ymin]


# load net
net = SiamRPNBIG()
net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNBIG.model')))
net.eval().cuda()

updatenet = UpdateResNet()
updatenet.load_state_dict(torch.load('../models/vot2016.pth.tar')['state_dict'])
updatenet.eval().cuda()

# image and init box
# image_files = sorted(glob.glob('./bag1/*.jpg'))
# init_rbox = [334.02, 128.36,
#              438.19, 188.78,
#              396.39, 260.83,
#              292.23, 200.41]

# image_files = sorted(glob.glob('./bag3/*.jpg'))
# init_rbox = [479, 358,
#              479, 369,
#              512, 369,
#              512, 358]

# image_files = sorted(glob.glob('./bag4/*.jpg'))
# init_rbox = [502, 253,
#              502, 286,
#              569, 286,
#              569, 253]

image_files = sorted(glob.glob('./bag5_ap7_iou/*.jpg'))
init_rbox = [502, 253,
             502, 286,
             569, 286,
             569, 253]


[cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)

# tracker init
target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
im = cv2.imread(image_files[0])  # HxWxC
state = SiamRPN_init_upd(im, target_pos, target_sz, net)

cv2.namedWindow("UpdateNet based Siamese", cv2.WINDOW_NORMAL)

total_iou = 0
iou_count = 0

# tracking and visualization
toc = 0
for f, image_file in enumerate(image_files):
    im = cv2.imread(image_file)
    tic = cv2.getTickCount()
    state = SiamRPN_track_upd(state, im, updatenet)  # track
    toc += cv2.getTickCount()-tic
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    res = [int(l) for l in res]
    cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
    
    # Ground Truth 읽어오기
    xml_file = image_file.replace(".jpg", ".xml")  # 이미지와 같은 이름의 XML 파일
    gt_box = read_ground_truth(xml_file)

    # 정확도 계산 (IoU)
    iou = calculate_iou(res, gt_box)
    total_iou += iou
    iou_count += 1
    
    
    cv2.imshow('UpdateNet based Siamese', im)
    
    if cv2.waitKey(50) & 0xFF == 27:
        break

# 평균 IoU 계산 및 추적 속도 출력
average_iou = total_iou / iou_count if iou_count > 0 else 0
print(f'Tracking Speed: {(len(image_files) - 1) / (toc / cv2.getTickFrequency()):.1f} fps')
print(f'Average IoU: {average_iou:.3f}')

cv2.destroyAllWindows()
