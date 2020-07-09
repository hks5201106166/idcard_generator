#-*-coding:utf-8-*-
#-*-coding:utf-8-*-
#-*-coding:utf-8-*-
import os
import cv2
import csv
import json
import matplotlib.pyplot as plt
import time
import numpy as np
#font={'xingming','xingbie','mingzhu','chusheng','dizhi','shengfengzhenghao'}
#back={'qianfajiguang','youxiaoqixian'}


import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import csv
import os
import os
import cv2
from sklearn.model_selection import train_test_split
import imgaug as ia
import imgaug.augmenters as iaa

def gen_faker_card_run():
    path_save = '/home/simple/mydemo/ocr_project/idcard_generator_project/split_text_idcard/'


    path = '/home/simple/mydemo/ocr_project/idcard_generator_project/test_results/data_split/'
    # labels = csv.reader(
    #     open('/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/src/generate_labels1.csv'))
    font_template = json.load(open(
        '/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/image_match/front.json'))
    xingming = font_template['shapes'][0]['points']
    xingbie = font_template['shapes'][1]['points']
    mingzhu = font_template['shapes'][2]['points']
    chusheng_year= font_template['shapes'][3]['points']
    chusheng_month=font_template['shapes'][4]['points']
    chusheng_day=font_template['shapes'][5]['points']
    dizhi_line1= font_template['shapes'][6]['points']
    dizhi_line2 = font_template['shapes'][7]['points']
    dizhi_line3=font_template['shapes'][8]['points']
    shengfengzhenghao = font_template['shapes'][9]['points']
    back_template = json.load(open(
        '/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/image_match/back1.json'))
    qianfajiguang1 = back_template['shapes'][1]['points']
    qianfajiguang2 = back_template['shapes'][2]['points']
    youxiaoqixian = back_template['shapes'][0]['points']


    images_name=os.listdir(path)
    for image_name in images_name:
            name=image_name.split('_')[-1]
            if name=='0.jpg':
                front=cv2.imread(path+image_name,0)
                xingming_roi = front[int(xingming[0][1]):int(xingming[1][1]), int(xingming[0][0]):int(xingming[1][0])]
                xingbie_roi = front[int(xingbie[0][1]):int(xingbie[1][1]), int(xingbie[0][0]):int(xingbie[1][0])]
                mingzhu_roi = front[int(mingzhu[0][1]):int(mingzhu[1][1]), int(mingzhu[0][0]):int(mingzhu[1][0])]
                chusheng_year_roi = front[int(chusheng_year[0][1]):int(chusheng_year[1][1]), int(chusheng_year[0][0]):int(chusheng_year[1][0])]
                chusheng_month_roi = front[int(chusheng_month[0][1]):int(chusheng_month[1][1]), int(chusheng_month[0][0]):int(chusheng_month[1][0])]
                chusheng_day_roi = front[int(chusheng_day[0][1]):int(chusheng_day[1][1]), int(chusheng_day[0][0]):int(chusheng_day[1][0])]
                dizhi_line1_roi = front[int(dizhi_line1[0][1]):int(dizhi_line1[1][1]), int(dizhi_line1[0][0]):int(dizhi_line1[1][0])]
                dizhi_line2_roi = front[int(dizhi_line2[0][1]+2):int(dizhi_line2[1][1]), int(dizhi_line2[0][0]):int(dizhi_line2[1][0])]
                dizhi_line3_roi = front[int(dizhi_line3[0][1]+3):int(dizhi_line3[1][1]), int(dizhi_line3[0][0]):int(dizhi_line3[1][0])]
                shengfengzhenghao_roi = front[int(shengfengzhenghao[0][1]):int(shengfengzhenghao[1][1]), int(shengfengzhenghao[0][0]+5):int(shengfengzhenghao[1][0])]
                # cv2.imshow('xingming_roi',xingming_roi)
                # cv2.imshow('xingbie_roi',xingbie_roi)
                # cv2.imshow('mingzhu_roi', mingzhu_roi)
                # cv2.imshow('chusheng_year_roi', chusheng_year_roi)
                # cv2.imshow('chusheng_month_roi', chusheng_month_roi)
                # cv2.imshow('chusheng_day_roi', chusheng_day_roi)
                # cv2.imshow('dizhi_line1_roi', dizhi_line1_roi)
                # cv2.imshow('dizhi_line2_roi', dizhi_line2_roi)
                # cv2.imshow('dizhi_line3_roi', dizhi_line3_roi)
                # cv2.imshow('shengfengzhenghao_roi', shengfengzhenghao_roi)
                # cv2.waitKey(1000)
            else:
                back = cv2.imread(path + image_name, 0)
                qianfajiguang1_roi=back[int(qianfajiguang1[0][1]+3):int(qianfajiguang1[1][1]), int(qianfajiguang1[0][0]):int(qianfajiguang1[1][0])]
                qianfajiguang2_roi= back[int(qianfajiguang2[0][1]-2):int(qianfajiguang2[1][1]-2), int(qianfajiguang2[0][0]):int(qianfajiguang2[1][0])]
                youxiaoqixian_roi = back[int(youxiaoqixian[0][1]+3):int(youxiaoqixian[1][1]), int(youxiaoqixian[0][0]):int(youxiaoqixian[1][0])]
                cv2.imshow('qianfajiguang1_roi',qianfajiguang1_roi)
                cv2.imshow('qianfajiguang2_roi',qianfajiguang2_roi)
                cv2.imshow('youxiaoqixian_roi',youxiaoqixian_roi)
                cv2.waitKey(1000)





if __name__ == "__main__":

    gen_faker_card_run()

