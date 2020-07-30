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
    path_save = '/home/ubuntu/hks/ocr/idcard_generator_project/datas/data_val_real/val_text_lines/'


    path = '/home/ubuntu/hks/ocr/idcard_generator_project/datas/data_val_real/val_without_logo/'
    # labels = csv.reader(
    #     open('/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/src/generate_labels1.csv'))
    font_template = json.load(open(
        '/home/ubuntu/hks/ocr/idcard_generator_project/idcard_generator/image_match/front.json'))
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
        '/home/ubuntu/hks/ocr/idcard_generator_project/idcard_generator/image_match/back1.json'))
    qianfajiguang1 = back_template['shapes'][1]['points']
    qianfajiguang2 = back_template['shapes'][2]['points']
    youxiaoqixian = back_template['shapes'][0]['points']


    images_name=os.listdir(path)
    for index,images_d in enumerate(images_name):
            print(index)
            images_name=os.listdir(path+'/'+images_d)
            if os.path.exists(path_save+images_d)==False:
                os.mkdir(path_save+images_d)
            for image_name in images_name:
                name=image_name.split('_')[0]
                ind=image_name.split('-')[-1].split('.jpg')[0]
                font_back_flag=image_name.split('_')[-1].split('-')[0]
                if font_back_flag=='0':
                    front=cv2.imread(path+name+'/'+image_name,0)
                    xingming_roi = front[int(xingming[0][1]):int(xingming[1][1]), int(xingming[0][0]):int(xingming[1][0])]
                    xingbie_roi = front[int(xingbie[0][1]):int(xingbie[1][1]), int(xingbie[0][0]):int(xingbie[1][0])]
                    mingzhu_roi = front[int(mingzhu[0][1]+3):int(mingzhu[1][1]-3), int(mingzhu[0][0]+10):int(mingzhu[1][0]+20)]
                    chusheng_year_roi = front[int(chusheng_year[0][1]):int(chusheng_year[1][1]), int(chusheng_year[0][0]):int(chusheng_year[1][0])]
                    chusheng_month_roi = front[int(chusheng_month[0][1]):int(chusheng_month[1][1]), int(chusheng_month[0][0]):int(chusheng_month[1][0])]
                    chusheng_day_roi = front[int(chusheng_day[0][1]):int(chusheng_day[1][1]), int(chusheng_day[0][0]):int(chusheng_day[1][0])]
                    dizhi_line1_roi = front[int(dizhi_line1[0][1]+2):int(dizhi_line1[1][1]-3), int(dizhi_line1[0][0]):int(dizhi_line1[1][0]-10)]
                    dizhi_line2_roi = front[int(dizhi_line2[0][1]+2):int(dizhi_line2[1][1]+1), int(dizhi_line2[0][0]):int(dizhi_line2[1][0])]
                    dizhi_line3_roi = front[int(dizhi_line3[0][1]+3-1):int(dizhi_line3[1][1]+1), int(dizhi_line3[0][0]):int(dizhi_line3[1][0])]
                    dizhi_line_roi=np.hstack([dizhi_line1_roi,dizhi_line2_roi,dizhi_line3_roi])
                    shengfengzhenghao_roi = front[int(shengfengzhenghao[0][1]):int(shengfengzhenghao[1][1]), int(shengfengzhenghao[0][0]+5):int(shengfengzhenghao[1][0])]

                    cv2.imwrite(path_save + images_d + '/'+'xingming_'+image_name,xingming_roi)
                    cv2.imwrite(path_save + images_d + '/'+'xingbie_'+image_name,xingbie_roi)
                    cv2.imwrite(path_save + images_d + '/' + 'mingzhu_'+image_name,mingzhu_roi)
                    cv2.imwrite(path_save + images_d + '/' + 'chushengyear_' + image_name, chusheng_year_roi)
                    cv2.imwrite(path_save + images_d + '/' + 'chushengmonth_' + image_name, chusheng_month_roi)
                    cv2.imwrite(path_save + images_d + '/' + 'chushengday_' + image_name, chusheng_day_roi)
                    cv2.imwrite(path_save + images_d + '/' + 'dizhi' + image_name, dizhi_line_roi)
                    cv2.imwrite(path_save + images_d + '/' + 'shengfengzhenghao_' + image_name, shengfengzhenghao_roi)
                    # cv2.imshow('xingming_roi',xingming_roi)
                    # cv2.imshow('xingbie_roi',xingbie_roi)
                    # cv2.imshow('mingzhu_roi', mingzhu_roi)
                    # cv2.imshow('chusheng_year_roi', chusheng_year_roi)
                    # cv2.imshow('chusheng_month_roi', chusheng_month_roi)
                    # cv2.imshow('chusheng_day_roi', chusheng_day_roi)
                    # # cv2.imshow('dizhi_line1_roi', dizhi_line1_roi)
                    # # cv2.imshow('dizhi_line2_roi', dizhi_line2_roi)
                    # # cv2.imshow('dizhi_line3_roi', dizhi_line3_roi)
                    # cv2.imshow('dizhi',dizhi_line_roi)
                    # cv2.imshow('shengfengzhenghao_roi', shengfengzhenghao_roi)
                    # cv2.waitKey(100)
                else:
                    back = cv2.imread(path +name+'/' +image_name, 0)
                    qianfajiguang1_roi=back[int(qianfajiguang1[0][1]+2):int(qianfajiguang1[1][1]-1), int(qianfajiguang1[0][0]):int(qianfajiguang1[1][0])]
                    qianfajiguang2_roi= back[int(qianfajiguang2[0][1]-2-1):int(qianfajiguang2[1][1]-2+1), int(qianfajiguang2[0][0]):int(qianfajiguang2[1][0])]
                    qianfajiguang_roi=np.hstack([qianfajiguang1_roi,qianfajiguang2_roi])
                    youxiaoqixian_roi = back[int(youxiaoqixian[0][1]):int(youxiaoqixian[1][1]), int(youxiaoqixian[0][0]):int(youxiaoqixian[1][0])]
                    cv2.imwrite(path_save + images_d + '/' + 'qianfajiguang_' + image_name, qianfajiguang_roi)
                    cv2.imwrite(path_save + images_d + '/' + 'youxiaoqixian_' + image_name, youxiaoqixian_roi)



                    # cv2.imshow('qianfajiguang1_roi',qianfajiguang1_roi)
                    # cv2.imshow('qianfajiguang2_roi',qianfajiguang2_roi)
                    # cv2.imshow('qianfajiguang_roi',qianfajiguang_roi)
                    # cv2.imshow('youxiaoqixian_roi',youxiaoqixian_roi)
                    # cv2.waitKey(100)





if __name__ == "__main__":

    gen_faker_card_run()

