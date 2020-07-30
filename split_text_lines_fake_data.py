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
    path_save = '/home/ubuntu/hks/ocr/idcard_generator_project/datas/data_train_fake/train_text_lines/'


    path = '/home/ubuntu/hks/ocr/idcard_generator_project/datas/data_train_fake/train_without_logo/'
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
                if font_back_flag=='1':
                    front=cv2.imread(path+name+'/'+image_name,0)

                    xingming_random_rows=np.random.randint(-4,4)
                    xingming_random_cols=np.random.randint(-5,5)
                    xingming_roi = front[int(xingming[0][1]+xingming_random_rows):int(xingming[1][1]+xingming_random_rows),
                                   int(xingming[0][0]+xingming_random_cols):int(xingming[1][0]+xingming_random_cols)]

                    xingbie_random_rows = np.random.randint(-4, 4)
                    xingbie_random_cols = np.random.randint(-5, 7)
                    xingbie_roi = front[int(xingbie[0][1]+xingbie_random_rows):int(xingbie[1][1]+xingbie_random_rows),
                                  int(xingbie[0][0]+xingming_random_cols):int(xingbie[1][0]+xingming_random_cols)]

                    mingzhu_random_rows = np.random.randint(-4, 4)
                    mingzhu_random_cols = np.random.randint(-5, 5)
                    mingzhu_roi = front[int(mingzhu[0][1]+3+mingzhu_random_rows):int(mingzhu[1][1]-3+mingzhu_random_rows),
                                  int(mingzhu[0][0]+10+mingzhu_random_cols):int(mingzhu[1][0]+20)+mingzhu_random_cols]

                    chusheng_year_random_rows = np.random.randint(-4, 4)
                    chusheng_year_random_cols = np.random.randint(0, 4)
                    chusheng_year_roi = front[int(chusheng_year[0][1]+chusheng_year_random_rows):int(chusheng_year[1][1]+chusheng_year_random_rows),
                                        int(chusheng_year[0][0]+chusheng_year_random_cols):int(chusheng_year[1][0])+chusheng_year_random_cols]

                    chusheng_month_random_rows = np.random.randint(-4, 4)
                    chusheng_month_random_cols = np.random.randint(-3, 3)
                    chusheng_month_roi = front[int(chusheng_month[0][1]+chusheng_month_random_rows):int(chusheng_month[1][1]+chusheng_month_random_rows),
                                         int(chusheng_month[0][0]+chusheng_month_random_cols):int(chusheng_month[1][0]+chusheng_month_random_cols)]

                    chusheng_day_random_rows = np.random.randint(-4, 4)
                    chusheng_day_random_cols = np.random.randint(-3, 3)
                    chusheng_day_roi = front[int(chusheng_day[0][1]+chusheng_day_random_rows):int(chusheng_day[1][1]+chusheng_day_random_rows),
                                       int(chusheng_day[0][0]+chusheng_day_random_cols):int(chusheng_day[1][0]+chusheng_day_random_cols)]

                    dizhi_line1_col=np.random.randint(-5,5)
                    dizhi_line1_row=np.random.randint(-2,5)
                    dizhi_line1_roi = front[int(dizhi_line1[0][1]+2+dizhi_line1_row):int(dizhi_line1[1][1]-3+dizhi_line1_row),
                                      int(dizhi_line1[0][0]+dizhi_line1_col):int(dizhi_line1[1][0]-10+dizhi_line1_col)]

                    dizhi_line2_col = np.random.randint(-5, 5)
                    dizhi_line2_row = np.random.randint(-2, 2)
                    dizhi_line2_roi = front[int(dizhi_line2[0][1]+2+dizhi_line2_row):int(dizhi_line2[1][1]+1+dizhi_line2_row),
                                      int(dizhi_line2[0][0]+dizhi_line2_col):int(dizhi_line2[1][0]+dizhi_line2_col)]

                    dizhi_line3_roi = front[int(dizhi_line3[0][1]+3-1):int(dizhi_line3[1][1]+1),
                                      int(dizhi_line3[0][0]):int(dizhi_line3[1][0])]
                    dizhi_line_roi=np.hstack([dizhi_line1_roi,dizhi_line2_roi,dizhi_line3_roi])

                    shengfengzhenghao_col = np.random.randint(-5, 5)
                    shengfengzhenghao_row = np.random.randint(-2, 2)
                    shengfengzhenghao_roi = front[int(shengfengzhenghao[0][1]+shengfengzhenghao_row):int(shengfengzhenghao[1][1]+shengfengzhenghao_row),
                                            int(shengfengzhenghao[0][0]+5+shengfengzhenghao_col):int(shengfengzhenghao[1][0]+shengfengzhenghao_col)]

                    # cv2.imwrite(path_save + images_d + '/'+'xingming_'+image_name,xingming_roi)
                    # cv2.imwrite(path_save + images_d + '/'+'xingbie_'+image_name,xingbie_roi)
                    # cv2.imwrite(path_save + images_d + '/' + 'mingzhu_'+image_name,mingzhu_roi)
                    # cv2.imwrite(path_save + images_d + '/' + 'chushengyear_' + image_name, chusheng_year_roi)
                    # cv2.imwrite(path_save + images_d + '/' + 'chushengmonth_' + image_name, chusheng_month_roi)
                    # cv2.imwrite(path_save + images_d + '/' + 'chushengday_' + image_name, chusheng_day_roi)
                    # cv2.imwrite(path_save + images_d + '/' + 'dizhi' + image_name, dizhi_line_roi)
                    # cv2.imwrite(path_save + images_d + '/' + 'shengfengzhenghao_' + image_name, shengfengzhenghao_roi)
                    cv2.imshow('xingming_roi',xingming_roi)
                    cv2.imshow('xingbie_roi',xingbie_roi)
                    cv2.imshow('mingzhu_roi', mingzhu_roi)
                    cv2.imshow('chusheng_year_roi', chusheng_year_roi)
                    cv2.imshow('chusheng_month_roi', chusheng_month_roi)
                    cv2.imshow('chusheng_day_roi', chusheng_day_roi)
                    # cv2.imshow('dizhi_line1_roi', dizhi_line1_roi)
                    # cv2.imshow('dizhi_line2_roi', dizhi_line2_roi)
                    # cv2.imshow('dizhi_line3_roi', dizhi_line3_roi)
                    cv2.imshow('dizhi',dizhi_line_roi)
                    cv2.imshow('shengfengzhenghao_roi', shengfengzhenghao_roi)
                    cv2.waitKey(100)
                else:
                    back = cv2.imread(path +name+'/' +image_name, 0)

                    qianfajiguang1_random_rows = np.random.randint(-2, 2)
                    qianfajiguang1_random_cols = np.random.randint(-3, 3)
                    qianfajiguang1_roi=back[int(qianfajiguang1[0][1]+2+qianfajiguang1_random_rows):int(qianfajiguang1[1][1]-1+qianfajiguang1_random_rows),
                                       int(qianfajiguang1[0][0]-5+qianfajiguang1_random_cols):int(qianfajiguang1[1][0]+qianfajiguang1_random_cols)]

                    qianfajiguang2_roi= back[int(qianfajiguang2[0][1]-2-1):int(qianfajiguang2[1][1]-2+1), int(qianfajiguang2[0][0]):int(qianfajiguang2[1][0])]
                    qianfajiguang_roi=np.hstack([qianfajiguang1_roi,qianfajiguang2_roi])

                    youxiaoqixian_random_rows = np.random.randint(-2, 2)
                    youxiaoqixian_random_cols = np.random.randint(-2, 2)
                    youxiaoqixian_roi = back[int(youxiaoqixian[0][1]+youxiaoqixian_random_rows):int(youxiaoqixian[1][1]+youxiaoqixian_random_rows),
                                        int(youxiaoqixian[0][0]+youxiaoqixian_random_cols):int(youxiaoqixian[1][0]+youxiaoqixian_random_cols)]
                    # cv2.imwrite(path_save + images_d + '/' + 'qianfajiguang_' + image_name, qianfajiguang_roi)
                    # cv2.imwrite(path_save + images_d + '/' + 'youxiaoqixian_' + image_name, youxiaoqixian_roi)



                    cv2.imshow('qianfajiguang1_roi',qianfajiguang1_roi)
                    cv2.imshow('qianfajiguang2_roi',qianfajiguang2_roi)
                    cv2.imshow('qianfajiguang_roi',qianfajiguang_roi)
                    cv2.imshow('youxiaoqixian_roi',youxiaoqixian_roi)
                    cv2.waitKey(100)





if __name__ == "__main__":

    gen_faker_card_run()

