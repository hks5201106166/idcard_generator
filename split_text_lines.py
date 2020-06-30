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
def dizhi_split(dizhi_rect):
    dizhi_texts=[]
    h_sum=np.sum(dizhi_rect[:,0:50],1)
    # cv2.imshow('dizhi_rect1', dizhi_rect[3:23,:])
    # cv2.imshow('dizhi_rect2', dizhi_rect[25:45, :])
    # cv2.imshow('dizhi_rect3', dizhi_rect[45:65, :])
    # cv2.imshow('dizhi_rect4',dizhi_rect[65:85,:])
    i=np.random.randint(0,10000)
    cv2.imwrite('/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/images/'+str(i)+'.jpg',dizhi_rect[65:85,:])
    cv2.waitKey(1000)
    # plt.plot(h_sum)
    # plt.show()
    #time.sleep(100000000)
    return dizhi_texts
path='/home/simple/mydemo/ocr_project/idcard_generator_project/generator_datas1/'
labels=csv.reader(open('/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/src/generate_labels1.csv'))
font_template=json.load(open('/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/split_text_template/0adyypn1yq_1.json'))
xingming = font_template['shapes'][0]['points']
xingbie  = font_template['shapes'][1]['points']
mingzhu  = font_template['shapes'][2]['points']
chusheng = font_template['shapes'][3]['points']
dizhi    = font_template['shapes'][4]['points']
shengfengzhenghao=font_template['shapes'][5]['points']
back_template=json.load(open('/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/split_text_template/0a754prsnl_0.json'))
qianfajiguang=back_template['shapes'][0]['points']
youxiaoqixian=back_template['shapes'][1]['points']
#files=os.listdir(path)
for label in labels:

    image_name_font=label[0]+'_1'+'.jpg'
    font=cv2.imread(path+image_name_font,0)
    xingming_rect=font[int(xingming[0][1]):int(xingming[1][1]),int(xingming[0][0]):int(xingming[1][0])]
    xingbie_rect=font[int(xingbie[0][1]):int(xingbie[1][1]),int(xingbie[0][0]):int(xingbie[1][0])]
    mingzhu_rect=font[int(mingzhu[0][1]):int(mingzhu[1][1]),int(mingzhu[0][0]):int(mingzhu[1][0])]
    chusheng_rect=font[int(chusheng[0][1]):int(chusheng[1][1]),int(chusheng[0][0]):int(chusheng[1][0])]
    dizhi_rect=font[int(dizhi[0][1]):int(dizhi[1][1]),int(dizhi[0][0]):int(dizhi[1][0])]
    dizhi_texts=dizhi_split(dizhi_rect)
    shengfengzhenghao_rect=font[int(shengfengzhenghao[0][1]):int(shengfengzhenghao[1][1]),int(shengfengzhenghao[0][0]):int(shengfengzhenghao[1][0])]
    image_name_back=label[0]+'_0'+'.jpg'


    back=cv2.imread(path+image_name_back,0)
    qianfajiguang_rect=back[int(qianfajiguang[0][1]):int(qianfajiguang[1][1]),int(qianfajiguang[0][0]):int(qianfajiguang[1][0])]
    youxiaoqixian_rect=back[int(youxiaoqixian[0][1]):int(youxiaoqixian[1][1]),int(youxiaoqixian[0][0]):int(youxiaoqixian[1][0])]
    # cv2.imshow('xingming_rect',xingming_rect)
    # cv2.imshow('xingbie_rect',xingbie_rect)
    # cv2.imshow('mingzhu_rect', mingzhu_rect)
    # cv2.imshow('chusheng_rect', chusheng_rect)
    # cv2.imshow('dizhi_rect', dizhi_rect)
    # cv2.imshow('shengfengzhenghao_rect', shengfengzhenghao_rect)
    # cv2.imshow('qianfajiguang_rect',qianfajiguang_rect)
    # cv2.imshow('shengfengzhenghao_rect',shengfengzhenghao_rect)
    # cv2.waitKey(0)