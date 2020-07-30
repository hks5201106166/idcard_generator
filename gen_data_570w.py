#/home/ubuntu/hks/ocr/idcard_generator_project/datas/data_570w/data_570w

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import csv
import os
import json
from multiprocessing import Pool
# 伪造正面
def gen_card_front(img, str):
    font1 = os.path.join(ori_path, 'src/msyh.ttc')
    font2 = os.path.join(ori_path, 'src/msyh.ttc')
    color_blue = (0, 191, 255)
    color_black = (0, 0, 0)

    img_res = img_put_text(img, '姓  名', 38, 53, font1, color_blue, 13)
    img_res = img_put_text(img_res, str[0], 92, 49, font1, color_black, 19)
    img_res = img_put_text(img_res, '性  别', 38, 86, font1, color_blue, 13)
    img_res = img_put_text(img_res, str[1], 90, 81, font1, color_black, 19)
    img_res = img_put_text(img_res, '民  族', 136, 86, font1, color_blue, 13)
    img_res = img_put_text(img_res, str[2], 191, 81, font1, color_black, 19)
    img_res = img_put_text(img_res, '出  生', 38, 118, font1, color_blue, 13)
    img_res = img_put_text(img_res, str[3], 89, 114, font1, color_black, 19)
    img_res = img_put_text(img_res, '年', 136, 118, font1, color_blue, 13)
    img_res = img_put_text(img_res, str[4], 154, 114, font1, color_black, 19)
    img_res = img_put_text(img_res, '月', 181, 118, font1, color_blue, 13)
    img_res = img_put_text(img_res, str[5], 204, 114, font1, color_black, 19)
    img_res = img_put_text(img_res, '日', 226, 118, font1, color_blue, 13)
    img_res = img_put_text(img_res, '住  址', 38, 150, font1, color_blue, 13)

    shenfen_list = ['公', '民', '身', '份', '号', '码']
    for i in range(0, len(shenfen_list)):
        img_res = img_put_text(img_res, shenfen_list[i], 38 + i * 14, 235, font1, color_blue, 13)

    if len(str[6]) > 12:
        addr_list1 = list(str[6][0:12])
        for i in range(0, len(addr_list1)):
            img_res = img_put_text(img_res, addr_list1[i], 87 + i * 17, 146, font1, color_black, 19)

        if len(str[6]) > 24:
            addr_list2 = list(str[6][12:24])
            for i in range(0, len(addr_list2)):
                img_res = img_put_text(img_res, addr_list2[i], 87 + i * 17, 168, font1, color_black, 19)

            addr_list3 = list(str[6][24:len(str[6])])
            for i in range(0, len(addr_list3)):
                img_res = img_put_text(img_res, addr_list3[i], 87 + i * 17, 190, font1, color_black, 19)

        else:
            addr_list2 = list(str[6][12:len(str[6])])
            for i in range(0, len(addr_list2)):
                img_res = img_put_text(img_res, addr_list2[i], 87 + i * 17, 168, font1, color_black, 19)

    else:
        addr_list = list(str[6])
        for i in range(0, len(addr_list)):
            img_res = img_put_text(img_res, addr_list[i], 87 + i * 17, 146, font1, color_black, 19)

    id_list = list(str[7])
    for i in range(0, len(id_list)):
        img_res = img_put_text(img_res, id_list[i], 136 + i * 13.4, 227, font2, color_black, 19)

    return img_res


# 伪造背面
def gen_card_back(img, str):
    font1 = os.path.join(ori_path, 'src/msyh.ttc')

    color_black = (0, 0, 0)
    img_res = img_put_text(img, '签发机关', 108, 202, font1, color_black, 15)

    if len(str[0]) > 12:
        institu_list = list(str[0][0:12])
        for i in range(0, len(institu_list)):
            img_res = img_put_text(img_res, institu_list[i][0:12], 175 + i * 17, 200, font1, color_black, 19)

        institu_list2 = list(str[0][12:len(str[0])])
        for i in range(0, len(institu_list2)):
            img_res = img_put_text(img_res, institu_list2[i][0:12], 175 + i * 17, 219, font1, color_black, 19)

    else:
        institu_list = list(str[0])
        for i in range(0, len(institu_list)):
            img_res = img_put_text(img_res, institu_list[i], 175 + i * 17, 200, font1, color_black, 19)

    img_res = img_put_text(img_res, '有效期限', 108, 236, font1, color_black, 15)

    date_list1 = list(str[1][0:4])
    for i in range(0, len(date_list1)):
        img_res = img_put_text(img_res, date_list1[i], 176 + i * 10, 233, font1, color_black, 19)

    img_res = img_put_text(img_res, '.', 215, 233, font1, color_black, 19)

    date_list2 = list(str[1][5:7])
    for i in range(0, len(date_list2)):
        img_res = img_put_text(img_res, date_list2[i], 219 + i * 10, 233, font1, color_black, 19)

    img_res = img_put_text(img_res, '.', 239, 233, font1, color_black, 19)

    date_list3 = list(str[1][8:10])
    for i in range(0, len(date_list3)):
        img_res = img_put_text(img_res, date_list3[i], 243 + i * 10, 233, font1, color_black, 19)

    img_res = img_put_text(img_res, '-', 263, 235, font1, color_black, 15)

    # 如果最后是数字
    str_temp = str[1][11:len(str[1])]
    if str_temp != '长期':

        str_temp_list1 = list(str_temp[0:4])
        for i in range(0, len(str_temp_list1)):
            img_res = img_put_text(img_res, str_temp_list1[i], 270 + i * 10, 233, font1, color_black, 19)

        img_res = img_put_text(img_res, '.', 309, 233, font1, color_black, 19)

        str_temp_list2 = list(str_temp[5:7])
        for i in range(0, len(str_temp_list2)):
            img_res = img_put_text(img_res, str_temp_list2[i], 313 + i * 10, 233, font1, color_black, 19)

        img_res = img_put_text(img_res, '.', 333, 233, font1, color_black, 19)

        str_temp_list3 = list(str_temp[8:10])
        for i in range(0, len(str_temp_list3)):
            img_res = img_put_text(img_res, str_temp_list3[i], 337 + i * 10, 233, font1, color_black, 19)

    else:
        img_res = img_put_text(img_res, '长期', 271.5, 233, font1, color_black, 18)

    return img_res


# 生成画布
def img_to_white(img):
    h = img.shape[0]
    w = img.shape[1]
    target = np.ones((h, w), dtype=np.uint8) * 255
    ret = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)

    for i in range(h):
        for j in range(w):
            ret[i, j, 0] = img[i, j, 0]
            ret[i, j, 1] = img[i, j, 1]
            ret[i, j, 2] = img[i, j, 2]
    return ret


# 在画布上书写文字
def img_put_text(img, str, pos_x, pos_y, font, color, size):
    img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    mfont = ImageFont.truetype(font, size)
    fillColor = color
    position = (pos_x, pos_y)
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, str, font=mfont, fill=fillColor,stroke_width=0,spacing=400)
    img_OpenCV = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img_OpenCV


def gen_faker_card_run(one_line):
    path_save='/home/ubuntu/hks/ocr/idcard_generator_project/datas/data_570w/data_570w/'
    font_template = json.load(open(
        '/home/ubuntu/hks/ocr/idcard_generator_project/idcard_generator/image_match/front.json'))
    xingming = font_template['shapes'][0]['points']
    xingbie = font_template['shapes'][1]['points']
    mingzhu = font_template['shapes'][2]['points']
    chusheng_year = font_template['shapes'][3]['points']
    chusheng_month = font_template['shapes'][4]['points']
    chusheng_day = font_template['shapes'][5]['points']
    dizhi_line1 = font_template['shapes'][6]['points']
    dizhi_line2 = font_template['shapes'][7]['points']
    dizhi_line3 = font_template['shapes'][8]['points']
    shengfengzhenghao = font_template['shapes'][9]['points']
    back_template = json.load(open(
        '/home/ubuntu/hks/ocr/idcard_generator_project/idcard_generator/image_match/back1.json'))
    qianfajiguang1 = back_template['shapes'][1]['points']
    qianfajiguang2 = back_template['shapes'][2]['points']
    youxiaoqixian = back_template['shapes'][0]['points']
    # date = []  # 创建列表准备接收csv各行数据
    cnt = 0  # 记录csv文件行数
    date=[]
    date.append(one_line)
    image_name=date[cnt][0]

    result_front = []
    result_back = []

    result_front.append(date[cnt][1])  # 姓名
    result_front.append(date[cnt][3])  # 性别
    result_front.append(date[cnt][2])  # 名族
    result_front.append(date[cnt][4])  # 年
    result_front.append(date[cnt][5])  # 月
    result_front.append(date[cnt][6])  # 日
    result_front.append(date[cnt][7])  # 地址
    result_front.append(date[cnt][8])  # 身份号

    result_back.append(date[cnt][9])  # 签发机关f
    result_back.append(date[cnt][10])  # 有效日期

    image1 = cv2.imread(front_img)  # 读取正面模板
    image2 = cv2.imread(back_img)  # 读取背面模板

    #img_new_white1 = img_to_white(image1)
    img_new_white1 = image1
    # cv2.imshow('hjs',img_new_white1)
    # 生成画布
    front = gen_card_front(img_new_white1, result_front)
    front= cv2.cvtColor(front,cv2.COLOR_BGR2GRAY)


    xingming_random_rows = np.random.randint(-2, 2)
    xingming_random_cols = np.random.randint(-5, 5)
    xingming_roi = front[int(xingming[0][1]+10 + xingming_random_rows):int(xingming[1][1]-3 + xingming_random_rows),
                   int(xingming[0][0] + xingming_random_cols):int(xingming[1][0]-40 + xingming_random_cols)]

    xingbie_random_rows = np.random.randint(-4, 4)
    xingbie_random_cols = np.random.randint(-5, 5)
    xingbie_roi = front[int(xingbie[0][1]+8 + xingbie_random_rows):int(xingbie[1][1] + xingbie_random_rows),
                  int(xingbie[0][0] + xingming_random_cols):int(xingbie[1][0] + xingming_random_cols)]

    mingzhu_random_rows = np.random.randint(-4, 4)
    mingzhu_random_cols = np.random.randint(-5, 5)
    mingzhu_roi = front[int(mingzhu[0][1] + 3 + mingzhu_random_rows):int(mingzhu[1][1] - 3 + mingzhu_random_rows),
                  int(mingzhu[0][0] + 10 + mingzhu_random_cols):int(mingzhu[1][0] + 20) + mingzhu_random_cols]

    chusheng_year_random_rows = np.random.randint(-2,2)
    chusheng_year_random_cols = np.random.randint(0, 4)
    chusheng_year_roi = front[int(chusheng_year[0][1] + chusheng_year_random_rows):int(
        chusheng_year[1][1] + chusheng_year_random_rows),
                        int(chusheng_year[0][0] + chusheng_year_random_cols):int(
                            chusheng_year[1][0]) + chusheng_year_random_cols]

    chusheng_month_random_rows = np.random.randint(-2, 2)
    chusheng_month_random_cols = np.random.randint(-2, 2)
    chusheng_month_roi = front[int(chusheng_month[0][1] + chusheng_month_random_rows):int(
        chusheng_month[1][1] + chusheng_month_random_rows),
                         int(chusheng_month[0][0] + chusheng_month_random_cols):int(
                             chusheng_month[1][0] + chusheng_month_random_cols)]

    chusheng_day_random_rows = np.random.randint(-2, 2)
    chusheng_day_random_cols = np.random.randint(-2, 2)
    chusheng_day_roi = front[int(chusheng_day[0][1] + chusheng_day_random_rows):int(
        chusheng_day[1][1] + chusheng_day_random_rows),
                       int(chusheng_day[0][0] + chusheng_day_random_cols):int(
                           chusheng_day[1][0] + chusheng_day_random_cols)]

    dizhi_line1_col = np.random.randint(-1, 1)
    dizhi_line1_row = np.random.randint(-1, 1)
    dizhi_line1_roi = front[
                      int(dizhi_line1[0][1]+7 + dizhi_line1_row):int(dizhi_line1[1][1]+2  + dizhi_line1_row),
                      int(dizhi_line1[0][0] + dizhi_line1_col):int(dizhi_line1[1][0] - 10 + dizhi_line1_col)]

    dizhi_line2_col = np.random.randint(-1, 1)
    dizhi_line2_row = np.random.randint(-1, 1)
    dizhi_line2_roi = front[
                      int(dizhi_line2[0][1] + 2+2 + dizhi_line2_row):int(dizhi_line2[1][1] + 1+2 + dizhi_line2_row),
                      int(dizhi_line2[0][0] + dizhi_line2_col):int(dizhi_line2[1][0] + dizhi_line2_col)]

    dizhi_line3_roi = front[int(dizhi_line3[0][1] + 3 - 1):int(dizhi_line3[1][1] + 1),
                      int(dizhi_line3[0][0]):int(dizhi_line3[1][0])]
    dizhi_line_roi = np.hstack([dizhi_line1_roi, dizhi_line2_roi, dizhi_line3_roi])

    shengfengzhenghao_col = np.random.randint(-2, 2)
    shengfengzhenghao_row = np.random.randint(-2, 2)
    shengfengzhenghao_roi = front[int(shengfengzhenghao[0][1]+3 + shengfengzhenghao_row):int(
        shengfengzhenghao[1][1]-3 + shengfengzhenghao_row),
                            int(shengfengzhenghao[0][0] + 5 + shengfengzhenghao_col):int(
                                shengfengzhenghao[1][0] + shengfengzhenghao_col)]

    cv2.imwrite(path_save  + '/'+'xingming_'+image_name+'.jpg',xingming_roi)
    cv2.imwrite(path_save  + '/'+'xingbie_'+image_name+'.jpg',xingbie_roi)
    cv2.imwrite(path_save  + '/' + 'mingzhu_'+image_name+'.jpg',mingzhu_roi)
    cv2.imwrite(path_save  + '/' + 'chushengyear_' + image_name+'.jpg', chusheng_year_roi)
    cv2.imwrite(path_save  + '/' + 'chushengmonth_' + image_name+'.jpg', chusheng_month_roi)
    cv2.imwrite(path_save  + '/' + 'chushengday_' + image_name+'.jpg', chusheng_day_roi)
    cv2.imwrite(path_save  + '/' + 'dizhi' + image_name+'.jpg', dizhi_line_roi)
    cv2.imwrite(path_save  + '/' + 'shengfengzhenghao_' + image_name+'.jpg', shengfengzhenghao_roi)
    # cv2.imshow('xingming_roi', xingming_roi)
    # cv2.imshow('xingbie_roi', xingbie_roi)
    # cv2.imshow('mingzhu_roi', mingzhu_roi)
    # cv2.imshow('chusheng_year_roi', chusheng_year_roi)
    # cv2.imshow('chusheng_month_roi', chusheng_month_roi)
    # cv2.imshow('chusheng_day_roi', chusheng_day_roi)
    # # cv2.imshow('dizhi_line1_roi', dizhi_line1_roi)
    # # cv2.imshow('dizhi_line2_roi', dizhi_line2_roi)
    # # cv2.imshow('dizhi_line3_roi', dizhi_line3_roi)
    # cv2.imshow('dizhi', dizhi_line_roi)
    # cv2.imshow('shengfengzhenghao_roi', shengfengzhenghao_roi)
    # cv2.waitKey(100)






    # 写入文字
    # cv2.imshow('font',img_res_f)
    # cv2.imwrite(result_card_path + '/{}_1.jpg'.format(image_name), img_res_f)

    #img_new_white2 = img_to_white(image2)
    img_new_white2 = image2
    back = gen_card_back(img_new_white2, result_back)
    back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)

    qianfajiguang1_random_rows = np.random.randint(-2, 2)
    qianfajiguang1_random_cols = np.random.randint(-3, 3)
    qianfajiguang1_roi = back[int(qianfajiguang1[0][1] + 5 + qianfajiguang1_random_rows):int(
        qianfajiguang1[1][1] +2 + qianfajiguang1_random_rows),
                         int(qianfajiguang1[0][0] - 5 + qianfajiguang1_random_cols):int(
                             qianfajiguang1[1][0] + qianfajiguang1_random_cols)]

    qianfajiguang2_roi = back[int(qianfajiguang2[0][1]):int(qianfajiguang2[1][1] +2),
                         int(qianfajiguang2[0][0]):int(qianfajiguang2[1][0])]
    qianfajiguang_roi = np.hstack([qianfajiguang1_roi, qianfajiguang2_roi])

    youxiaoqixian_random_rows = np.random.randint(-1, 1)
    youxiaoqixian_random_cols = np.random.randint(-1, 1)
    youxiaoqixian_roi = back[int(youxiaoqixian[0][1] + youxiaoqixian_random_rows):int(
        youxiaoqixian[1][1] + youxiaoqixian_random_rows),
                        int(youxiaoqixian[0][0] + youxiaoqixian_random_cols):int(
                            youxiaoqixian[1][0] + youxiaoqixian_random_cols)]
    cv2.imwrite(path_save  + '/' + 'qianfajiguang_' + image_name+'.jpg', qianfajiguang_roi)
    cv2.imwrite(path_save  + '/' + 'youxiaoqixian_' + image_name+'.jpg', youxiaoqixian_roi)

    # cv2.imshow('qianfajiguang1_roi', qianfajiguang1_roi)
    # cv2.imshow('qianfajiguang2_roi', qianfajiguang2_roi)
    # cv2.imshow('qianfajiguang_roi', qianfajiguang_roi)
    # cv2.imshow('youxiaoqixian_roi', youxiaoqixian_roi)
    # cv2.waitKey(100)
    # cv2.imshow('back',img_res_b)
    # cv2.imwrite(result_card_path + '/{}_0.jpg'.format(image_name), img_res_b)
    # cnt = cnt + 1
    # print(cnt)
    # cv2.waitKey(100)


if __name__ == "__main__":
    ori_path = os.path.abspath('.')
    front_img = os.path.join(ori_path, 'src/front.png')  # 正面模板
    back_img = os.path.join(ori_path, 'src/back.png')  # 背面模板
    result_card_path = os.path.join(ori_path, r'generator_datas1')  # 伪造原始身份证样本结果路径
    # csv_files = ["Train_Labels.csv", "Train_Labels2.csv"]
    ori_csv_file = os.path.join(ori_path, r'src', "data_570w.csv")  # 原始数据
    csv_file = open(ori_csv_file, 'r', encoding='UTF-8')
    csv_reader_lines = list(csv.reader(csv_file))  # 逐行读取csv文件


    pool = Pool(8)
    pool.map(gen_faker_card_run, csv_reader_lines)
    # pool.close()
    # pool.join()
    print('done')

    # gen_faker_card_run()
