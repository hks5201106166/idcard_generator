import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import csv
import os
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import imgaug as ia
import imgaug.augmenters as iaa
# 伪造正面
def gen_card_front(img, str):
    font1 = os.path.join(ori_path, 'src/black.TTF')
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
    font1 = os.path.join(ori_path, 'src/black.TTF')

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
    draw.text(position, str, font=mfont, fill=fillColor)
    img_OpenCV = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img_OpenCV


def gen_faker_card_run():
    aug_brightness = iaa.MultiplyBrightness((0.5, 1.))
    aug_gaussian =iaa.GaussianBlur((0, 2.0))

    # blur images with a sigma between 0 and 3.0
    csv_file = open(ori_csv_file, 'r', encoding='UTF-8')
    csv_reader_lines = list(csv.reader(csv_file))
    csv_reader_lines_train,csv_reader_lines_val=train_test_split(csv_reader_lines,test_size=0.005, random_state=0)# 逐行读取csv文件
    date = []  # 创建列表准备接收csv各行数据
    cnt = 0  # 记录csv文件行数
    path='/home/ubuntu/hks/ocr/idcard_generator_project/idcard_generator/template/'
    files = os.listdir(os.path.join(path, 'fuzhiwuxiao_mask'))
    for one_line in csv_reader_lines_train:
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

            result_back.append(date[cnt][9])  # 签发机关
            result_back.append(date[cnt][10])  # 有效日期

            image1 = cv2.imread(front_img)  # 读取正面模板
            image2 = cv2.imread(back_img)  # 读取背面模板

            #img_new_white1 = img_to_white(image1)
            img_new_white1 = image1
            # cv2.imshow('hjs',img_new_white1)
            # cv2.waitKey(0)# 生成画布
            img_res_f = gen_card_front(img_new_white1, result_front)
            img_res_f = cv2.cvtColor(img_res_f,cv2.COLOR_BGR2GRAY)
            # 写入文字
            #cv2.imwrite(result_card_path + '/{}_1.jpg'.format(image_name), img_res_f)

            #img_new_white2 = img_to_white(image2)
            img_new_white2 = image2
            img_res_b = gen_card_back(img_new_white2, result_back)
            img_res_b = cv2.cvtColor(img_res_b, cv2.COLOR_BGR2GRAY)
            #cv2.imwrite(result_card_path + '/{}_0.jpg'.format(image_name), img_res_b)
            cnt = cnt + 1
            print(cnt)

            for i in range(1):
                l = len(files)
                index = np.random.randint(0, l)

                image_gen_copy = img_res_f.copy()
                mask = cv2.imread(os.path.join(path + 'fuzhiwuxiao_mask', files[index]), 0)
                mask[mask > 150] = 255
                mask[mask <= 150] = 0
                image = cv2.imread(os.path.join(path + 'fuzhiwuxiao_template', files[index]), 0)
                template = cv2.bitwise_and(image, image, mask=mask)

                h_image_gen, w_image_gen= img_res_f.shape
                point_x, point_y = h_image_gen, w_image_gen
                h, w = template.shape
                while point_x + h >= h_image_gen or point_y + w >= w_image_gen:
                    point_x, point_y = np.random.randint(0, h_image_gen), np.random.randint(0, w_image_gen)

                rect = image_gen_copy[point_x:(h + point_x), point_y:(w + point_y)]
                rect1 = cv2.bitwise_and(rect, rect, mask=mask)
                mask = cv2.bitwise_not(mask)
                rect2 = cv2.bitwise_and(rect, rect, mask=mask)
                image_with_logo = cv2.addWeighted(rect1, 0.3, template, 0.3, 0)
                add_logo = image_with_logo + rect2

                image_gen_copy_logo = image_gen_copy.copy()
                image_gen_copy_logo[point_x:(h + point_x), point_y:(w + point_y)] = add_logo

                image_gen_copy_logo=np.stack([image_gen_copy_logo,image_gen_copy_logo,image_gen_copy_logo],axis=2)
                aug_brightness_deterministic= aug_brightness.to_deterministic()
                aug_gaussian_deterministic=aug_gaussian.to_deterministic()
                image_gen_copy_logo=aug_brightness_deterministic.augment_images(images=[image_gen_copy_logo])[0]
                image_gen_copy_logo=aug_gaussian_deterministic.augment_images(images=[image_gen_copy_logo])[0]

                image_gen_copy = np.stack([image_gen_copy, image_gen_copy, image_gen_copy], axis=2)
                image_gen_copy = aug_brightness_deterministic.augment_images(images=[image_gen_copy])[0]
                #image_gen_copy = aug_gaussian_deterministic.augment_images(images=[image_gen_copy])[0]
                #image_gen_copy=seq_nologo(image=np.stack([image_gen_copy,image_gen_copy,image_gen_copy],axis=2))

                train_data = np.hstack((image_gen_copy_logo,image_gen_copy))
                #image_gen_copy[point_x:(h + point_x), point_y:(w + point_y)] = add_logo[:,:]
                cv2.imwrite('/home/ubuntu/hks/ocr/idcard_generator_project/remove_logo_and_aug_image3/train/' +
                            image_name +'_'+'0'+ '_' + str(i) + '.jpg', train_data)

            for i in range(1):
                l = len(files)
                index = np.random.randint(0, l)
                image_gen_copy = img_res_b.copy()
                mask = cv2.imread(os.path.join(path + 'fuzhiwuxiao_mask', files[index]), 0)
                mask[mask > 150] = 255
                mask[mask <= 150] = 0
                image = cv2.imread(os.path.join(path + 'fuzhiwuxiao_template', files[index]), 0)
                template = cv2.bitwise_and(image, image, mask=mask)

                h_image_gen, w_image_gen = img_res_b.shape
                point_x, point_y = h_image_gen, w_image_gen
                h, w = template.shape
                while point_x + h >= h_image_gen or point_y + w >= w_image_gen:
                    l_point = len(point_select)
                    l_index = np.random.randint(0, l_point)
                    point_x, point_y = point_select[l_index]

                rect = image_gen_copy[point_x:(h + point_x), point_y:(w + point_y)]
                rect1 = cv2.bitwise_and(rect, rect, mask=mask)
                mask = cv2.bitwise_not(mask)
                rect2 = cv2.bitwise_and(rect, rect, mask=mask)
                image_with_logo = cv2.addWeighted(rect1, 0.3, template, 0.3, 0)
                add_logo = image_with_logo + rect2
                image_gen_copy_logo = image_gen_copy.copy()
                image_gen_copy_logo[point_x:(h + point_x), point_y:(w + point_y)] = add_logo

                image_gen_copy_logo=np.stack([image_gen_copy_logo,image_gen_copy_logo,image_gen_copy_logo],axis=2)
                aug_brightness_deterministic= aug_brightness.to_deterministic()
                aug_gaussian_deterministic=aug_gaussian.to_deterministic()
                image_gen_copy_logo=aug_brightness_deterministic.augment_images(images=[image_gen_copy_logo])[0]
                image_gen_copy_logo=aug_gaussian_deterministic.augment_images(images=[image_gen_copy_logo])[0]

                image_gen_copy = np.stack([image_gen_copy, image_gen_copy, image_gen_copy], axis=2)
                image_gen_copy = aug_brightness_deterministic.augment_images(images=[image_gen_copy])[0]
                #image_gen_copy = aug_gaussian_deterministic.augment_images(images=[image_gen_copy])[0]

                train_data = np.hstack((image_gen_copy_logo,image_gen_copy))

                # cv2.imshow('template', template)
                # cv2.imshow('logo',image_with_logo)
                # cv2.imshow('image_gen',image_gen_copy)
                # #cv2.imshow('rect',rect2)
                # # cv2.imshow('image', train_data)
                # cv2.waitKey(300000)
                cv2.imwrite('/home/ubuntu/hks/ocr/idcard_generator_project/remove_logo_and_aug_image3/train/' +
                            image_name+'_'+'1'+ '_' + str(i) + '.jpg', train_data)
                del image_gen_copy_logo
    for one_line in csv_reader_lines_val:
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

        result_back.append(date[cnt][9])  # 签发机关
        result_back.append(date[cnt][10])  # 有效日期

        image1 = cv2.imread(front_img)  # 读取正面模板
        image2 = cv2.imread(back_img)  # 读取背面模板

        #img_new_white1 = img_to_white(image1)
        img_new_white1 = image1
        # cv2.imshow('hjs',img_new_white1)
        # cv2.waitKey(0)# 生成画布
        img_res_f = gen_card_front(img_new_white1, result_front)
        img_res_f = cv2.cvtColor(img_res_f,cv2.COLOR_BGR2GRAY)
        # 写入文字
        #cv2.imwrite(result_card_path + '/{}_1.jpg'.format(image_name), img_res_f)

        #img_new_white2 = img_to_white(image2)
        img_new_white2 = image2
        img_res_b = gen_card_back(img_new_white2, result_back)
        img_res_b = cv2.cvtColor(img_res_b, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite(result_card_path + '/{}_0.jpg'.format(image_name), img_res_b)
        cnt = cnt + 1
        print(cnt)

        for i in range(4):
            l = len(files)
            index = np.random.randint(0, l)

            image_gen_copy = img_res_f.copy()
            mask = cv2.imread(os.path.join(path + 'fuzhiwuxiao_mask', files[index]), 0)
            mask[mask > 150] = 255
            mask[mask <= 150] = 0
            image = cv2.imread(os.path.join(path + 'fuzhiwuxiao_template', files[index]), 0)
            template = cv2.bitwise_and(image, image, mask=mask)

            h_image_gen, w_image_gen= img_res_f.shape
            point_x, point_y = h_image_gen, w_image_gen
            h, w = template.shape
            while point_x + h >= h_image_gen or point_y + w >= w_image_gen:
                point_x, point_y = np.random.randint(0, h_image_gen), np.random.randint(0, w_image_gen)

            rect = image_gen_copy[point_x:(h + point_x), point_y:(w + point_y)]
            rect1 = cv2.bitwise_and(rect, rect, mask=mask)
            mask = cv2.bitwise_not(mask)
            rect2 = cv2.bitwise_and(rect, rect, mask=mask)
            image_with_logo = cv2.addWeighted(rect1, 0.3, template, 0.3, 0)
            add_logo = image_with_logo + rect2

            image_gen_copy_logo = image_gen_copy.copy()
            image_gen_copy_logo[point_x:(h + point_x), point_y:(w + point_y)] = add_logo

            image_gen_copy_logo = np.stack([image_gen_copy_logo, image_gen_copy_logo, image_gen_copy_logo], axis=2)
            aug_brightness_deterministic = aug_brightness.to_deterministic()
            aug_gaussian_deterministic = aug_gaussian.to_deterministic()
            image_gen_copy_logo = aug_brightness_deterministic.augment_images(images=[image_gen_copy_logo])[0]
            image_gen_copy_logo = aug_gaussian_deterministic.augment_images(images=[image_gen_copy_logo])[0]

            image_gen_copy = np.stack([image_gen_copy, image_gen_copy, image_gen_copy], axis=2)
            image_gen_copy = aug_brightness_deterministic.augment_images(images=[image_gen_copy])[0]
            #image_gen_copy = aug_gaussian_deterministic.augment_images(images=[image_gen_copy])[0]
            #image_gen_copy=seq_nologo(image=np.stack([image_gen_copy,image_gen_copy,image_gen_copy],axis=2))

            train_data = np.hstack((image_gen_copy_logo,image_gen_copy))
            #image_gen_copy[point_x:(h + point_x), point_y:(w + point_y)] = add_logo[:,:]
            cv2.imwrite('/home/ubuntu/hks/ocr/idcard_generator_project/remove_logo_and_aug_image3/val/' +
                        image_name +'_'+'0'+ '_' + str(i) + '.jpg', train_data)

        for i in range(4):
            l = len(files)
            index = np.random.randint(0, l)
            image_gen_copy = img_res_b.copy()
            mask = cv2.imread(os.path.join(path + 'fuzhiwuxiao_mask', files[index]), 0)
            mask[mask > 150] = 255
            mask[mask <= 150] = 0
            image = cv2.imread(os.path.join(path + 'fuzhiwuxiao_template', files[index]), 0)
            template = cv2.bitwise_and(image, image, mask=mask)

            h_image_gen, w_image_gen = img_res_b.shape
            point_x, point_y = h_image_gen, w_image_gen
            h, w = template.shape
            while point_x + h >= h_image_gen or point_y + w >= w_image_gen:
                l_point = len(point_select)
                l_index = np.random.randint(0, l_point)
                point_x, point_y = point_select[l_index]

            rect = image_gen_copy[point_x:(h + point_x), point_y:(w + point_y)]
            rect1 = cv2.bitwise_and(rect, rect, mask=mask)
            mask = cv2.bitwise_not(mask)
            rect2 = cv2.bitwise_and(rect, rect, mask=mask)
            image_with_logo = cv2.addWeighted(rect1, 0.3, template, 0.3, 0)
            add_logo = image_with_logo + rect2
            image_gen_copy_logo = image_gen_copy.copy()
            image_gen_copy_logo[point_x:(h + point_x), point_y:(w + point_y)] = add_logo
            image_gen_copy_logo = np.stack([image_gen_copy_logo, image_gen_copy_logo, image_gen_copy_logo], axis=2)

            aug_brightness_deterministic = aug_brightness.to_deterministic()
            aug_gaussian_deterministic = aug_gaussian.to_deterministic()
            image_gen_copy_logo = aug_brightness_deterministic.augment_images(images=[image_gen_copy_logo])[0]
            image_gen_copy_logo = aug_gaussian_deterministic.augment_images(images=[image_gen_copy_logo])[0]

            image_gen_copy = np.stack([image_gen_copy, image_gen_copy, image_gen_copy], axis=2)
            image_gen_copy = aug_brightness_deterministic.augment_images(images=[image_gen_copy])[0]
            #image_gen_copy = aug_gaussian_deterministic.augment_images(images=[image_gen_copy])[0]
            train_data = np.hstack((image_gen_copy_logo,image_gen_copy))

            # cv2.imshow('template', template)
            # cv2.imshow('logo',image_with_logo)
            # cv2.imshow('image_gen',image_gen_copy)
            # #cv2.imshow('rect',rect2)
            # # cv2.imshow('image', train_data)
            # cv2.waitKey(300000)
            cv2.imwrite('/home/ubuntu/hks/ocr/idcard_generator_project/remove_logo_and_aug_image3/val/' +
                        image_name+'_'+'1'+ '_' + str(i) + '.jpg', train_data)
            del image_gen_copy_logo



if __name__ == "__main__":
    roi_mask = cv2.imread('/home/ubuntu/hks/ocr/idcard_generator_project/idcard_generator/背面.jpg', 0)
    roi_mask[roi_mask <= 150] = 0
    roi_mask[roi_mask > 150] = 255
    x, y = roi_mask.shape
    point_select = []
    for i in range(x):
        for j in range(y):
            if roi_mask[i, j] == 255:
                point_select.append([i, j])
    ori_path = os.path.abspath('.')
    front_img = os.path.join(ori_path, 'src/front.png')  # 正面模板
    back_img = os.path.join(ori_path, 'src/back.png')  # 背面模板
    result_card_path = os.path.join(ori_path, r'generator_datas1')  # 伪造原始身份证样本结果路径
    csv_files = ["Train_Labels.csv", "Train_Labels2.csv"]
    ori_csv_file = os.path.join(ori_path, r'src', "generate_labels1.csv")  # 原始数据
    gen_faker_card_run()
