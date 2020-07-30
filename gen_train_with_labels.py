import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt







path='/home/ubuntu/hks/ocr/idcard_generator_project/idcard_pix2pix/data_val/'
dirs=os.listdir(path)

for index,d in enumerate(dirs):
    os.mkdir('/home/ubuntu/hks/ocr/idcard_generator_project/idcard_pix2pix/data_val_with_aug/'+d)
    print(index)
    front = cv2.imread(path + d + '/' + d + '_0.jpg', 0)
    back = cv2.imread(path + d + '/' + d + '_1.jpg', 0)

    cv2.imwrite('/home/ubuntu/hks/ocr/idcard_generator_project/idcard_pix2pix/data_val_with_aug/'+d+'/' + d + '_0-0.jpg', front)
    cv2.imwrite('/home/ubuntu/hks/ocr/idcard_generator_project/idcard_pix2pix/data_val_with_aug/'+d+'/' + d + '_1-0.jpg', back)

    # var_front=getImageVar(front)
    # var_back=getImageVar(back)
    # l.append(var_back)
    # l.append(var_front)
    # if var_front>30:
    #     cv2.imshow('front',np.hstack([front,front1]))
    #     print('var_front:{}'.format(var_front))
    # if var_back>30:
    #     cv2.imshow('back',np.hstack([back,back1]))
    #     print('var_back:{}'.format(var_back))
    cv2.waitKey(300)

# print(min(l))
# plt.hist(l,bins=100,range=(0,400),)
# plt.show()
    # front_mask=cv2.imread(path+d+'/'+'mask-'+d+
    # back_mask = cv2.imread(path + d + '/' + 'mask-' + d + '_1.jpg',0)

    # cv2.imwrite('/home/ubuntu/hks/ocr/idcard_generator_project/idcard_pix2pix/data_val_with_aug/'+d+'/' + d + '_0-0.jpg', front)
    # cv2.imwrite('/home/ubuntu/hks/ocr/idcard_generator_project/idcard_pix2pix/data_val_with_aug/'+d+'/' + d + '_1-0.jpg', back)
    # cv2.imwrite('/home/ubuntu/hks/ocr/idcard_generator_project/idcard_pix2pix/data_val_with_aug/'+d+'/' + 'mask-'+d + '_0-0.jpg', front_mask)
    # cv2.imwrite('/home/ubuntu/hks/ocr/idcard_generator_project/idcard_pix2pix/data_val_with_aug/'+d+'/' + 'mask-'+d + '_1-0.jpg', back_mask)

    # kernel = np.ones((10, 10), np.uint8)
    # front_mask = cv2.erode(front_mask, kernel=np.ones((5, 5), np.uint8))
    # front_mask = cv2.dilate(front_mask, kernel=np.ones((10, 10), np.uint8))
    # contours, hierarchy = cv2.findContours(front_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    # ls = []
    # for contour in contours:
    #     l = cv2.contourArea(contour)
    #     ls.append(l)
    # index_max = np.argmax(ls)
    # x, y, w, h = cv2.boundingRect(contours[index_max])
    # # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
    # front_roi = front[y:y + h, x:x + w]
    #
    # #kernel = np.ones((10, 10), np.uint8)
    # back_mask = cv2.erode(back_mask, kernel=np.ones((5, 5), np.uint8))
    # back_mask = cv2.dilate(back_mask, kernel=np.ones((10, 10), np.uint8))
    # contours, hierarchy = cv2.findContours(back_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    # ls = []
    # for contour in contours:
    #     l = cv2.contourArea(contour)
    #     ls.append(l)
    # index_max = np.argmax(ls)
    # x, y, w, h = cv2.boundingRect(contours[index_max])
    # # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
    # back_roi = back[y:y + h, x:x + w]
    #
    # cv2.imwrite('/home/ubuntu/hks/ocr/idcard_generator_project/logo_data/'+d+'_0.jpg',front_roi)
    # cv2.imwrite('/home/ubuntu/hks/ocr/idcard_generator_project/logo_data/' + d + '_1.jpg', back_roi)
