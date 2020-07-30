import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt




# save to local


# read from local
# f = open("dict.txt", 'r')
# dict_ = eval(f.read())
# f.close()
# print("read from local : ", dict_)



def getImageVar(img):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    imagevar=cv2.Laplacian(img,cv2.CV_64F).var()
    return imagevar
aug_brightness = iaa.MultiplyBrightness((0.7, 1.1))
aug_gaussian =iaa.GaussianBlur((0, 2.0))
kernel_sharpen_1 = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]])
path='/home/ubuntu/hks/ocr/idcard_generator_project/idcard_pix2pix/data_train_with_aug/'
dirs=os.listdir(path)
d_dict={}
for d in dirs:
    d_dict[d+'_0']=0
    d_dict[d+'_1']=0
l=[]
for i in range(1,20):
    for index,d in enumerate(dirs):
        # os.mkdir('/home/ubuntu/hks/ocr/idcard_generator_project/idcard_pix2pix/data_val_with_aug/'+d)
        print(index)
        front=cv2.imread(path+d+'/'+d+'_0-0.jpg',0)
        back=cv2.imread(path+d+'/'+d+'_1-0.jpg',0)

        front1 = front.copy()
        front1= np.stack([front1,front1,front1],axis=2)
        back1 = back.copy()
        back1=np.stack([back1,back1,back1],axis=2)
        var_front1 = getImageVar(front1)
        var_back1 = getImageVar(back1)
        print('var_front:{}'.format(var_back1))
        if var_front1>40:
            front = np.stack([front, front, front], axis=2)
            front = aug_brightness.augment_images(images=[front])[0]
            front = aug_gaussian.augment_images(images=[front])[0]
            var_front = getImageVar(front)
            if var_front>25:
                d_dict[d+'_0']+=1
                cv2.imwrite('/home/ubuntu/hks/ocr/idcard_generator_project/idcard_pix2pix/data_train_with_aug/'+d+'/'+d+'_0-'+str(d_dict[d+'_0'])+'.jpg',front)
                # cv2.imshow('front',front)
                # output_1 = cv2.filter2D(front[:,:,0], -1, kernel_sharpen_1)
                # cv2.imshow('front', np.hstack([front,front1,np.stack([output_1,output_1,output_1],axis=2)]))

        if var_back1>40:
            back = np.stack([back, back, back], axis=2)
            back = aug_brightness.augment_images(images=[back])[0]
            back = aug_gaussian.augment_images(images=[back])[0]
            var_back = getImageVar(back)
            if var_back>25:
                d_dict[d+'_1']+=1
                cv2.imwrite('/home/ubuntu/hks/ocr/idcard_generator_project/idcard_pix2pix/data_train_with_aug/'+d+'/'+d +'_1-'+str(d_dict[d+'_1'])+'.jpg', back)
                # cv2.imshow('back',back)
                # output_2 = cv2.filter2D(back[:, :, 0], -1, kernel_sharpen_1)
                # cv2.imshow('back', np.hstack([back,back1, np.stack([output_2,output_2,output_2],axis=2)]))
                # print('var_back:{}'.format(var_back1))

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
        # cv2.waitKey(300)
f = open("dict.txt", 'w')
f.write(str(d_dict))
f.close()
print("save dict successfully.")
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
