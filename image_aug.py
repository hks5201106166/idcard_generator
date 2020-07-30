#-*-coding:utf-8-*-
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
# random example images
#-*-coding:utf-8-*-
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
files=os.listdir('/home/simple/mydemo/ocr_project/idcard_generator_project/gen_data_with_logo')
files_train,files_val=train_test_split(files,test_size=0.005, random_state=0)
seq = iaa.Sequential(
    [
        iaa.OneOf([
                    iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
                ]),
        iaa.MultiplyBrightness((0.5, 1.)),
    ],
    random_order=True)
for i,file in enumerate(files_val):
    print('gen val data {}'.format(i))
    image = cv2.imread('/home/simple/mydemo/ocr_project/idcard_generator_project/gen_data_with_logo/' + file)
    h,w,c=image.shape
    image_without_logo=image[:,0:int(w/2),:]
    image_with_logo=image[:,int(w/2):w,:]
    image_aug = seq(image=image_with_logo)
    img=np.hstack((image_aug,image_without_logo))
    cv2.imwrite('/home/simple/mydemo/ocr_project/idcard_generator_project/remove_logo_and_aug/val/'+file,img)
for i,file in enumerate(files_train):
    print('gen train data {}'.format(i))
    image = cv2.imread('/home/simple/mydemo/ocr_project/idcard_generator_project/gen_data_with_logo/' + file)
    h,w,c=image.shape
    image_without_logo=image[:,0:int(w/2),:]
    image_with_logo=image[:,int(w/2):w,:]
    image_aug = seq(image=image_with_logo)
    img=np.hstack((image_aug,image_without_logo))
    cv2.imwrite('/home/simple/mydemo/ocr_project/idcard_generator_project/remove_logo_and_aug/train/'+file,img)
    # cv2.imshow('hks',image_aug)
    # cv2.waitKey(0)
print()